import os
import re
import json
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Pydantic でバリデーションを行う
from pydantic import BaseModel, ValidationError


# --- 環境変数の読み込み ---
if os.getenv("ENV", "development") != "production":
    load_dotenv()

HOTPEPPER_API_KEY = os.getenv("HOTPEPPER_API_KEY")
if not HOTPEPPER_API_KEY:
    st.error("HOTPEPPER_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
client = OpenAI(api_key=openai_api_key)


# --- Pydanticモデル ---
# recommendation_reason を削除
class RestaurantModel(BaseModel):
    name: str
    address: str
    genre: str
    budget: str


# --- dataclass 定義 ---
@dataclass
class Restaurant:
    name: str
    address: str
    genre: str
    budget: str


# --- JSON パース用の関数 ---
def safe_parse_json(json_string: str) -> Any:
    """
    LLMが生成した文字列から、まず配列のJSON部分 [ ... ] を抜き出しパースを試みる。
    失敗した場合は { ... } をすべて抽出してパースしたリストを返す。
    """
    start = json_string.find('[')
    end = json_string.rfind(']')
    if start != -1 and end != -1 and end > start:
        array_str = json_string[start:end+1]
        try:
            return json.loads(array_str)
        except json.JSONDecodeError:
            pass

    # 配列抽出に失敗orデコードエラーの場合
    matches = re.findall(r'\{.*?\}', json_string, re.DOTALL)
    parsed_objects = []
    for match in matches:
        try:
            parsed_objects.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return parsed_objects


def parse_restaurants_to_dataclass(json_data: Any) -> List[Restaurant]:
    """
    Pydantic でバリデーションを行い、問題なければ dataclass に変換して返す。
    """
    if isinstance(json_data, dict):
        json_data = [json_data]
    elif not isinstance(json_data, list):
        return []

    valid_restaurants: List[Restaurant] = []
    for item in json_data:
        try:
            model = RestaurantModel(**item)
            valid_restaurants.append(
                Restaurant(
                    name=model.name,
                    address=model.address,
                    genre=model.genre,
                    budget=model.budget,
                )
            )
        except ValidationError:
            continue
    return valid_restaurants


# --- シングルクォートをダブルクォートに置換する「応急処置」の関数 ---
def fix_json_single_quotes(json_string: str) -> str:
    """
    単純な正規表現ベースで、キーや値のシングルクォートをダブルクォートに置換する。
    """
    # キー 'key': → "key":
    text = re.sub(r"(\s|{|,)'(\w+)':", r'\1"\2":', json_string)
    # 値 : 'value' → : "value"
    text = re.sub(r':\s*\'([^\']*)\'', r': "\1"', text)
    return text


# --- Level1: 基本検索 ---
def generate_restaurant_recommendations_level1(
    location: str,
    genre: str,
    menu: Optional[str] = "",
    budget: Optional[int] = 0
) -> List[Restaurant]:
    """
    Level1 の基本検索: 場所、ジャンル、（任意）メニュー、（任意）予算を条件として
    食事処のおすすめ情報をLLM経由で取得する。
    """
    user_prompt = (
        "以下の条件に合わせた食事処のおすすめ情報を、必ず JSON 配列のみで返してください。\n"
        "先頭や末尾に余計な文章を一切つけないでください。\n"
        "もし1件しかなくても、必ず配列の形にしてください。\n\n"
        f"- 場所: {location}\n"
        f"- ジャンル: {genre}\n"
    )
    if menu and menu.strip():
        user_prompt += f"- メニュー: {menu}\n"
    if budget and budget > 0:
        user_prompt += f"- 予算: 約{budget}円以下\n"
    user_prompt += (
        "各レストランオブジェクトは以下のキーを必ず含みます:\n"
        "name, address, genre, budget\n"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはプロのアシスタントです。必ずユーザーの指示に従って、"
                        "正しいJSON形式（配列）だけを出力してください。"
                    )
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        message_content = completion.choices[0].message.content

        # safe_parse_json
        json_data = safe_parse_json(message_content)
        restaurants = parse_restaurants_to_dataclass(json_data)

        # もしパースできなかったら、シングルクォート置き換えを試す
        if not restaurants:
            fixed_content = fix_json_single_quotes(message_content)
            json_data = safe_parse_json(fixed_content)
            restaurants = parse_restaurants_to_dataclass(json_data)

        if not restaurants:
            raise ValueError("抽出されたレストラン情報がありません。")
        return restaurants

    except Exception as e:
        st.error(f"LLM の結果取得に失敗しました: {e}")
        return [
            Restaurant(
                name="サンプル食堂",
                address=f"{location} サンプル町1-2-3",
                genre=genre,
                budget=f"{budget}円以下" if budget and budget > 0 else "不明",
            )
        ]


# --- AI によるトップ店舗選定関数 (LLM で上位店舗を選ぶ) ---
def select_top_restaurants(shops: List[dict], office: str, outing_genre: str) -> List[dict]:
    """
    与えられた店舗情報リストから、内定者バイト向けにおすすめの上位5店舗を
    AI（LLM）に選定させます。回答は必ずJSON配列形式のみで返してください。
    """
    user_prompt = (
        "以下の店舗情報リストがあります。これらの中から、内定者バイト向けに最適な上位5店舗を選んでください。\n"
        "回答は必ず JSON の配列のみで返してください。先頭や末尾に余計な文章をつけないでください。\n"
        "店舗情報:\n"
        + json.dumps(shops, ensure_ascii=False)
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはプロのアシスタントです。ユーザーの指示に従い、"
                        "必ず配列の JSON のみを出力してください。キーや文字列は必ずダブルクォートを使ってください。"
                    )
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        response_text = completion.choices[0].message.content.strip()

        # 配列 JSON 抽出
        start = response_text.find('[')
        end = response_text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
        else:
            json_str = response_text

        try:
            selected = json.loads(json_str)
        except json.JSONDecodeError:
            # シングルクォートをダブルクォートに置換して再トライ
            fixed_str = fix_json_single_quotes(json_str)
            selected = json.loads(fixed_str)

        if isinstance(selected, list):
            return selected
        else:
            return shops[:5]
    except Exception as e:
        st.error(f"トップ店舗選定に失敗しました: {e}")
        return shops[:5]


# --- Level2: 高度検索（ホットペッパーAPI呼び出し版） ---
def generate_restaurant_recommendations_level2(
    office: str,
    outing_genre: str,
    genre: Tuple[str, str],
    distance: int = 1000
) -> List[Restaurant]:
    """
    Level2（サイバー特化検索）の検索機能:
      - Hot Pepper API でお店を取得 → LLM でトップ5選定
      - レコメンド理由の文章は出さない
    """
    if outing_genre == "内定者バイトランチ":
        budget_code = "B001,B002"
        lunch_param = "1"
    else:
        budget_code = "B008,B003"
        lunch_param = None

    office_coordinates = {
        "渋谷スクランブルスクエア": {"lat": 35.65839321, "lng": 139.70230429},
        "abema towers": {"lat": 35.661, "lng": 139.710}
    }
    coords = office_coordinates.get(office, {"lat": 35.65839321, "lng": 139.70230429})

    distance_mapping = {300: 1, 500: 2, 1000: 3, 2000: 4, 3000: 5}
    range_val = distance_mapping.get(distance, 3)

    genre_code, genre_name = genre

    params = {
        "lat": coords["lat"],
        "lng": coords["lng"],
        "range": range_val,
        "format": "json",
        "key": HOTPEPPER_API_KEY,
        "genre": genre_code,
        "budget": budget_code,
    }
    if lunch_param:
        params["lunch"] = lunch_param

    endpoint = "http://webservice.recruit.co.jp/hotpepper/gourmet/v1/"
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        shops = data.get("results", {}).get("shop", [])

        # AIにより最適な店舗を上位5件に選定
        shops = select_top_restaurants(shops, office, outing_genre)

        restaurants: List[Restaurant] = []
        for shop in shops:
            name = shop.get("name", "不明")
            address = shop.get("address", "不明")
            shop_genre = shop.get("genre", {}).get("name", genre_name)
            shop_budget_raw = shop.get("budget", "不明")

            if isinstance(shop_budget_raw, dict):
                shop_budget = shop_budget_raw.get("name", "不明")
            else:
                shop_budget = shop_budget_raw

            restaurant = Restaurant(
                name=name,
                address=address,
                genre=shop_genre,
                budget=shop_budget,
            )
            restaurants.append(restaurant)

        if not restaurants:
            raise ValueError("抽出されたレストラン情報がありません。")
        return restaurants

    except Exception as e:
        st.error(f"ホットペッパーAPIの呼び出しに失敗しました: {e}")
        fallback = Restaurant(
            name="サンプル食堂",
            address=f"{office}周辺 サンプル町1-2-3",
            genre=genre_name,
            budget=budget_code,
        )
        return [fallback]


# --- メイン処理 ---
def main() -> None:
    st.set_page_config(page_title="食事処検索サイト", layout="wide")
    st.title("食事処検索サイト")
    st.write("LLM を活用して食事処を探します（レコメンド文章は表示しません）。")
    
    genre_options: List[Tuple[str, str]] = [
        ("G001", "居酒屋"),
        ("G002", "ダイニングバー・バル"),
        ("G003", "創作料理"),
        ("G004", "和食"),
        ("G005", "洋食"),
        ("G006", "イタリアン・フレンチ"),
        ("G007", "中華"),
        ("G008", "焼肉・ホルモン"),
        ("G017", "韓国料理"),
        ("G009", "アジア・エスニック料理"),
        ("G010", "各国料理"),
        ("G011", "カラオケ・パーティ"),
        ("G012", "バー・カクテル"),
        ("G013", "ラーメン"),
        ("G016", "お好み焼き・もんじゃ"),
        ("G014", "カフェ・スイーツ"),
        ("G015", "その他グルメ")
    ]
    
    with st.sidebar:
        st.header("検索条件")
        search_level = st.radio("検索レベル", options=["Level1（基本検索）", "Level2（サイバー特化検索）"])
        
        if search_level == "Level1（基本検索）":
            location = st.text_input("場所（都道府県 + 市区町村）", value="東京都渋谷区")
            genre_text = st.selectbox("ジャンル", options=["日本食", "中華", "イタリアン", "フレンチ", "アメリカン"])
            menu = st.text_input("メニュー（任意）", value="")
            budget = st.number_input("予算 (円)（任意）", min_value=0, max_value=20000, step=500, value=0)
        else:
            office = st.selectbox("オフィス", options=["渋谷スクランブルスクエア", "abema towers"], index=0)
            outing_genre = st.selectbox("お出かけジャンル", options=["内定者バイトランチ", "内定者バイト飲み"])
            selected_genre = st.selectbox("ジャンル", options=genre_options, format_func=lambda x: x[1])
            distance = st.selectbox("距離", options=[300, 500, 1000, 2000, 3000], index=2)
        
        search_button = st.button("検索")

    if search_button:
        if search_level == "Level1（基本検索）":
            display_text = f"{location} の {genre_text}"
            if menu and menu.strip():
                display_text += f"（メニュー: {menu}）"
            if budget and budget > 0:
                display_text += f"（予算: 約 {budget}円以下）"
            st.subheader(f"{display_text} のおすすめ食事処")
            restaurants = generate_restaurant_recommendations_level1(location, genre_text, menu, budget)
        else:
            display_text = f"{office} 周辺の {selected_genre[1]}"
            display_text += f"（{outing_genre}：一人あたり {'2000' if outing_genre=='内定者バイトランチ' else '5000'}円想定、距離: {distance}m以内）"
            st.subheader(f"{display_text} のおすすめ食事処")
            restaurants = generate_restaurant_recommendations_level2(office, outing_genre, selected_genre, distance)
        
        if restaurants:
            for restaurant in restaurants:
                st.markdown(f"### {restaurant.name}")
                st.write(f"**住所:** {restaurant.address}")
                st.write(f"**ジャンル:** {restaurant.genre}")
                st.write(f"**予算:** {restaurant.budget}")
                st.markdown("---")
        else:
            st.warning("条件に合う食事処が見つかりませんでした。")
        st.success("検索完了！")


if __name__ == "__main__":
    main()

import os
import re
import json
from dataclasses import dataclass
from typing import List, Any, Optional

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# --- 環境変数の読み込み ---
if os.getenv("ENV", "development") != "production":
    load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")

client = OpenAI(api_key=openai_api_key)

# --- データモデル ---
@dataclass
class Restaurant:
    name: str
    address: str
    genre: str
    budget: str  # 例: '1000円以下' など

# --- 補助関数 ---
def safe_parse_json(json_string: str) -> Any:
    """
    JSONDecodeError が発生した場合、正規表現で完全な JSON オブジェクト部分を抽出して返す。
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        matches = re.findall(r'\{.*?\}', json_string, re.DOTALL)
        parsed_objects = []
        for match in matches:
            try:
                parsed_objects.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        return parsed_objects

# --- Level1: 基本検索 ---
def generate_restaurant_recommendations_level1(
    location: str,
    genre: str,
    menu: Optional[str] = "",
    budget: Optional[int] = 0
) -> List[Restaurant]:
    """
    Level1 の基本検索: 場所、ジャンル、（任意）メニュー、（任意）予算を条件として食事処のおすすめ情報を取得する。
    """
    prompt = "以下の条件に合わせた食事処のおすすめ情報を、JSON形式で出力してください。\n"
    prompt += f"- 場所: {location}\n"
    prompt += f"- ジャンル: {genre}\n"
    if menu and menu.strip():
        prompt += f"- メニュー: {menu}\n"
    if budget and budget > 0:
        prompt += f"- 予算: 約{budget}円以下\n"
    prompt += "各レストランは 'name', 'address', 'genre', 'budget' のキーを持っていること。"

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        message_content = completion.choices[0].message.content
        try:
            restaurants_data = json.loads(message_content)
        except json.JSONDecodeError:
            restaurants_data = safe_parse_json(message_content)
        
        if not isinstance(restaurants_data, list):
            restaurants_data = [restaurants_data]
        restaurants = [
            Restaurant(**restaurant)
            for restaurant in restaurants_data
            if isinstance(restaurant, dict)
        ]
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

# --- Level2: 高度検索（新条件） ---
def generate_restaurant_recommendations_level2(
    office: str,
    outing_genre: str,
    genre: str,
    desired_food: Optional[str] = ""
) -> List[Restaurant]:
    """
    Level2 の高度検索:
      - オフィス: 「渋谷スクランブルスクエア」または「abema towers」から徒歩10分圏内
      - お出かけジャンル: 「内定者バイトランチ」または「内定者バイト飲み」
         → 内定者バイトランチの場合は一人あたり2000円、内定者バイト飲みの場合は一人あたり5000円
      - ジャンル: 「日本食」「中華」など
      - 食べたいもの: 任意（例: 寿司）
    
    これらの条件に合わせた食事処のおすすめ情報を取得する。
    """
    # 予算をお出かけジャンルから決定
    budget = 2000 if outing_genre == "内定者バイトランチ" else 5000

    prompt = "以下の条件に合わせた食事処のおすすめ情報を、JSON形式で出力してください。\n"
    prompt += f"- オフィス: {office}（徒歩10分圏内）\n"
    prompt += f"- お出かけジャンル: {outing_genre}\n"
    prompt += f"- 予算: 一人あたり {budget}円以下\n"
    prompt += f"- ジャンル: {genre}\n"
    if desired_food and desired_food.strip():
        prompt += f"- 食べたいもの: {desired_food}\n"
    prompt += "各レストランは 'name', 'address', 'genre', 'budget' のキーを持っていること。"

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250,
        )
        message_content = completion.choices[0].message.content
        try:
            restaurants_data = json.loads(message_content)
        except json.JSONDecodeError:
            restaurants_data = safe_parse_json(message_content)
        
        if not isinstance(restaurants_data, list):
            restaurants_data = [restaurants_data]
        restaurants = [
            Restaurant(**restaurant)
            for restaurant in restaurants_data
            if isinstance(restaurant, dict)
        ]
        if not restaurants:
            raise ValueError("抽出されたレストラン情報がありません。")
        return restaurants

    except Exception as e:
        st.error(f"LLM の結果取得に失敗しました: {e}")
        return [
            Restaurant(
                name="サンプル食堂",
                address=f"{office}周辺 サンプル町1-2-3",
                genre=genre,
                budget=f"{budget}円以下",
            )
        ]

# --- メイン処理 ---
def main() -> None:
    st.set_page_config(page_title="食事処検索サイト", layout="wide")
    st.title("食事処検索サイト")
    st.write("LLM を活用してあなたにぴったりの食事処を探します。")
    
    with st.sidebar:
        st.header("検索条件")
        # 検索レベルの選択
        search_level = st.radio("検索レベル", options=["Level1（基本検索）", "Level2（サイバー特化検索）"])
        
        if search_level == "Level1（基本検索）":
            # Level1 の入力フィールド
            location = st.text_input("場所（都道府県 + 市区町村）", value="東京都渋谷区")
            genre = st.selectbox("ジャンル", options=["日本食", "中華", "イタリアン", "フレンチ", "アメリカン"])
            menu = st.text_input("メニュー（任意）", value="")
            budget = st.number_input("予算 (円)（任意）", min_value=0, max_value=20000, step=500, value=0)
        else:
            # Level2 の入力フィールド（高度検索）
            office = st.selectbox("オフィス", options=["渋谷スクランブルスクエア", "abema towers"], index=0)
            outing_genre = st.selectbox("お出かけジャンル", options=["内定者バイトランチ", "内定者バイト飲み"])
            # 予算はお出かけジャンルから自動で決定（内定者バイトランチ:2000円、内定者バイト飲み:5000円）
            genre = st.selectbox("ジャンル", options=["日本食", "中華", "イタリアン", "フレンチ", "アメリカン"])
            desired_food = st.text_input("食べたいもの（任意）", value="")
        
        search_button = st.button("検索")
    
    if search_button:
        if search_level == "Level1（基本検索）":
            display_text = f"{location} の {genre}"
            if menu and menu.strip():
                display_text += f"（メニュー: {menu}）"
            if budget and budget > 0:
                display_text += f"（予算: 約 {budget}円以下）"
            st.subheader(f"{display_text} のおすすめ食事処")
            with st.spinner("おすすめの食事処を探索中..."):
                restaurants = generate_restaurant_recommendations_level1(location, genre, menu, budget)
        else:
            display_text = f"{office} の周辺の {genre}"
            display_text += f"（{outing_genre}：一人あたり {2000 if outing_genre=='内定者バイトランチ' else 5000}円以下）"
            if desired_food and desired_food.strip():
                display_text += f"（食べたいもの: {desired_food}）"
            st.subheader(f"{display_text} のおすすめ食事処")
            with st.spinner("おすすめの食事処を探索中..."):
                restaurants = generate_restaurant_recommendations_level2(office, outing_genre, genre, desired_food)
        
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

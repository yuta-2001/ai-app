import os
import re
import json
from dataclasses import dataclass
from typing import List, Any

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# 開発環境の場合、.env から環境変数をロード（本番では外部環境変数利用）
if os.getenv("ENV", "development") != "production":
    load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")

client = OpenAI(api_key=openai_api_key)


@dataclass
class Restaurant:
    name: str
    address: str
    genre: str
    budget: str  # 例: '1000円以下' のような文字列


def safe_parse_json(json_string: str) -> Any:
    """
    JSONDecodeError が発生した場合でも、正規表現で
    完全な JSON オブジェクト部分だけを抽出して返す補助関数。
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        matches = re.findall(r'\{.*?\}', json_string, re.DOTALL)
        parsed_objects = []
        for match in matches:
            try:
                obj = json.loads(match)
                parsed_objects.append(obj)
            except json.JSONDecodeError:
                continue
        return parsed_objects


def generate_restaurant_recommendations(genre: str, budget: int) -> List[Restaurant]:
    """
    OpenAI API を利用して、指定のジャンルと予算に合わせた食事処のおすすめ情報を取得します。

    パラメーター:
      - genre (str): 例 "日本食", "中華" など
      - budget (int): 予算の目安（円）

    戻り値:
      List[Restaurant]: レストラン情報のリスト（名前、住所、ジャンル、予算）
    """
    prompt = (
        f"以下の条件に合わせた食事処のおすすめ情報を、JSON形式で出力してください。\n"
        f"- ジャンル: {genre}\n"
        f"- 予算: 約{budget}円以下\n"
        f"各レストランは 'name', 'address', 'genre', 'budget' のキーを持っていること。"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        message_content = completion.choices[0].message.content
        # 直接 JSON を試みる。切れている場合は safe_parse_json により部分的な抽出を試みる
        restaurants_data = safe_parse_json(message_content)
        
        # restaurants_data がリストでない場合は、直接リストに変換する
        if not isinstance(restaurants_data, list):
            restaurants_data = [restaurants_data]
            
        restaurants = [Restaurant(**restaurant) for restaurant in restaurants_data if isinstance(restaurant, dict)]
        if not restaurants:
            raise ValueError("抽出されたレストラン情報が空です。")
        return restaurants

    except Exception as e:
        st.error(f"LLM の結果取得に失敗しました: {e}")
        # フォールバックのためのサンプルデータ
        return [
            Restaurant(
                name="サンプル食堂",
                address="東京都新宿区サンプル1-2-3",
                genre=genre,
                budget=f"{budget}円以下",
            )
        ]

def main() -> None:
    st.set_page_config(page_title="食事処検索サイト", layout="wide")
    st.title("おしゃれな食事処検索サイト")
    st.write("LLM を活用してあなたにぴったりの食事処を探します。")
    
    with st.sidebar:
        st.header("検索条件")
        genre = st.selectbox("ジャンル", options=["日本食", "中華", "イタリアン", "フレンチ", "アメリカン"])
        budget = st.number_input("予算 (円)", min_value=500, max_value=20000, step=500, value=1000)
        search_button = st.button("検索")
    
    if search_button:
        st.subheader(f"{genre}（予算約 {budget}円 以下）のおすすめ食事処")
        with st.spinner("おすすめの食事処を探索中..."):
            restaurants = generate_restaurant_recommendations(genre, budget)
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


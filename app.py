import os
import json
from dataclasses import dataclass
from typing import List, Any

import streamlit as st
import openai
from dotenv import load_dotenv

# 環境変数 "ENV" が "production" でない場合は .env ファイルを読み込みます。
if os.getenv("ENV", "development") != "production":
    load_dotenv()

# 環境変数から API キーを取得（プロダクションでは必ず外部から設定してください）
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")

@dataclass
class Restaurant:
    name: str
    address: str
    genre: str
    budget: str  # 例: '1000円' などの文字列形式


def generate_restaurant_recommendations(genre: str, budget: int) -> List[Restaurant]:
    """
    OpenAI API を利用して、指定のジャンルと予算に合わせた食事処のおすすめ情報を取得します。
    
    パラメータ:
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
        response: Any = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        message = response['choices'][0]['message']['content']
        # LLM の出力を JSON としてパース
        restaurants_data = json.loads(message)
        restaurants = [Restaurant(**restaurant) for restaurant in restaurants_data]
        return restaurants
    except Exception as e:
        st.error("LLM の結果取得に失敗しました。サンプルデータを表示します。")
        # フォールバックとしてサンプルデータを返す
        sample_restaurant = Restaurant(
            name="サンプル食堂",
            address="東京都新宿区サンプル1-2-3",
            genre=genre,
            budget=f"{budget}円"
        )
        return [sample_restaurant]


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

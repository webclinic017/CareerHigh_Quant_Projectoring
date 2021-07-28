import streamlit as st
import src.pages.introduction
import src.pages.cross_asset_momentum
import src.pages.single_asset_momentum

IMG_SRC = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/google/298/chart-increasing_1f4c8.png"

st.set_page_config(
    page_title = "Quant Projectoring",
    page_icon = IMG_SRC,
)

# Display header.
st.markdown("<br>", unsafe_allow_html=True)
st.image(IMG_SRC, width=80)

"""
# Quant Projectoring - Momentum Backtesting
[![Career High](https://img.shields.io/badge/-Career%20High-brightgreen)](https://https://cafe.naver.com/careerhighproject/)
&nbsp[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yeongkyu%20Kim-blue)](https://linkedin.com/in/yeongkyu-kim)
"""
st.markdown("<br>", unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

PAGES = {
    "Introduction": src.pages.introduction,
    "Cross-Asset Momentum Backtest": src.pages.cross_asset_momentum,
    "Single-Asset Momentum Backtest": src.pages.single_asset_momentum,
}

st.sidebar.title("Navigation")
selection = st.sidebar.selectbox('Select',list(PAGES.keys()))

page = PAGES[selection]

with st.spinner(f"Running {selection} ..."):
    page.app()
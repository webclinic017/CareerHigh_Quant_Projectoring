import streamlit as st
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply
import FinanceDataReader as fdr

from backtesting.test import SMA

def app():
    st.title('Single-Asset Momentum Backtesting')

    st.write('Backtest simple momentum strategies using `Backtesting.py` with daily historical price data')

    stock_dict = {
        "(Stock) Apple": "AAPL",
        "(Stock) Microsoft": "MSFT",
        "(Stock) Alphabet": "GOOG",
        "(Stock) Facebook": "FB",
        "(Stock) Samsung Electronics": "005930",
        "(Stock) SK Hynics": "000660",
        "(Stock) Naver": "035420",
        "(Stock) Kakao": "035720",
        "(Crypto) BTC/USD": "BTC/USD",
        "(Crypto) ETH/USD": "ETH/USD",
        "(Crypto) XRP/USD": "XRP/USD",
        "(FX) USD/EUR": "USD/EUR",
        "(FX) USD/JPY": "USD/JPY",
        "(FX) USD/KRW": "USD/KRW",
    }

    selected_stock_key = st.selectbox('Select price data to use in backtest', list(stock_dict.keys()))
    selected_stock_value = stock_dict[selected_stock_key]

    strategy_dict = {
        "Moving Average Crossover": SmaCross,
        "Relative Strength Index": RSIStrategy,
        "Bollinger Band": BBStrategy,
        "Donchain Channel": DonchainStrategy
    }

    # Select a Strategy
    selected_strategy_key = st.selectbox('Select a strategy', list(strategy_dict.keys()))
    selected_strategy = strategy_dict[selected_strategy_key]

    # Set Strategy Parameters
    params = dict()
    if selected_strategy_key == "Moving Average Crossover":
        short_term = st.number_input("Set Short-term Moving Average Lookback Period", value=10)
        long_term = st.number_input("Set Long-term Moving Average Lookback Period", value=20)
        params['short_term'] = short_term
        params['long_term'] = long_term

    elif selected_strategy_key == "Relative Strength Index":
        lookback_period = st.number_input("Set RSI Lookback Period", value=14)
        buy_level = st.number_input("Set RSI Buy Level", value=50)
        sell_level = st.number_input("Set RSI Sell Level", value=50)
        params['lookback_period'] = lookback_period
        params['buy_level'] = buy_level
        params['sell_level'] = sell_level

    elif selected_strategy_key == "Bollinger Band":
        lookback_period = st.number_input("Set Bollinger Band Lookback Period", value=20)
        params['lookback_period'] = lookback_period

    elif selected_strategy_key == "Donchain Channel":
        lookback_period = st.number_input("Set Donchain Channel Lookback Period", value=100)
        params['lookback_period'] = lookback_period

    # Set Transaction Cost (%)
    cost = st.number_input("Set Transaction Cost (%)", value=0.1) * 0.01

    if st.button('Execute backtest'):
        # 데이터 로드
        price_df = fdr.DataReader(selected_stock_value, '2015-01-01')

        # 백테스트 진행
        bt = Backtest(price_df, selected_strategy,
                      cash=1000000, commission=cost,
                      exclusive_orders=True)

        output = bt.run(**params)
        output_df = pd.DataFrame(output)
        st.dataframe(output_df[:-2], height=800)
        bt.plot(open_browser=False, filename="backtest_plot")
        with open("backtest_plot.html", "r", encoding='utf-8') as f:
            plot_html = f.read()
        st.components.v1.html(plot_html, height=1000)

class SmaCross(Strategy):

    short_term = 10
    long_term = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.short_term)
        self.sma2 = self.I(SMA, close, self.long_term)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

class RSIStrategy(Strategy):

    lookback_period = 14
    buy_level = 50
    sell_level = 50

    def init(self, lookback_period=14):
        # Compute moving averages the strategy demands
        self.ma10 = self.I(SMA, self.data.Close, 10)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)
        self.ma100 = self.I(SMA, self.data.Close, 100)

        # Compute daily RSI
        self.daily_rsi = self.I(RSI, self.data.Close, self.lookback_period)

    def next(self):
        price = self.data.Close[-1]

        # If we don't already have a position, and
        # if all conditions are satisfied, enter long.
        if self.daily_rsi[-1] > self.buy_level and not self.position.is_long:
            self.buy()

        elif self.daily_rsi[-1] < self.sell_level and not self.position.is_short:
            self.sell()

def BB(array, n, is_upper):
    sma = pd.Series(array).rolling(n).mean()
    std = pd.Series(array).rolling(n).std()
    upper_bb = sma + std * 1
    lower_bb = sma - std * 1
    if is_upper:
        return upper_bb
    else:
        return lower_bb

class BBStrategy(Strategy):

    lookback_period = 20

    def init(self):
        # Compute daily Bollinger Band
        self.upper_bb = self.I(BB, self.data.Close, self.lookback_period, True)
        self.lower_bb = self.I(BB, self.data.Close, self.lookback_period, False)

    def next(self):
        price = self.data.Close[-1]

        if self.upper_bb[-1] < price and not self.position.is_long:
            self.buy()

        elif self.lower_bb[-1] > price and not self.position.is_short:
            self.sell()

def Donchain(array, n, is_upper):
    rolling_max = pd.Series(array).rolling(n).max()
    rolling_min = pd.Series(array).rolling(n).min()
    if is_upper:
        return rolling_max
    else:
        return rolling_min

class DonchainStrategy(Strategy):

    lookback_period = 100

    def init(self):
        # Compute Donchain Channel
        self.upper_dc = self.I(Donchain, self.data.Close, self.lookback_period, True)
        self.lower_dc = self.I(Donchain, self.data.Close, self.lookback_period, False)

    def next(self):
        price = self.data.Close[-1]

        if self.upper_dc[-1] <= price and not self.position.is_long:
            self.buy()
        elif self.lower_dc[-1] >= price and not self.position.is_short:
            self.sell()
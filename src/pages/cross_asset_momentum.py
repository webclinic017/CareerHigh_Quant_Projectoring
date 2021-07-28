import streamlit as st

import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib
import matplotlib.pyplot as plt

def app():
    st.title('Cross-Asset Momentum Backtesting')

    st.write('퀀트대디님의 `CrossAssetMomentum` 구현체를 Streamlit GUI로 Wrapping하여 백테스트')

    url = 'https://raw.githubusercontent.com/davidkim0523/Momentum/main/Data.csv'

    # Set Lookback Period
    lookback_period = st.number_input("Set Lookback Period", value=120)

    # Set Holding Period
    holding_period = st.number_input("Set Holding Period", value=20)

    # Set Number of Stocks to Select
    n_selection = st.number_input("Set Number of Stocks to Select", value=19)

    # Select Momentum Strategy
    strategy_dict = {'Absolute Momentum': 'am', 'Relative Momentum': 'rm', 'Dual Momentum': 'dm'}
    selected_strategy = st.radio("Select Momentum Strategy", list(strategy_dict.keys()))
    signal_method = strategy_dict[selected_strategy]

    # Select Cross-Sectional Risk Model
    cs_dict = {'Equal Weights': 'ew', 'Equal Marginal Volatility': 'emv'}
    selected_cs_risk_model = st.radio("Select Cross-Sectional Risk Model", list(cs_dict.keys()))
    cs_weighting = cs_dict[selected_cs_risk_model]

    # Select Time-Series Risk Model
    ts_dict = {'Volatility Targeting': 'vt'}
    selected_ts_risk_model = st.radio("Select Time-Series Risk Model", list(ts_dict.keys()))
    ts_weighting = ts_dict[selected_ts_risk_model]

    if ts_weighting == 'vt':
        target_vol = st.number_input("Set Target Volatility", value=0.01)

    # Set Transaction Cost (%)
    cost = st.number_input("Set Transaction Cost (%)", value=0.1) * 0.01

    # Set Long-only Option
    long_only = st.checkbox('Set Long-only Option')

    if (st.button('Execute backtest')):
        prices = get_price_df(url)
        momentum = CrossAssetMomentum(prices, lookback_period, holding_period, n_selection, cost, signal_method,
                     cs_weighting, long_only, target_vol)

        # Jupyter notebook 밖에서 백테스트 통계량 table 출력하기가 어려워서, 우회적으로 직접 호출
        date_dict = momentum.get_perf_date_dict()
        st.dataframe(date_dict)
        perf_stats = momentum.get_perf_stats_table()
        st.dataframe(perf_stats, height=1000)
        drawdown_df = momentum.get_drawdown_table()
        st.dataframe(drawdown_df)


        fig = momentum.performance_analytics()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

class CrossAssetMomentum():
    def __init__(self, prices, lookback_period, holding_period, n_selection, cost=0.001, signal_method='dm',
                 weightings='emv', long_only=False, target_vol=0.01):
        self.returns = self.get_returns(prices)
        self.holding_returns = self.get_holding_returns(prices, holding_period)

        if signal_method == 'am':
            self.signal = self.absolute_momentum(prices, lookback_period, long_only)
        elif signal_method == 'rm':
            self.signal = self.relative_momentum(prices, lookback_period, n_selection, long_only)
        elif signal_method == 'dm':
            self.signal = self.dual_momentum(prices, lookback_period, n_selection, long_only)

        if weightings == 'ew':
            self.cs_risk_weight = self.equal_weight(self.signal)
        elif weightings == 'emv':
            self.cs_risk_weight = self.equal_marginal_volatility(self.returns, self.signal)

        self.rebalance_weight = 1 / holding_period
        self.cost = self.transaction_cost(self.signal, cost)

        self.port_rets_wo_cash = self.backtest(self.holding_returns, self.signal, self.cost, self.rebalance_weight,
                                               self.cs_risk_weight)

        self.ts_risk_weight = self.volatility_targeting(self.port_rets_wo_cash, target_vol)

        self.port_rets = self.port_rets_wo_cash * self.ts_risk_weight

    def get_returns(self, prices):
        """Returns the historical daily returns

        Paramters
        ---------
        prices : dataframe
            Historical daily prices

        Returns
        -------
        returns : dataframe
            Historical daily returns
        """
        returns = prices.pct_change().fillna(0)
        return returns

    def get_holding_returns(self, prices, holding_period):
        """Returns the periodic returns for each holding period

        Paramters
        ---------
        returns : dataframe
            Historical daily returns
        holding_period : int
            Holding Period

        Returns
        -------
        holding_returns : dataframe
            Periodic returns for each holding period. Pulled by N (holding_period) days forward to keep inline with trading signals.
        """
        holding_returns = prices.pct_change(periods=holding_period).shift(-holding_period).fillna(0)
        return holding_returns

    def absolute_momentum(self, prices, lookback, long_only=False):
        """Returns Absolute Momentum Signals

        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        lookback : int
            Lookback window for signal generation
        long_only : bool, optional
            Indicator for long-only momentum, False is default value

        Returns
        -------
        returns : dataframe
            Absolute momentum signals
        """
        returns = prices.pct_change(periods=lookback).fillna(0)
        long_signal = (returns > 0).applymap(self.bool_converter)
        short_signal = -(returns < 0).applymap(self.bool_converter)
        if long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        return signal

    def relative_momentum(self, prices, lookback, n_selection, long_only=False):
        """Returns Relative Momentum Signals

        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        lookback : int
            Lookback Window for Signal Generation
        n_selection : int
            Number of asset to be traded at one side
        long_only : bool, optional
            Indicator for long-only momentum, False is default value

        Returns
        -------
        returns : dataframe
            Relative momentum signals
        """
        returns = prices.pct_change(periods=lookback).fillna(0)
        rank = returns.rank(axis=1, ascending=False)
        long_signal = (rank <= n_selection).applymap(self.bool_converter)
        short_signal = -(rank >= len(rank.columns) - n_selection + 1).applymap(self.bool_converter)
        if long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        return signal

    def dual_momentum(self, prices, lookback, n_selection, long_only=False):
        """Returns Dual Momentum Signals

        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        lookback : int
            Lookback Window for Signal Generation
        n_selection : int
            Number of asset to be traded at one side
        long_only : bool, optional
            Indicator for long-only momentum, False is default value

        Returns
        -------
        returns : dataframe
            Dual momentum signals
        """
        abs_signal = self.absolute_momentum(prices, lookback, long_only)
        rel_signal = self.relative_momentum(prices, lookback, n_selection, long_only)
        signal = (abs_signal == rel_signal).applymap(self.bool_converter) * abs_signal
        return signal

    def equal_weight(self, signal):
        """Returns Equal Weights

        Parameters
        ----------
        signal : dataframe
            Momentum signal dataframe

        Returns
        -------
        weight : dataframe
            Equal weights for cross-asset momentum portfolio
        """
        total_signal = 1 / abs(signal).sum(axis=1)
        total_signal.replace([np.inf, -np.inf], 0, inplace=True)
        weight = pd.DataFrame(index=signal.index, columns=signal.columns).fillna(value=1)
        weight = weight.mul(total_signal, axis=0)
        return weight

    def equal_marginal_volatility(self, returns, signal):
        """Returns Equal Marginal Volatility (Inverse Volatility)

        Parameters
        ----------
        returns : dataframe
            Historical daily returns
        signal : dataframe
            Momentum signal dataframe

        Returns
        -------
        weight : dataframe
            Weights using equal marginal volatility

        """
        vol = (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        vol_signal = vol * abs(signal)
        inv_vol = 1 / vol_signal
        inv_vol.replace([np.inf, -np.inf], 0, inplace=True)
        weight = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
        return weight

    def volatility_targeting(self, returns, target_vol=0.01):
        """Returns Weights based on Vol Target

        Parameters
        ----------
        returns : dataframe
            Historical daily returns of backtested portfolio
        target_vol : float, optional
            Target volatility, Default target volatility is 1%

        Returns
        -------
        weights : dataframe
            Weights using equal marginal volatility

        """
        weight = target_vol / (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        weight.replace([np.inf, -np.inf], 0, inplace=True)
        weight = weight.shift(1).fillna(0)
        return weight

    def transaction_cost(self, signal, cost=0.001):
        """Returns Transaction Costs

        Parameters
        ----------
        signal : dataframe
            Momentum signal dataframe
        cost : float, optional
            Transaction cost (%) per each trade. The default is 0.001.

        Returns
        -------
        cost_df : dataframe
            Transaction cost dataframe

        """
        cost_df = (signal.diff() != 0).applymap(self.bool_converter) * cost
        cost_df.iloc[0] = 0
        return cost_df

    def backtest(self, returns, signal, cost, rebalance_weight, weighting):
        """Returns Portfolio Returns without Time-Series Risk Weights

        Parameters
        ----------
        returns : dataframe
            Historical daily returns
        signal : dataframe
            Momentum signal dataframe
        cost : dataframe
            Transaction cost dataframe
        rebalance_weight : float
            Rebalance weight
        weighting : dataframe
            Weighting dataframe

        Returns
        -------
        port_rets : dataframe
            Portfolio returns dataframe without applying time-series risk model

        """
        port_rets = ((signal * returns - cost) * rebalance_weight * weighting).sum(axis=1)
        return port_rets

    def performance_analytics(self):
        """Returns Perforamnce Analytics using pyfolio package

        Parameters
        ----------
        returns : series
            backtestd portfolio returns

        Returns
        -------
        None

        """
        return pf.create_returns_tear_sheet(self.port_rets, return_fig=True)

    def get_perf_date_dict(self):

        date_dict = dict()
        date_dict['Start date'] = self.port_rets.index[0].strftime('%Y-%m-%d')
        date_dict['End date'] = self.port_rets.index[-1].strftime('%Y-%m-%d')

        date_dict['Total months'] = int(len(self.port_rets) / 21)

        return pd.DataFrame([date_dict])

    def get_perf_stats_table(self):

        STAT_FUNCS_PCT = [
            'Annual return',
            'Cumulative returns',
            'Annual volatility',
            'Max drawdown',
            'Daily value at risk',
            'Daily turnover'
        ]

        # perf_stats 추출
        perf_stats = pd.DataFrame(pf.timeseries.perf_stats(self.port_rets), columns=['Backtest'])

        # perf_stats 전처리
        for column in perf_stats.columns:
            for stat, value in perf_stats[column].iteritems():
                if stat in STAT_FUNCS_PCT:
                    perf_stats.loc[stat, column] = str(np.round(value * 100, 3)) + '%'

        return perf_stats

    def get_drawdown_table(self, top=5):
        """Returns Drawdown Table from Portfolio Returns

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        top : int, optional
            The amount of top drawdowns to find (default 10).

        Returns
        -------
        df_drawdowns : pd.DataFrame
        Information about top drawdowns.

        """
        return pf.timeseries.gen_drawdown_table(self.port_rets, top=top)

    def bool_converter(self, bool_var):
        """Returns Integer Value from Boolean Value

        Parameters
        ----------
        bool_var : boolean
            Boolean variables representing trade signals

        Returns
        -------
        result : int
            Integer variables representing trade signals

        """
        if bool_var == True:
            result = 1
        elif bool_var == False:
            result = 0
        return result


def get_price_df(url):
    """Returns price dataframe from given URL

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).

    Returns
    -------
    df : dataframe
        Imported price dataframe from URL
    """
    df = pd.read_csv(url).dropna()
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Date'])
    return df


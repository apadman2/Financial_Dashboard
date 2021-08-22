from matplotlib import rcParams
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

class Forecast:
    def __init__(self, data, tp):
        if 'Close' not in data.columns:
            data['Close'] = data['Value']
        self.data = data['Close']
        self.dataframe = pd.DataFrame(self.data)
        split_length = int((tp/100)*len(self.dataframe))
        self.train = self.dataframe.iloc[:split_length+1,]
        self.test = self.dataframe.iloc[split_length:,]
        
    def Forecast_function(self, thing='regular'):
        timeseries = pd.Series(self.train['Close'])  
        log_timeseries = np.log(timeseries)
        if thing =='Split Graph':
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=self.dataframe.index, y=self.train['Close'],
                    mode='lines',
                    name='Training Values'))
            fig1.add_trace(go.Scatter(x=self.test.index, y=self.test['Close'],
                    mode='lines',
                    name='Testing Values'))
            fig1.update_layout(
                xaxis_title="Date",
                yaxis_title="Price($)"
            )
            fig1.update_layout(plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4))
            fig1.update_xaxes(showgrid=False)
            fig1.update_yaxes(showgrid=False)
            return fig1
        elif thing == 'Statistics':
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                                'P-value',
                                                'Number of Lags',
                                                'Number of Observations'])
            for key, value in dftest[4].items():
                dfoutput['Critical Value {%s}'%key] =value
            dfoutput = dfoutput.to_frame()
            return dfoutput
        elif thing == 'Rolling Graph':
            rolling1 = self.dataframe.copy()
            rolling1['Rolling Mean']= rolling1['Close'].rolling(12).mean()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=rolling1.index, y=rolling1['Close'],
                    mode='lines',
                    name='Rolling Mean'))
            fig1.add_trace(go.Scatter(x=rolling1.index, y=rolling1['Rolling Mean'],
                    mode='lines',
                    name='Rolling Mean'))
            fig1.update_layout(
                xaxis_title="Date",
                yaxis_title="Price($)"
            )
            fig1.update_layout(plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4))
            fig1.update_xaxes(showgrid=False)
            fig1.update_yaxes(showgrid=False)
            rolling2 = self.dataframe.copy()
            rolling2['Rolling Standard Deviation']= rolling2['Close'].rolling(12).std()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rolling2.index, 
                                    y=rolling2['Rolling Standard Deviation'], 
                                    fill='tozeroy',
                                    mode='lines'))
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="Standard Deviation"
            )
            fig2.update_layout(plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4))
            fig2.update_xaxes(showgrid=False)
            fig2.update_yaxes(showgrid=False)
            return fig1, fig2
        elif thing == 'ACF Graph':
            with plt.rc_context({'axes.edgecolor':'white',
                                'xtick.color':'white', 
                                'ytick.color':'white',
                                'text.color': 'white', 
                                'figure.facecolor':'#0e1117',
                                'axes.facecolor': '#0e1117'}):
                fig = sm.graphics.tsa.plot_acf(self.train, lags=40)
            return fig
        elif thing == 'PACF Graph':
            with plt.rc_context({'axes.edgecolor':'white',
                                'xtick.color':'white', 
                                'ytick.color':'white',
                                'text.color': 'white', 
                                'figure.facecolor':'#0e1117',
                                'axes.facecolor': '#0e1117'}):
                fig = sm.graphics.tsa.plot_pacf(self.train, lags=40)
            return fig
        elif thing =='Model Graph':
            model = pm.auto_arima(log_timeseries, trace=True, error_action='ignore', suppress_warnings=True)
            model.fit(log_timeseries)
            forecast = model.predict(len(self.test))
            forecast = np.exp(forecast)
            forecast = pd.DataFrame(forecast, index=self.test.index, columns=['Predicted Values'])
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=self.dataframe.index, y=self.train['Close'],
                    mode='lines',
                    name='Training Values'))
            fig1.add_trace(go.Scatter(x=self.test.index, y=self.test['Close'],
                    mode='lines',
                    name='Testing Values'))
            fig1.add_trace(go.Scatter(x=self.test.index, y=forecast['Predicted Values'],
                    mode='lines',
                    name='Predicted Values'))
            fig1.update_layout(
                xaxis_title="Date",
                yaxis_title="Price($)"
            )
            fig1.update_layout(plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4))
            fig1.update_xaxes(showgrid=False)
            fig1.update_yaxes(showgrid=False)
            return fig1

    def Model_function(self, p, d, q):
        history = list(y for y in self.train['Close'])
        predictions = []
        for t in range(len(self.test)):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = self.test['Close'][t]
            history.append(obs)
        results = self.test.copy()
        results['Predicted Values']=predictions
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=self.dataframe.index, y=self.train['Close'],
                mode='lines',
                name='Training Values'))
        fig1.add_trace(go.Scatter(x=self.test.index, y=results['Close'],
                mode='lines',
                name='Testing Values'))
        fig1.add_trace(go.Scatter(x=self.test.index, y=results['Predicted Values'],
                mode='lines',
                name='Predicted Values'))
        fig1.update_layout(
            xaxis_title="Date",
            yaxis_title="Price($)"
        )
        fig1.update_layout(plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font_color="white",
                    width=1100,
                    height=500,
                    margin=dict(l=50, r=50, b=100, t=100,pad=4))
        fig1.update_xaxes(showgrid=False)
        fig1.update_yaxes(showgrid=False)
        return fig1

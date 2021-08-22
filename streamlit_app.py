###################### Modules needed #########################
from attr import attr, attrs
import yfinance as yf
import streamlit as st
import streamlit.components as stc
st.set_page_config(layout="wide")
import datetime as dt
import pandas as pd
import numpy as np
import requests as rq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import bs4
import base64 
import time
from arima import Forecast
from rwb import RWB
from rsi import RSI
from kvo import KVO
timestr = time.strftime("%Y%m%d")

###################### Frontend #########################
def main():
    df = load_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Page: ", ["Stock Prediction", "Stock Back-testing","Upload your own CSV","Technical Analysis", "Fundamental Analysis",  "News Analysis"])

    if page == "Stock Prediction":
        resource = st.sidebar.selectbox("Method: ", ["ARIMA"])
        ticker = st.sidebar.text_input('Stock Symbol', value='AAPL')
        end = st.sidebar.date_input('End Date', value=dt.datetime.now())

        if resource == "ARIMA":
            st.title(str(page)+" using "+str(resource))
            # Yfinance
            tp = st.slider('Training percentage', value=80, min_value=10, max_value=90, step=5)
            x = yf.Ticker(str(ticker))
            yf_data = x.history(period='1d', start = end-dt.timedelta(365), end = end)
            yf_data = yf_data[['Close']]
            # Content
            arima_class = Forecast(yf_data, tp)
            st.header("Training and Testing Values")
            st.plotly_chart(arima_class.Forecast_function('Split Graph'))
            st.header("Training Data Statistics")
            temp = pd.DataFrame(arima_class.Forecast_function('Statistics'))
            st.dataframe(temp)
            st.header("Actual Price and Rolling Mean of Price")
            temp1, temp2 = arima_class.Forecast_function('Rolling Graph')
            st.plotly_chart(temp1)
            st.header("Rolling Standard Deviation")
            st.plotly_chart(temp2)
            for i in range(1):
                cols = st.columns(2)
                cols[0].header("ACF Plot")
                cols[1].header("PACF Plot")
            for i in range(1):
                cols = st.columns(2)
                cols[0].pyplot(arima_class.Forecast_function('ACF Graph'))
                cols[1].pyplot(arima_class.Forecast_function('PACF Graph'))
            st.header("Model Prediction - Rolling forecast and manual input of p, d and q")
            p = st.number_input('P', value = 1)
            d = st.number_input('D', value = 1)
            q = st.number_input('Q', value = 1)
            st.plotly_chart(arima_class.Model_function(p=p, d=d, q=q))
            st.header("Model Prediction - Auto ARIMA")
            st.plotly_chart(arima_class.Forecast_function('Model Graph'))
    elif page == "Stock Back-testing":
        backtest_ticker = st.sidebar.text_input('Stock Symbol', value='AAPL')
        start_ = st.sidebar.date_input('Backtesting End Date', value=dt.datetime.now()-dt.timedelta(365))
        end_ = st.sidebar.date_input('Backtesting Start Date', value=dt.datetime.now())
        performence_ = ['Number of Trades', 'Gain to Loss ratio',
            'Average Gain (%)', 'Average Loss (%)',
            'Maximum Return (%)', 'Maximum Loss (%)',
            'Total Return (%)']
        st.header(str(page))
        strat1 = RWB(str(backtest_ticker), start_, end_)
        strat2 = RSI(str(backtest_ticker), start_, end_)
        strat3 = KVO(str(backtest_ticker), start_, end_)
        x1, y1, z1 = strat1.analysis()
        x2, y2, z2 = strat2.calculator()
        x3, y3, z3 = strat3.kvo()
        for i in range(len(y1)):
            y1[i] = (y1[i]-1)*100
        for i in range(len(y2)):
            y2[i] = (y2[i]-1)*100
        for i in range(len(y3)):
            y3[i] = (y3[i]-1)*100
        st.subheader("Results:")
        backtest_results = pd.DataFrame({"RWB": x1, "RSI": x2, "KVO": x3}, index=performence_)
        st.dataframe(backtest_results)

        st.subheader("Returns:")
        backtest_return = pd.DataFrame({"RWB":y1, "RSI":y2, "KVO": y3}, index=z1)
        backtest_return["Date"]=backtest_return.index.to_series().astype(str)
        fig = px.line(backtest_return,
                    x="Date",
                    y=backtest_return.columns)
        fig.update_layout(plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4),
                        hovermode='closest',
                        xaxis ={'showspikes': True})
        fig.update_traces(mode='lines')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
 
    elif page == "Technical Analysis":
        technical_ticker = st.sidebar.text_input('Stock Symbol', value='AAPL')
        end = st.sidebar.date_input('End Date', value=dt.datetime.now())
        st.title(page)
        # Yfinance
        x = yf.Ticker(str(technical_ticker))
        yf_data = x.history(period='1d', start = end-dt.timedelta(365), end = end)
        yf_data = yf_data[['Open', 'High', 'Low','Close', 'Volume']]
        indicators = st.multiselect('Indicators:', ['SMA 5','SMA 10','SMA 20', 'Bollinger Bands'])
        # Plotting
        figure = technical_plot(yf_data, technical_ticker, indicators) 
        st.plotly_chart(figure)
        figure_volume = px.bar(yf_data, x=yf_data.index, y='Volume')
        figure_volume.update_layout(plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4))
        figure_volume.update_xaxes(showgrid=False)
        figure_volume.update_yaxes(showgrid=False)
        st.plotly_chart(figure_volume)

    elif page == "Fundamental Analysis":
        fundamental_ticker = st.sidebar.text_input('Stock Symbol', value='AAPL')
        fundamental_page = st.sidebar.selectbox("Resource: ", ["Company Overview", "Quarterly Earnings Report ($ Billion)", "Annual Balance Sheet ($ Billion)"])
        stock_info = yf.Ticker(str(fundamental_ticker))
        if fundamental_page == "Company Overview":
            st.title(fundamental_page)
            for i in range(1):
                cols = st.columns(9)
                cols[0].image(stock_info.info["logo_url"])
                cols[1].header(str(fundamental_ticker).upper())
            st.write("### Sector")
            st.write(stock_info.info["sector"])
            st.write("### Business Summary")
            st.write(stock_info.info["longBusinessSummary"])
        elif fundamental_page == "Quarterly Earnings Report ($ Billion)":
            st.title(str(fundamental_page))
            temp = pd.DataFrame.from_dict(stock_info.quarterly_financials)
            temp = temp.dropna()
            temp = temp/1000000000
            temp = pd.DataFrame(temp).round(2)
            temp.columns = pd.to_datetime(temp.columns).to_period('M')
            variables = st.multiselect('Earnings Report KPIs:', temp.index)
            temp_final = temp[temp.index.isin(variables)]
            fig = fundamental_plot(temp_final, variables)
            if len(temp_final) == 0:
                pass
            else:
                st.table(temp_final)
                csv_downloader(temp_final)
                st.plotly_chart(fig)
        elif fundamental_page == "Annual Balance Sheet ($ Billion)":
            st.title(str(fundamental_page))  
            temp1 = pd.DataFrame.from_dict(stock_info.balancesheet)
            temp1 = temp1.dropna()
            temp1 = temp1/1000000000
            temp1 = pd.DataFrame(temp1).round(2)
            temp1.columns = pd.to_datetime(temp1.columns).to_period('M')
            variables1 = st.multiselect('Balance Sheet KPIs:', temp1.index)
            temp1_final = temp1[temp1.index.isin(variables1)]
            fig = fundamental_plot(temp1_final, variables1)
            if len(temp1_final) == 0:
                pass
            else:
                st.table(temp1_final)
                csv_downloader(temp1_final)
                st.plotly_chart(fig)


    elif page == "News Analysis":
        stocknews_ticker = st.sidebar.text_input('Stock Symbol', value='AAPL')
        st.title(str(page))  
        nltk.download('stopwords')
        nltk.download('punkt')
        ################# GOOGLE NEWS
        st.subheader("Google News Word Cloud")
        request_result=rq.get("https://www.google.com/search?q="+str(stocknews_ticker).lower()+'%20stock&source=lnms&tbm=nws')
        soup = bs4.BeautifulSoup(request_result.text, "html.parser")
        final = ""
        for each in soup.find_all('div', attrs={'class':'kCrYT'}):
            title = each.find('div', attrs={"class":"BNeawe vvjwJb AP7Wnd"})
            temp = str(title).split('">')
            if len(temp)==1:
                pass
            else:
                temp = temp[1]
                temp = temp[:-6]
                final+=" "+temp
        stop_words1 = set(stopwords.words('english'))
        stop_words1.add(str(stocknews_ticker))
        word_tokens1 = word_tokenize(final)
        punc1 = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' 
        filtered_sentence = [w for w in word_tokens1 if not w.lower() in stop_words1]
        filtered_sentence = []
        for w in word_tokens1:
            if w not in stop_words1:
                filtered_sentence.append(w)
        unique_string=(" ").join(filtered_sentence)
        for ele in unique_string:
            if ele in punc1:
                unique_string = unique_string.replace(ele, "")
        # Generate wordcloud
        wordcloud = WordCloud(width = 2000, height = 2000, random_state=1, background_color='black', 
        colormap='Set2', collocations=False, stopwords = STOPWORDS, min_font_size=10, max_font_size=500).generate(unique_string)
        # Saving figure
        plot_cloud(wordcloud,2)
        st.image("wordcloud2.jpg")
        ################# STOCKTWEET
        st.subheader("Stocktweet Word Cloud")  
        req1 = rq.get("https://api.stocktwits.com/api/2/streams/symbol/"+str(stocknews_ticker)+".json")
        all_messages = ""
        for message in req1.json()['messages']:
            all_messages+= " "+str(message["body"]).lower()
        stop_words = set(stopwords.words('english'))
        stop_words.add(str(stocknews_ticker))
        word_tokens = word_tokenize(all_messages)
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' 
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        unique_string=(" ").join(filtered_sentence)
        for ele in unique_string:
            if ele in punc:
                unique_string = unique_string.replace(ele, "")
        # Generate wordcloud
        wordcloud = WordCloud(width = 2000, height = 2000, random_state=1, background_color='black', 
        colormap='Set2', collocations=False, stopwords = STOPWORDS, min_font_size=10, max_font_size=500).generate(unique_string)
        # Saving figure
        plot_cloud(wordcloud, 1)
        st.image("wordcloud1.jpg")

    elif page == "Upload your own CSV":
        st.title(page)
        resource = st.sidebar.selectbox("Method: ", ["ARIMA"])
        if resource == "ARIMA":
            st.subheader("How to format CSV")
            st.write('''Create a CSV file with the exact format mentioned in the image below. Make sure to have at least 100 rows of 
            data and ensure that the first column is formatted as a date. 
            ''')
            st.image("CSV format.PNG")
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                uploaded_df = pd.read_csv(uploaded_file, index_col=0)
                st.subheader("Data Sample")
                st.write(uploaded_df.head())
                st.title("Using "+str(resource))
                # Yfinance
                tp = st.slider('Training percentage', value=80, min_value=10, max_value=90, step=5)
                # Content
                arima_class = Forecast(uploaded_df, tp)
                st.header("Training and Testing Values")
                st.plotly_chart(arima_class.Forecast_function('Split Graph'))
                st.header("Training Data Statistics")
                temp = pd.DataFrame(arima_class.Forecast_function('Statistics'), columns=['Values'])
                st.dataframe(temp)
                st.header("Actual Price and Rolling Mean of Price")
                temp1, temp2 = arima_class.Forecast_function('Rolling Graph')
                st.plotly_chart(temp1)
                st.header("Rolling Standard Deviation")
                st.plotly_chart(temp2)
                for i in range(1):
                    cols = st.columns(2)
                    cols[0].header("ACF Plot")
                    cols[1].header("PACF Plot")
                for i in range(1):
                    cols = st.columns(2)
                    cols[0].pyplot(arima_class.Forecast_function('ACF Graph'))
                    cols[1].pyplot(arima_class.Forecast_function('PACF Graph'))
                st.header("Model Prediction - Rolling forecast and manual input of p, d and q")
                p = st.number_input('P', value = 1)
                d = st.number_input('D', value = 1)
                q = st.number_input('Q', value = 1)
                st.plotly_chart(arima_class.Model_function(p=p, d=d, q=q))
                st.header("Model Prediction - Auto ARIMA")
                st.plotly_chart(arima_class.Forecast_function('Model Graph'))

    st.sidebar.write("Made by *Aniruddh Padmanaban*")   

def plot_cloud(wordcloud, l):
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud) 
    plt.tight_layout(pad=0)
    plt.axis("off")
    if l==1:
        plt.savefig("wordcloud1.jpg", facecolor='black', transparent=True)
    else:
        plt.savefig("wordcloud2.jpg", facecolor='black', transparent=True)

def technical_plot(yf_data, technical_ticker, indicators):
    yf_data["SMA 5"] = yf_data.iloc[:,3].rolling(window=5).mean()
    yf_data["SMA 10"] = yf_data.iloc[:,3].rolling(window=10).mean()
    yf_data["SMA 20"] = yf_data.iloc[:,3].rolling(window=20).mean()
    figure = go.Figure(data=[go.Candlestick(x=yf_data.index,
                                            open=yf_data['Open'],
                                            high=yf_data['High'],
                                            low=yf_data['Low'],
                                            close=yf_data['Close'],)],
                        layout= {"title": {"text": str(technical_ticker).upper()+" over the last year"}})
    figure.update_layout(xaxis_rangeslider_visible=False,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color="white",
                        width=1100,
                        height=500,
                        margin=dict(l=50, r=50, b=100, t=100,pad=4))
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=False)
    if "SMA 5" in indicators:
        figure.add_trace(go.Scatter(x=yf_data.index,
                                    y=yf_data["SMA 5"],
                                    mode='lines',
                                    name='SMA 5',
                                    line=go.scatter.Line(color='rgb(51,400,255)')))
    if "SMA 10" in indicators:
        figure.add_trace(go.Scatter(x=yf_data.index,
                                    y=yf_data["SMA 10"],
                                    mode='lines',
                                    name='SMA 10',
                                    line=go.scatter.Line(color='rgb(51, 200, 255 )')))
    if "SMA 20" in indicators:
        figure.add_trace(go.Scatter(x=yf_data.index,
                                    y=yf_data["SMA 20"],
                                    mode='lines',
                                    name='SMA 20',
                                    line=go.scatter.Line(color='rgb(55, 50, 255)')))
    if "Bollinger Bands" in indicators:
        BB_period = st.slider("Period", min_value=10, max_value=30, step=5, value= 20)
        BB_n = st.slider("No. of Standard Deviations", min_value=1, max_value=3, step=1, value=2)
        yf_data["Upper Bollinger Band"], yf_data["Lower Bollinger Band"] = bb(yf_data["Close"], yf_data["SMA 20"], BB_period, BB_n)
        figure.add_trace(go.Scatter(x=yf_data.index,
                                    y=yf_data["Upper Bollinger Band"],
                                    mode='lines',
                                    name='BB Upper',
                                    line=dict(color='white', width=2, dash='dot')))
        figure.add_trace(go.Scatter(x=yf_data.index,
                                    y=yf_data["Lower Bollinger Band"],
                                    mode='lines',
                                    name='BB Lower',
                                    line=dict(color='white', width=2, dash='dot')))
    return figure 

def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "{}.csv".format(timestr)
	# st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download as CSV</a>'
	st.markdown(href,unsafe_allow_html=True)

def fundamental_plot(data, variables):
    plotting_data = data[data.index.isin(variables)]
    plotting_data =plotting_data.T
    plotting_data['Date']=plotting_data.index.to_series().astype(str)
    fig = px.line(plotting_data,
                 x='Date',
                 y=plotting_data.columns,
                 labels=dict(Date="Date", value="Value ($B)", variable="KPI"))
    fig.update_layout(plot_bgcolor='#0e1117',
                      paper_bgcolor='#0e1117',
                      font_color="white",
                      width=1100,
                      height=500,
                      margin=dict(l=50, r=50, b=100, t=100,pad=4),
                      hovermode='closest',
                      xaxis ={'showspikes': True})
    fig.update_traces(mode='markers+lines')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def bb(data, sma, window, n):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * n
    lower_bb = sma - std * n
    return upper_bb, lower_bb

@st.cache
def load_data():
    return 

if __name__ == "__main__":
    main()

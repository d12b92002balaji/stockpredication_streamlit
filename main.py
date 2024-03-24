import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.title("Stock Dashboard")

ticker = st.sidebar.text_input('Enter Ticker ')

# Specify the date range
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

try:
    data = yf.download(ticker, start=start_date, end=end_date)
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

if not data.empty:
    # Create line chart
    fig = px.line(data, x=data.index, y=data['Adj Close'], title=f'{ticker} Stock Price')
    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected date range or ticker.")

pricing_data, fundamental_data, news, about, technical, stockpredication = st.tabs(
    ["Pricing Data", "Fundamental Data", "Top 10 News", "ABOUT", "Techincal Analysis Dashboard",
     "Stock predication model"])

with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1)
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean() * 252 * 100
    st.write('Annual Return is ', annual_return, '%')
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.write('Standard Deviation', stdev * 100, '%')
    st.write('Risk Adj, Return is ', annual_return / (stdev * 100))

# from alpha_vantage.fundamentaldata import FundamentalData
#
with fundamental_data:
    st.title("Fundamental Analysis")

    # Define the ticker symbol input field
    # ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")

    # Fetch fundamental data using yfinance
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    income_statement = company.financials
    cash_flow_statement = company.cashflow

    # Display balance sheet
    st.subheader("Balance Sheet")
    st.write(balance_sheet)

    # Display income statement
    st.subheader("Income Statement")
    st.write(income_statement)

    # Display cash flow statement
    st.subheader("Cash Flow Statement")
    st.write(cash_flow_statement)
#     st.write("data")
#     key = '1OMGNLXLB2AOJWBM'
#     fd = FundamentalData(key, output_format='pandas')
#     st.subheader('balance sheet')
#     balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
#     bs = balance_sheet.T[2:]
#     st.columns = list(balance_sheet.T.iloc[0])
#     st.write(bs)
#     st.subheader('income statement')
#     income_statement = fd.get_income_statement_annual(ticker)[0]
#     is1 = income_statement.T[2:]
#     is1.columns = list(income_statement.T.iloc[0])
#     st.write(is1)
#     st.subheader('Cash flow statement')
#     cash_flow = fd.get_cash_flow_annual(ticker)[0]
#     cf = cash_flow.T[2:]
#     cf.columns = list(cash_flow.T.iloc[0])
#     st.write(cf)

from stocknews import StockNews

with news:
    st.write("stock news")
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News{i + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment{title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')



with about:

    about_data=yf.Ticker(ticker).info

    sector=about_data.get("sector")
    sector = about_data.get("sector")
    industry = about_data.get("industry")
    website = about_data.get("website")
    market_cap = about_data.get("marketCap")
    long_business_summary = about_data.get("longBusinessSummary")

    st.write(sector)
    st.write(industry)
    st.write(website)
    st.write(market_cap)
    st.write(long_business_summary)


import pandas_ta as ta

with technical:
    st.subheader('Technical Analysis Dashborad:')
    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list=True)
    # st.write(ind_list)
    technical_indicator = st.selectbox('Tech Indicator', options=ind_list)
    method = technical_indicator
    indicator = pd.DataFrame(getattr(ta, method)(low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume']))
    indicator['High'] = data['High']
    indicator['Low'] = data['Low']
    indicator['Open'] = data['Open']
    indicator['Close'] = data['Close']
    indicator['Volume'] = data['Volume']
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

with stockpredication:
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    stocks = ('GOOG', 'AAPL', 'MSFT', 'NVDA')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365


    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())


    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

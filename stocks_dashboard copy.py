import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# Title
st.title("📈 Stock Market Dashboard")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Calculate technical indicators
def calculate_indicators(data, indicators):
    if 'SMA 50' in indicators:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
    if 'SMA 200' in indicators:
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
    if 'EMA 20' in indicators:
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    if 'RSI' in indicators:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Plot stock data with indicators
def plot_stock_data(data, indicators):
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))

    # Add selected indicators
    if 'SMA 50' in indicators and 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1.5)
        ))
    if 'SMA 200' in indicators and 'SMA_200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_200'],
            name='SMA 200',
            line=dict(color='red', width=1.5)
        ))
    if 'EMA 20' in indicators and 'EMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA_20'],
            name='EMA 20',
            line=dict(color='green', width=1.5)
        ))
    
    # RSI (if selected, plot in a separate subplot)
    if 'RSI' in indicators and 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple', width=1.5),
            yaxis='y2'
        ))
        fig.update_layout(
            yaxis2=dict(
                title='RSI',
                overlaying='y',
                side='right',
                range=[0, 100]
            )
        )

    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# User inputs
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
indicators = st.sidebar.multiselect(
    "Technical Indicators",
    ['SMA 50', 'SMA 200', 'EMA 20', 'RSI'],
    default=['EMA 20']
)

# Fetch and display data
if st.sidebar.button("Fetch Data"):
    data = fetch_stock_data(ticker, start_date, end_date)
    if not data.empty:
        data = calculate_indicators(data, indicators)
        st.subheader(f"{ticker} Stock Data")
        st.dataframe(data.tail(10))  # Show last 10 rows
        plot_stock_data(data, indicators)
    else:
        st.error("Failed to fetch data. Check ticker symbol and date range.")

# Run with: `streamlit run stocks_dashboard.py`










# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# import yfinance as yf
# import numpy as np  # Added for array operations
# from datetime import datetime, timedelta
# import pytz
# import ta

# ##########################################################################################
# ## PART 1: Fixed Functions with Proper Array Handling ##
# ##########################################################################################

# def fetch_stock_data(ticker, period, interval):
#     """Fetch stock data with error handling"""
#     try:
#         end_date = datetime.now()
#         if period == '1wk':
#             start_date = end_date - timedelta(days=7)
#             data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
#         else:
#             data = yf.download(ticker, period=period, interval=interval)
#         return data
#     except Exception as e:
#         st.error(f"Error fetching data for {ticker}: {str(e)}")
#         return pd.DataFrame()

# def process_data(data):
#     """Process data with proper array conversion"""
#     if data.empty:
#         return data
        
#     if data.index.tzinfo is None:
#         data.index = data.index.tz_localize('UTC')
#     data.index = data.index.tz_convert('US/Eastern')
#     data.reset_index(inplace=True)
#     data.rename(columns={'Date': 'Datetime'}, inplace=True)
#     return data

# def add_technical_indicators(data):
#     """Add technical indicators with proper array flattening"""
#     if data.empty:
#         return data
        
#     # Convert to 1D numpy array if needed
#     if isinstance(data['Close'], pd.Series):
#         close_prices = data['Close'].values
#     else:
#         close_prices = np.array(data['Close']).flatten()
    
#     # Calculate indicators
#     data['SMA_20'] = ta.trend.sma_indicator(close_prices, window=20)
#     data['EMA_20'] = ta.trend.ema_indicator(close_prices, window=20)
    
#     return data

# def calculate_metrics(data):
#     """Calculate metrics with explicit scalar conversion"""
#     if data.empty:
#         return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    
#     # Explicit conversion to scalar values
#     last_close = float(data['Close'].iloc[-1])
#     prev_close = float(data['Close'].iloc[0])
#     change = float(last_close - prev_close)
#     pct_change = float((change / prev_close) * 100)
#     high = float(data['High'].max())
#     low = float(data['Low'].min())
#     volume = int(data['Volume'].sum())
    
#     return last_close, change, pct_change, high, low, volume

# ###############################################
# ## PART 2: Dashboard Implementation ##
# ###############################################

# # Set up Streamlit page layout
# st.set_page_config(layout="wide")
# st.title('Real Time Stock Dashboard')

# # Sidebar parameters
# st.sidebar.header('Chart Parameters')
# ticker = st.sidebar.text_input('Ticker', 'ADBE').strip().upper()
# time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
# chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
# indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])

# interval_mapping = {
#     '1d': '1m',
#     '1wk': '30m',
#     '1mo': '1d',
#     '1y': '1wk',
#     'max': '1wk'
# }

# if st.sidebar.button('Update'):
#     data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    
#     if not data.empty:
#         data = process_data(data)
#         data = add_technical_indicators(data)
        
#         last_close, change, pct_change, high, low, volume = calculate_metrics(data)
        
#         # Display metrics
#         st.metric(label=f"{ticker} Last Price", 
#                 value=f"{last_close:.2f} USD", 
#                 delta=f"{change:.2f} ({pct_change:.2f}%)")
        
#         col1, col2, col3 = st.columns(3)
#         col1.metric("High", f"{high:.2f} USD")
#         col2.metric("Low", f"{low:.2f} USD")
#         col3.metric("Volume", f"{volume:,}")
        
#         # Plotting code
#         fig = go.Figure()
#         if chart_type == 'Candlestick':
#             fig.add_trace(go.Candlestick(
#                 x=data['Datetime'],
#                 open=data['Open'],
#                 high=data['High'],
#                 low=data['Low'],
#                 close=data['Close']
#             ))
#         else:
#             fig = px.line(data, x='Datetime', y='Close')
        
#         # Add indicators if selected
#         if 'SMA 20' in indicators and 'SMA_20' in data.columns:
#             fig.add_trace(go.Scatter(
#                 x=data['Datetime'], 
#                 y=data['SMA_20'], 
#                 name='SMA 20',
#                 line=dict(color='orange', width=2)
#             )
        
#         if 'EMA 20' in indicators and 'EMA_20' in data.columns:
#             fig.add_trace(go.Scatter(
#                 x=data['Datetime'], 
#                 y=data['EMA_20'], 
#                 name='EMA 20',
#                 line=dict(color='green', width=2)
#             )
        
#         fig.update_layout(
#             title=f'{ticker} {time_period.upper()} Chart',
#             xaxis_title='Time',
#             yaxis_title='Price (USD)',
#             height=600,
#             hovermode='x unified'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Display data tables
#         st.subheader('Historical Data')
#         st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])
        
#         if any(col in data.columns for col in ['SMA_20', 'EMA_20']):
#             st.subheader('Technical Indicators')
#             st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']].dropna())
#     else:
#         st.warning(f"No data available for {ticker}")

# # Sidebar real-time prices
# st.sidebar.header('Real-Time Stock Prices')
# stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']

# for symbol in stock_symbols:
#     try:
#         real_time_data = fetch_stock_data(symbol, '1d', '1m')
#         if not real_time_data.empty:
#             real_time_data = process_data(real_time_data)
#             last_price = float(real_time_data['Close'].iloc[-1])
#             open_price = float(real_time_data['Open'].iloc[0])
#             change = float(last_price - open_price)
#             pct_change = float((change / open_price) * 100)
            
#             st.sidebar.metric(
#                 label=symbol,
#                 value=f"{last_price:.2f} USD",
#                 delta=f"{change:.2f} ({pct_change:.2f}%)"
#             )
#     except Exception as e:
#         st.sidebar.warning(f"Couldn't load {symbol} data")

# st.sidebar.subheader('About')
# st.sidebar.info('This dashboard provides properly formatted stock data with technical indicators.')
import streamlit as st
from ollama import chat
from pydantic import BaseModel
import pandas as pd
from gnews import GNews
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
import numpy as np
import ta
import pytz
import json
import threading
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import torch
import yfinance as yf
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly.subplots import make_subplots

# Check for GPU availability and set device preference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up the Streamlit page
st.set_page_config(
    page_title="Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and tabs
st.title("📈 Advanced Stock Analysis Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stock Analysis", "News Sentiment", "Combined View", "AI Advisor", "Technical Prediction"])

################################################################################
# Utility Functions
################################################################################

def safe_float_format(value, precision=2):
    """Safely convert and format a value to float with given precision"""
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"

################################################################################
# Stock Analysis Functions
################################################################################

def fetch_stock_data(ticker, start_date=None, end_date=None):
    """Fetch actual stock data using yfinance"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for ticker: {ticker}")
            return pd.DataFrame()
        
        if 'Close' not in df.columns:
            st.error(f"Missing required columns in data for {ticker}")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(data, indicators):
    if data.empty:
        return data
        
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

def plot_stock_data(data, indicators, ticker):
    if data.empty:
        return go.Figure()
        
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
    return fig

################################################################################
# News Sentiment Analysis Functions
################################################################################

class NewsAnalysis(BaseModel):
    sentiment: str  
    future_looking: bool  

def fetch_news(ticker, num_articles=5):
    google_news = GNews()
    try:
        news = google_news.get_news(ticker)
        if not news:
            st.warning(f"No news found for {ticker}")
            return None
                
        filtered_news = [
            article for article in news 
            if ticker.lower() in article['title'].lower() or 
               ticker.upper() in article['title'].upper()
        ][:num_articles]
        
        if not filtered_news:
            st.warning(f"No relevant news found for {ticker}")
            return None
                
        news_data = [{
            'Title': article['title'],
            'Date': article.get('published date', 'Unknown'),
            'Link': article.get('url', '#')
        } for article in filtered_news]
        
        return pd.DataFrame(news_data)
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return None

def zero_shot_analysis(news_df):
    results = []
    for _, row in news_df.iterrows():
        try:
            response = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': f"""Analyze this title for sentiment (positive, negative, neutral) 
                                       and whether it provides future-looking financial insight: {row['Title']}
                        """,
                    }
                ],
                model='llama3',
                format=NewsAnalysis.model_json_schema(),
            )
            sentiment_analysis = NewsAnalysis.model_validate_json(response['message']['content'])
            results.append({
                'Date': row['Date'],
                'Title': row['Title'],
                'Link': row['Link'],
                'Sentiment': sentiment_analysis.sentiment,
                'Future Looking': sentiment_analysis.future_looking
            })
        except Exception as e:
            st.warning(f"Error analyzing article: {row['Title']}. Error: {str(e)}")
            continue
    return pd.DataFrame(results)

################################################################################
# Technical Prediction Functions
################################################################################

def prepare_prediction_data(ticker, start_date=None, end_date=None):
    """Prepare data for technical prediction with lag structure"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()
    
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Create lag structure
    df['Previous_Close'] = df['Close'].shift(1)
    df['Close_shifted'] = df['Close'].shift(1)
    df['Open_shifted'] = df['Open'].shift(1)
    df['High_shifted'] = df['High'].shift(1)
    df['Low_shifted'] = df['Low'].shift(1)
    
    # Calculate technical indicators using ta
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close_shifted'], window=50).sma_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close_shifted'], window=50).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close_shifted'], window=14).rsi()
    
    macd = ta.trend.MACD(df['Close_shifted'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['Signal_Line'] = macd.macd_signal()
    
    bb = ta.volatility.BollingerBands(df['Close_shifted'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    
    stoch = ta.momentum.StochasticOscillator(high=df['High_shifted'], low=df['Low_shifted'], 
                                           close=df['Close_shifted'], window=14, smooth_window=3)
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High_shifted'], low=df['Low_shifted'], 
                                            close=df['Close_shifted'], window=14).average_true_range()
    
    return df.dropna()

def run_technical_prediction(df, window_size=20):
    """Run technical prediction with rolling window"""
    indicators = ['SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 
                 'BB_Upper', 'BB_Middle', 'BB_Lower', '%K', '%D', 'ATR', 
                 'Close_shifted', 'Previous_Close']
    
    results = {indicator: {'predictions': [], 'actual': [], 'daily_mae': []} 
               for indicator in indicators}
    
    for i in range(window_size, len(df) - 1):
        train_df = df.iloc[i - window_size:i]
        test_index = i + 1
        actual_close_price = df['Close'].iloc[test_index]

        for indicator in indicators[:-1]:
            X_train = train_df[[indicator, 'Previous_Close']]
            y_train = train_df['Close']
            X_train = sm.add_constant(X_train)

            model = sm.OLS(y_train, X_train).fit()
            X_test = pd.DataFrame({indicator: [df[indicator].iloc[test_index]], 
                                 'Previous_Close': [df['Previous_Close'].iloc[test_index]]})
            X_test = sm.add_constant(X_test, has_constant='add')

            prediction = model.predict(X_test)[0]
            results[indicator]['predictions'].append(prediction)
            results[indicator]['actual'].append(actual_close_price)
            daily_mae = mean_absolute_error([actual_close_price], [prediction])
            results[indicator]['daily_mae'].append(daily_mae)
    
    return results

def calculate_prediction_accuracy(results):
    """Calculate accuracy metrics for predictions"""
    accuracy_data = {
        'Indicator': [],
        'MAE': [],
        'MSE': []
    }

    for indicator in results.keys():
        if indicator != 'Previous_Close' and results[indicator]['actual']:
            mae = mean_absolute_error(results[indicator]['actual'], 
                                    results[indicator]['predictions'])
            mse = mean_squared_error(results[indicator]['actual'], 
                                   results[indicator]['predictions'])
            accuracy_data['Indicator'].append(indicator)
            accuracy_data['MAE'].append(mae)
            accuracy_data['MSE'].append(mse)

    return pd.DataFrame(accuracy_data).sort_values(by='MAE').reset_index(drop=True)

def plot_prediction_results(df, results, ticker):
    """Plot prediction results"""
    indicators = list(results.keys())
    fig = make_subplots(rows=len(indicators), cols=1, shared_xaxes=True, 
                       vertical_spacing=0.02,
                       subplot_titles=[f"{ind} Daily MAE" for ind in indicators[:-1]])

    y_values = [results[ind]['daily_mae'] for ind in indicators[:-1]]
    y_min = min(min(y) for y in y_values)
    y_max = max(max(y) for y in y_values)

    for idx, indicator in enumerate(indicators[:-1]):
        fig.add_trace(
            go.Scatter(
                x=df.index[window_size + 1:],
                y=results[indicator]['daily_mae'],
                mode='lines',
                name=f'{indicator} Daily MAE'
            ),
            row=idx + 1, col=1
        )

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_xaxes(title_text="Date", row=len(indicators), col=1)

    fig.update_layout(
        height=150 * (len(indicators)),
        title=f"Daily MAE of Technical Indicators on {ticker} Closing Price",
        yaxis_title="Daily MAE",
        showlegend=False,
        template="plotly_white"
    )
    return fig

def plot_technical_indicators(df, ticker):
    """Plot technical indicators overlay"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                           name='Close Price', line=dict(color='white', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', 
                           name='SMA 50', line=dict(color='yellow', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', 
                           name='EMA 50', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', 
                           name='BB Upper', line=dict(color='blue', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', 
                           name='BB Lower', line=dict(color='blue', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', 
                           name='BB Middle', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', 
                           name='MACD', line=dict(color='cyan', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', 
                           name='Signal Line', line=dict(color='purple', width=1)))

    fig.update_layout(
        title=f"Technical Indicators on {ticker} Close Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color="white"),
        width=800,
        height=600
    )
    return fig

################################################################################
# AI Advisor Functions
################################################################################

class State(TypedDict):
    messages: Annotated[list, add_messages]
    symbol: str
    llm: ChatOpenAI
    results: Dict

def run_ollama_serve():
    subprocess.Popen(["ollama", "serve"])

# Start Ollama server in background
thread = threading.Thread(target=run_ollama_serve)
thread.start()
time.sleep(5)

def setup_llm():
    return ChatOpenAI(
        model="deepseek-r1:8b",
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        temperature=0,
        top_p=0.7
    )

def technical_analysis(state: State) -> State:
    """Node for technical analysis"""
    symbol = state["symbol"]
    llm = state["llm"]

    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")
    
    if data.empty:
        state["results"]["technical"] = {
            "data": {},
            "analysis": "No technical data available"
        }
        return state

    sma_20 = data['Close'].rolling(window=20).mean()
    sma_50 = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    tech_data = {
        'current_price': data['Close'].iloc[-1],
        'sma_20': sma_20.iloc[-1],
        'sma_50': sma_50.iloc[-1],
        'rsi': rsi.iloc[-1],
        'volume_trend': data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:].mean()
    }

    prompt = PromptTemplate.from_template(
        """Analyze these technical indicators for {symbol}:
        {data}

        Provide:
        1. Trend analysis
        2. Support/Resistance levels
        3. Technical rating (Bullish/Neutral/Bearish)
        4. Key signals
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    analysis = chain.run(symbol=symbol, data=json.dumps(tech_data, indent=2))

    state["results"]["technical"] = {
        "data": tech_data,
        "analysis": analysis
    }
    return state

def market_analysis(state: State) -> State:
    """Node for market analysis"""
    symbol = state["symbol"]
    llm = state["llm"]

    stock = yf.Ticker(symbol)
    info = stock.info
    
    if not info:
        state["results"]["market"] = {
            "data": {},
            "analysis": "No market data available"
        }
        return state

    data = {
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', 0),
        'beta': info.get('beta', 1.0),
        'pe_ratio': info.get('trailingPE', 0)
    }

    prompt = PromptTemplate.from_template(
        """Analyze the market context for {symbol}:
        {data}

        Provide:
        1. Market sentiment
        2. Sector analysis
        3. Risk assessment
        4. Market outlook
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    analysis = chain.run(symbol=symbol, data=json.dumps(data, indent=2))

    state["results"]["market"] = {
        "data": data,
        "analysis": analysis
    }
    return state

def fetch_news_for_advisor(ticker, num_articles=6):
    """Fetch news specifically for AI advisor"""
    google_news = GNews()
    try:
        news = google_news.get_news(ticker)
        if not news:
            st.warning(f"No news found for {ticker}")
            return None
                
        filtered_news = [
            article for article in news 
            if ticker.lower() in article['title'].lower() or 
               ticker.upper() in article['title'].upper()
        ][:max(num_articles, 6)]
        
        if not filtered_news:
            st.warning(f"No relevant news found for {ticker}")
            return None
                
        news_data = [{
            'title': article['title'],
            'publisher': article.get('publisher', 'Unknown'),
            'date': article.get('published date', 'Unknown'),
            'link': article.get('url', '#'),
            'description': article.get('description', '')
        } for article in filtered_news]
        
        return news_data
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return None

def news_analysis(state: State) -> State:
    """Node for news analysis"""
    symbol = state["symbol"]
    llm = state["llm"]

    news_data = fetch_news_for_advisor(symbol)
    
    if not news_data:
        state["results"]["news"] = {
            "data": [],
            "analysis": "No relevant news found for analysis"
        }
        return state

    prompt = PromptTemplate.from_template(
        """Analyze these recent news items for {symbol}:
        {news}

        Provide:
        1. Overall sentiment (positive/negative/neutral)
        2. Key developments mentioned
        3. Potential impact on the stock
        4. Risk factors identified
        5. Any notable analyst opinions or ratings
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    analysis = chain.run(symbol=symbol, news=json.dumps(news_data, indent=2))

    state["results"]["news"] = {
        "data": news_data,
        "analysis": analysis
    }
    return state

def generate_recommendation(state: State) -> State:
    """Node for final recommendation"""
    symbol = state["symbol"]
    llm = state["llm"]
    results = state["results"]

    prompt = PromptTemplate.from_template(
        """Based on the following analyses for {symbol}, provide a final recommendation:

        TECHNICAL ANALYSIS:
        {technical}

        MARKET ANALYSIS:
        {market}

        NEWS ANALYSIS:
        {news}

        Provide a detailed report with:
        1. Final recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        2. Confidence score (1-10)
        3. Key reasons supporting the recommendation
        4. Main risk factors to consider
        5. Target price range with justification
        6. Suggested time horizon for the investment
        7. Any alternative scenarios to watch for
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    final_recommendation = chain.run(
        symbol=symbol,
        technical=results["technical"]["analysis"],
        market=results["market"]["analysis"],
        news=results["news"]["analysis"]
    )

    state["results"]["recommendation"] = final_recommendation
    return state

def create_analysis_graph() -> Graph:
    """Create the analysis workflow graph"""
    workflow = StateGraph(State)

    workflow.add_node("technical", technical_analysis)
    workflow.add_node("market", market_analysis)
    workflow.add_node("news", news_analysis)
    workflow.add_node("recommendation", generate_recommendation)

    workflow.add_edge("technical", "market")
    workflow.add_edge("market", "news")
    workflow.add_edge("news", "recommendation")
    workflow.add_edge(START, "technical")

    workflow.add_edge("recommendation", END)

    return workflow.compile()

class StockAdvisor:
    def __init__(self):
        self.llm = setup_llm()
        self.graph = create_analysis_graph()

    def analyze_stock(self, symbol: str) -> Dict:
        init_sate: State = {
            "symbol": symbol,
            "llm": self.llm,
            "results": {}
        }
        final_state = self.graph.invoke(init_sate)
        return final_state["results"]

################################################################################
# Streamlit UI Implementation
################################################################################

# Initialize session state
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"

# Tab 1: Stock Analysis
with tab1:
    st.header("Stock Technical Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        st.session_state.ticker = st.text_input("Stock Ticker", st.session_state.ticker).upper()
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        indicators = st.multiselect(
            "Technical Indicators",
            ['SMA 50', 'SMA 200', 'EMA 20', 'RSI'],
            default=['EMA 20']
        )
        
        if st.button("Analyze Stock"):
            with st.spinner("Fetching stock data..."):
                data = fetch_stock_data(st.session_state.ticker, start_date, end_date)
                if not data.empty:
                    data = calculate_technical_indicators(data, indicators)
                    st.session_state.stock_data = data
                    st.session_state.indicators = indicators
    
    with col2:
        if 'stock_data' in st.session_state and not st.session_state.stock_data.empty:
            st.subheader(f"{st.session_state.ticker} Analysis")
            
            last_close = float(st.session_state.stock_data['Close'].iloc[-1])
            prev_close = float(st.session_state.stock_data['Close'].iloc[-2]) if len(st.session_state.stock_data) > 1 else last_close
            change = last_close - prev_close
            pct_change = (change / prev_close) * 100
            
            st.metric(
                label=f"{st.session_state.ticker} Price",
                value=safe_float_format(last_close),
                delta=f"{safe_float_format(change)} ({safe_float_format(pct_change)}%)"
            )
            
            fig = plot_stock_data(
                st.session_state.stock_data, 
                st.session_state.indicators,
                st.session_state.ticker
            )
            st.plotly_chart(fig, use_container_width=True, key=f"stock_chart_{st.session_state.ticker}_{datetime.now().timestamp()}")
            
            with st.expander("View Raw Data"):
                st.dataframe(st.session_state.stock_data.tail(10))
        elif 'stock_data' in st.session_state:
            st.warning("No stock data available for the selected parameters")

# Tab 2: News Sentiment Analysis
with tab2:
    st.header("News Sentiment Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        num_articles = st.slider("Number of Articles", 1, 20, 5)
        
        if st.button("Analyze News"):
            with st.spinner("Fetching and analyzing news..."):
                news_df = fetch_news(st.session_state.ticker, num_articles)
                if news_df is not None:
                    results = zero_shot_analysis(news_df)
                    if results is not None and not results.empty:
                        st.session_state.news_results = results
    
    with col2:
        if 'news_results' in st.session_state and not st.session_state.news_results.empty:
            st.subheader(f"{st.session_state.ticker} News Analysis")
            
            st.dataframe(st.session_state.news_results, hide_index=True)
            
            if 'Future Looking' in st.session_state.news_results.columns:
                pos_count = len(st.session_state.news_results[
                    st.session_state.news_results['Sentiment'] == 'positive'])
                future_count = len(st.session_state.news_results[
                    st.session_state.news_results['Future Looking'] == True])
                
                col1, col2 = st.columns(2)
                col1.metric("Positive Sentiment", f"{pos_count}/{len(st.session_state.news_results)}")
                col2.metric("Future-Looking", f"{future_count}/{len(st.session_state.news_results)}")
                
                tab1, tab2 = st.tabs(["Sentiment", "Future-Looking"])
                with tab1:
                    # Convert to DataFrame for bar chart
                    sentiment_counts = st.session_state.news_results['Sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                with tab2:
                    # Convert to DataFrame for bar chart
                    future_counts = st.session_state.news_results['Future Looking'].value_counts()
                    st.bar_chart(future_counts)
        elif 'news_results' in st.session_state:
            st.warning("No news analysis results available")

# Tab 3: Combined View
with tab3:
    st.header("Combined Stock & News Analysis")
    
    if 'stock_data' in st.session_state and not st.session_state.stock_data.empty and \
       'news_results' in st.session_state and not st.session_state.news_results.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{st.session_state.ticker} Stock Chart")
            fig = plot_stock_data(
                st.session_state.stock_data, 
                st.session_state.indicators,
                st.session_state.ticker
            )
            st.plotly_chart(fig, use_container_width=True, 
                           key=f"combined_stock_chart_{st.session_state.ticker}")
        
        with col2:
            st.subheader(f"{st.session_state.ticker} News Sentiment")
            st.dataframe(st.session_state.news_results, hide_index=True)
            
            st.subheader("Correlation Analysis")
            last_price = float(st.session_state.stock_data['Close'].iloc[-1])
            prev_price = float(st.session_state.stock_data['Close'].iloc[-2]) if len(st.session_state.stock_data) > 1 else last_price
            pos_count = len(st.session_state.news_results[
                st.session_state.news_results['Sentiment'] == 'positive'])
            pos_ratio = pos_count / len(st.session_state.news_results)
            
            st.write(f"Current Price: {safe_float_format(last_price)}")
            st.write(f"Positive News Ratio: {safe_float_format(pos_ratio*100)}%")
            
            if pos_ratio > 0.6 and last_price > prev_price:
                st.success("Strong positive sentiment with upward price movement")
            elif pos_ratio < 0.4 and last_price < prev_price:
                st.warning("Negative sentiment with downward price movement")
            else:
                st.info("Mixed signals between news sentiment and price movement")
    else:
        st.warning("Please analyze both stock and news data first")

# Tab 4: AI Advisor
with tab4:
    st.header("AI Stock Advisor")
    
    if st.button("Run AI Analysis"):
        with st.spinner("Running comprehensive AI analysis..."):
            advisor = StockAdvisor()
            results = advisor.analyze_stock(st.session_state.ticker)
            
            with st.expander("📰 News Articles Analyzed", expanded=True):
                if results["news"]["data"]:
                    st.write(f"Found {len(results['news']['data'])} relevant news articles:")
                    for i, article in enumerate(results["news"]["data"], 1):
                        st.markdown(f"""
                        ### Article {i}
                        **Title:** {article['title']}  
                        **Publisher:** {article['publisher']}  
                        **Date:** {article['date']}  
                        **Description:** {article.get('description', 'No description available')}  
                        [Read more]({article['link']})
                        """)
                        st.write("---")
                else:
                    st.warning("No news articles found for analysis")
            
            st.subheader("Technical Analysis")
            st.write(results["technical"]["analysis"])
            
            st.subheader("Market Analysis")
            st.write(results["market"]["analysis"])
            
            st.subheader("News Analysis")
            st.write(results["news"]["analysis"])
            
            st.subheader("📊 Final Recommendation")
            st.markdown(results["recommendation"])

# Tab 5: Technical Prediction
with tab5:
    st.header("Technical Prediction Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        window_size = st.slider("Rolling Window Size", 10, 60, 20)
        
        if st.button("Run Technical Prediction"):
            with st.spinner("Preparing data and running predictions..."):
                try:
                    pred_df = prepare_prediction_data(st.session_state.ticker)
                    results = run_technical_prediction(pred_df, window_size)
                    accuracy_df = calculate_prediction_accuracy(results)
                    
                    st.session_state.pred_df = pred_df
                    st.session_state.pred_results = results
                    st.session_state.accuracy_df = accuracy_df
                except Exception as e:
                    st.error(f"Error running prediction: {str(e)}")

    with col2:
        if 'pred_results' in st.session_state:
            st.subheader(f"{st.session_state.ticker} Technical Prediction Results")
            
            st.markdown("### Prediction Accuracy by Indicator")
            st.dataframe(st.session_state.accuracy_df.style.highlight_min(axis=0, color='lightgreen'))
            
            st.markdown("### Daily MAE of Technical Indicators")
            mae_fig = plot_prediction_results(
                st.session_state.pred_df,
                st.session_state.pred_results,
                st.session_state.ticker
            )
            st.plotly_chart(mae_fig, use_container_width=True, 
                          key=f"mae_chart_{st.session_state.ticker}_{window_size}")
            
            st.markdown("### Technical Indicators Overlay")
            indicators_fig = plot_technical_indicators(
                st.session_state.pred_df,
                st.session_state.ticker
            )
            st.plotly_chart(indicators_fig, use_container_width=True,
                           key=f"tech_indicators_{st.session_state.ticker}")

# Sidebar
with st.sidebar:
    st.header("Market News Summary")
    
    google_news = GNews()
    market_news = google_news.get_news("market")[:5]
    
    if market_news:
        st.subheader("Top Market News")
        for i, article in enumerate(market_news, 1):
            st.markdown(f"""
            **{i}. {article['title']}**  
            [Read more]({article.get('url', '#')})
            """)
            st.write("---")
    else:
        st.warning("No market news available")
    
    st.header("About")
    st.info("""
    This dashboard provides comprehensive stock analysis including:
    - Technical indicators and price trends
    - News sentiment analysis
    - AI-powered stock recommendations
    - Technical price predictions
    """)
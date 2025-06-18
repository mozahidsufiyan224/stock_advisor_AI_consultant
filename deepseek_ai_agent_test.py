# !pip install langchain langgraph yfinance langchain-openai langchain-community
# !pip install ollama
# !pip install pydantic
# !pip install pandas
# !pip install gnews

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict
import json

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import ollama
from ollama import chat
from pydantic import BaseModel
import pandas as pd
from gnews import GNews

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    symbol: str
    llm: ChatOpenAI
    results: Dict

import threading
import subprocess
import time

def run_ollama_serve():
    subprocess.Popen(["ollama", "serve"])

thread = threading.Thread(target=run_ollama_serve)
thread.start()
time.sleep(5)  # Give it time to start


# Initialize the ollama endpoint
def setup_llm():
    return ChatOpenAI(
        model="deepseek-r1:8b",
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        temperature=0,
        top_p=0.7
    )

# Technical Analysis Node
def technical_analysis(state: State) -> State:
    """Node for technical analysis"""
    symbol = state["symbol"]
    llm = state["llm"]

    # Fetch technical data
    stock = yf.Ticker(symbol)
    hist = stock.history(period='1y')

    # Calculate indicators
    sma_20 = hist['Close'].rolling(window=20).mean()
    sma_50 = hist['Close'].rolling(window=50).mean()
    rsi = calculate_rsi(hist['Close'])

    data = {
        'current_price': hist['Close'].iloc[-1],
        'sma_20': sma_20.iloc[-1],
        'sma_50': sma_50.iloc[-1],
        'rsi': rsi.iloc[-1],
        'volume_trend': hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:].mean()
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
    analysis = chain.run(symbol=symbol, data=json.dumps(data, indent=2))

    state["results"]["technical"] = {
        "data": data,
        "analysis": analysis
    }
    return state

# Market Analysis Node
def market_analysis(state: State) -> State:
    """Node for market analysis"""
    symbol = state["symbol"]
    llm = state["llm"]

    # Fetch market data
    stock = yf.Ticker(symbol)
    info = stock.info

    data = {
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
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

# News Analysis Node
def news_analysis(state: State) -> State:
    """Node for news analysis"""
    symbol = state["symbol"]
    llm = state["llm"]

    # Fetch news################################################################
    # stock = yf.Ticker(symbol)
    # news = stock.news[:5]  # Last 5 news items################################

    # # Fetch news articles
    google_news = GNews(period='1m') # google_news = GNews(period='7d')  # You can change the period (e.g., '1d', '7d', '1m')
    news = google_news.get_news(symbol)

    news_data = [{
        'title': item.get('title', ''),
        'publisher': item.get('publisher', ''),
        'timestamp': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
    } for item in news]

    prompt = PromptTemplate.from_template(
        """Analyze these recent news items for {symbol}:
        {news}

        Provide:
        1. Overall sentiment
        2. Key developments
        3. Potential impact
        4. Risk factors
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    analysis = chain.run(symbol=symbol, news=json.dumps(news_data, indent=2))

    state["results"]["news"] = {
        "data": news_data,
        "analysis": analysis
    }
    return state

# Final Recommendation Node
def generate_recommendation(state: State) -> State:
    """Node for final recommendation"""
    symbol = state["symbol"]
    llm = state["llm"]
    results = state["results"]

    prompt = PromptTemplate.from_template(
        """Based on the following analyses for {symbol}, provide a final recommendation:

        Technical Analysis:
        {technical}

        Market Analysis:
        {market}

        News Analysis:
        {news}

        Provide:
        1. Final recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        2. Confidence score (1-10)
        3. Key reasons
        4. Risk factors
        5. Target price range
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

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_analysis_graph() -> Graph:
    """Create the analysis workflow graph"""
    # Create workflow graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("technical", technical_analysis)
    workflow.add_node("market", market_analysis)
    workflow.add_node("news", news_analysis)
    workflow.add_node("recommendation", generate_recommendation)

    # Define edges
    workflow.add_edge("technical", "market")
    workflow.add_edge("market", "news")
    workflow.add_edge("news", "recommendation")
    workflow.add_edge(START, "technical")

    # Set end node
    workflow.add_edge("recommendation", END)


    return workflow.compile()

class StockAdvisor:
    def __init__(self):
        self.llm = setup_llm()
        self.graph = create_analysis_graph()

    def analyze_stock(self, symbol: str) -> Dict:
        """Run complete stock analysis"""
        print(f"\nAnalyzing {symbol}...")

        # Initialize state
        init_sate: State = {
            "symbol": symbol,
            "llm": self.llm,
            "results": {}
        }

        # Run analysis
        final_state = self.graph.invoke(init_sate)
        return final_state["results"]

# Helper function to run analysis
def run_analysis(symbol: str):
    """Run stock analysis and print results"""
    advisor = StockAdvisor()
    results = advisor.analyze_stock(symbol)

    print(f"\n=== Stock Analysis Report for {symbol} ===")

    print("\n=== Technical Analysis ===")
    print(results["technical"]["analysis"])

    print("\n=== Market Analysis ===")
    print(results["market"]["analysis"])

    print("\n=== News Analysis ===")
    print(results["news"]["analysis"])

    print("\n=== Final Recommendation ===")
    print(results["recommendation"])

    # Convert the results to a DataFrame#######################################
    df = pd.DataFrame(news_data)
    df ########################################################################

    return results

from IPython.display import Image, display

try:
    display(Image(create_analysis_graph().get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# Example usage:
results = run_analysis("MSFT")

"""# New section

# New section
"""
"""
AI Stock Predictor using News Sentiment Analysis
Deep Charts YouTube Channel: https://www.youtube.com/@DeepCharts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from gnews import GNews
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from textblob import TextBlob
import matplotlib.dates as mdates

# Initialize GNews client
google_news = GNews(language='en', country='US', period='30d', max_results=100)

def get_news_data(ticker):
    """Fetch news headlines for a given stock ticker"""
    print(f"Fetching news for {ticker}...")
    try:
        articles = google_news.get_news(f'{ticker} stock OR {ticker} shares OR {ticker} earnings')
        
        if not articles:
            print(f"No news articles found for {ticker}")
            return pd.DataFrame()
        
        news_data = []
        for article in articles:
            try:
                published_date = pd.to_datetime(article['published date'])
                news_data.append({
                    'date': published_date,
                    'title': article['title'],
                    'description': article['description'],
                    'source': article['publisher']['title'],
                    'url': article['url']
                })
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        news_df = pd.DataFrame(news_data)
        
        if news_df.empty:
            return pd.DataFrame()
        
        # Normalize dates (remove time component)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()
        return news_df
    
    except Exception as e:
        print(f"Failed to fetch news data: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if not text or pd.isna(text):
        return 0.0  # Neutral
    
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity  # Range from -1 (negative) to 1 (positive)

def process_sentiment_data(news_df):
    """Process news data into sentiment metrics"""
    if news_df.empty:
        return pd.DataFrame()
    
    # Calculate sentiment scores
    print("Analyzing sentiment...")
    news_df['sentiment_score'] = news_df['title'].apply(analyze_sentiment)
    news_df['description_sentiment'] = news_df['description'].apply(analyze_sentiment)
    
    # Combine title and description sentiment
    news_df['combined_sentiment'] = (news_df['sentiment_score'] + news_df['description_sentiment']) / 2
    
    # Group by date and calculate metrics
    daily_sentiment = news_df.groupby('date').agg({
        'combined_sentiment': ['mean', 'count'],
        'sentiment_score': 'mean',
        'description_sentiment': 'mean'
    })
    
    # Flatten multi-index columns
    daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
    
    # Calculate rolling sentiment
    daily_sentiment['7day_sentiment_ma'] = daily_sentiment['combined_sentiment_mean'].rolling(7).mean()
    daily_sentiment['article_count'] = daily_sentiment['combined_sentiment_count']
    
    return daily_sentiment.reset_index()

def generate_synthetic_price(sentiment_df):
    """Generate synthetic price data based on sentiment trends"""
    if sentiment_df.empty:
        return pd.DataFrame()
    
    print("Generating synthetic price data...")
    # Base price starts at 100
    base_price = 100.0
    
    # Create synthetic price movements based on sentiment
    prices = [base_price]
    for i in range(1, len(sentiment_df)):
        # Price change is influenced by sentiment and random noise
        sentiment_effect = sentiment_df.iloc[i]['7day_sentiment_ma'] * 2  # Amplify effect
        noise = np.random.normal(0, 0.5)  # Random market noise
        price_change = sentiment_effect + noise
        
        new_price = prices[i-1] * (1 + price_change/100)
        prices.append(new_price)
    
    sentiment_df['synthetic_price'] = prices
    sentiment_df['pct_change'] = sentiment_df['synthetic_price'].pct_change() * 100
    
    return sentiment_df

def fit_and_forecast(df):
    """Generate forecasts using SARIMAX"""
    if df.empty or 'pct_change' not in df.columns:
        return pd.Series(), pd.DataFrame(), []
    
    print("Generating forecast...")
    try:
        model = SARIMAX(df['pct_change'], 
                      order=(1,1,1), 
                      seasonal_order=(1,1,1,7))
        results = model.fit(disp=False)
        
        forecast = results.get_forecast(steps=7)
        mean = forecast.predicted_mean
        ci = forecast.conf_int()
        dates = pd.date_range(df['date'].iloc[-1] + timedelta(days=1), periods=7)
        
        return mean, ci, dates
    except Exception as e:
        print(f"Forecasting failed: {str(e)}")
        return pd.Series(), pd.DataFrame(), []

def create_plot(historical, forecast_mean, forecast_ci, forecast_dates, ticker):
    """Create and display plot with proper formatting"""
    plt.figure(figsize=(14, 7))
    
    # Convert dates to matplotlib format
    hist_dates = mdates.date2num(historical['date'])
    forecast_dates_mpl = mdates.date2num(forecast_dates)
    
    # Historical prices - make line thicker and more visible
    plt.plot(hist_dates, 
             historical['synthetic_price'], 
             label='Synthetic Price', 
             color='blue',
             linewidth=2.5,
             marker='o',
             markersize=5)
    
    # Forecast - make line thicker and dashed
    if not forecast_mean.empty:
        # Convert forecast to price levels
        last_price = historical['synthetic_price'].iloc[-1]
        forecast_prices = [last_price * (1 + fc/100) for fc in forecast_mean.cumsum()/100]
        
        plt.plot(forecast_dates_mpl, 
                forecast_prices, 
                label='Forecast', 
                color='green', 
                linestyle='--',
                linewidth=2.5,
                marker='s',
                markersize=6)
        
        # Confidence interval - make it more visible
        lower_prices = [last_price * (1 + fc/100) for fc in forecast_ci.iloc[:, 0].cumsum()/100]
        upper_prices = [last_price * (1 + fc/100) for fc in forecast_ci.iloc[:, 1].cumsum()/100]
        
        plt.fill_between(forecast_dates_mpl,
                       lower_prices,
                       upper_prices,
                       color='green', 
                       alpha=0.2,
                       label='Confidence Interval')
    
    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.title(f'{ticker} Synthetic Price Forecast Based on News Sentiment', pad=20)
    plt.xlabel('Date', labelpad=10)
    plt.ylabel('Price', labelpad=10)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def run_analysis(ticker="AAPL"):
    """Main analysis pipeline"""
    print(f"\n=== Running News-Based Analysis for {ticker} ===")
    
    # 1. Get news data
    news_df = get_news_data(ticker)
    if news_df.empty:
        print("Analysis stopped: No news data available")
        return
    
    # 2. Process sentiment
    sentiment_df = process_sentiment_data(news_df)
    if sentiment_df.empty:
        print("Analysis stopped: No sentiment data processed")
        return
    
    # 3. Generate synthetic price data
    analysis_df = generate_synthetic_price(sentiment_df)
    if analysis_df.empty:
        print("Analysis stopped: Could not generate price data")
        return
    
    # 4. Calculate correlation between sentiment and price
    correlation = analysis_df['7day_sentiment_ma'].corr(analysis_df['pct_change'])
    print(f"\nSentiment-Price Correlation: {correlation:.2f}")
    
    # 5. Show recent data
    print("\nRecent Data:")
    print(analysis_df[['date', '7day_sentiment_ma', 'pct_change', 'synthetic_price']].tail())
    
    # 6. Generate forecast
    forecast_mean, forecast_ci, forecast_dates = fit_and_forecast(analysis_df)
    
    if not forecast_mean.empty:
        print("\nForecast Results:")
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Daily % Change Forecast': forecast_mean,
            'Lower CI': forecast_ci.iloc[:, 0],
            'Upper CI': forecast_ci.iloc[:, 1]
        })
        print(forecast_df)
    
    # 7. Create plot
    create_plot(analysis_df, forecast_mean, forecast_ci, forecast_dates, ticker)
    
    # 8. Show sample news
    print("\nSample News Headlines:")
    print(news_df[['date', 'title', 'source']].head(5).to_string(index=False))

if __name__ == "__main__":
    # Run analysis for a stock
    run_analysis("AAPL")  # Change to any ticker you want to analyze
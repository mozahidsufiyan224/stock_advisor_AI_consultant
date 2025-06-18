import streamlit as st
from ollama import chat
from pydantic import BaseModel
import pandas as pd
from gnews import GNews

# Set up the Streamlit page
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Add title and description
st.title("📰 News Sentiment Analyzer")
st.markdown("""
Analyze financial news sentiment and extract future-looking insights using LLMs.
[Deep Charts YouTube Channel](https://www.youtube.com/@DeepCharts)
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Stock Ticker", "LDG.L").upper()
    num_articles = st.slider("Number of Articles", 1, 10, 6)
    analyze_button = st.button("Analyze News", type="primary")

# Define BaseModel for news analysis
class NewsAnalysis(BaseModel):
    sentiment: str  
    future_looking: bool  

# Main analysis function
def analyze_news(ticker, num_articles):
    # Initialize GNews client
    google_news = GNews()
    
    # Show loading spinner while fetching news
    with st.spinner(f"Fetching {num_articles} news articles about {ticker}..."):
        try:
            # Fetch news articles
            news = google_news.get_news(ticker)
            
            if not news:
                st.error("No news articles found for this ticker")
                return None
                
            # Extract news titles
            news_titles = [article['title'] for article in news[:num_articles]]
            
            # Display raw news titles
            with st.expander("View Raw News Headlines"):
                for i, title in enumerate(news_titles, 1):
                    st.write(f"{i}. {title}")
            
            return news_titles
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return None

# Analysis function
def perform_sentiment_analysis(news_titles):
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, title in enumerate(news_titles):
        # Update progress - FIXED THIS LINE
        progress = int((i + 1) / len(news_titles) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing article {i + 1} of {len(news_titles)}...")
        
        try:
            # Call LLM for analysis
            response = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': f"""Analyze the following title for sentiment (positive, negative, or neutral) 
                                       and whether it provides future-looking financial insight, predictions, or 
                                       guidance on whether to buy/hold/sell the stock (True or False): {title}
                        """,
                    }
                ],
                model='llama3.2',
                format=NewsAnalysis.model_json_schema(),
            )

            # Parse the response
            sentiment_analysis = NewsAnalysis.model_validate_json(response['message']['content'])

            # Store results
            results.append({
                'Title': title,
                'Sentiment': sentiment_analysis.sentiment,
                'Future Looking': sentiment_analysis.future_looking
            })
            
        except Exception as e:
            st.warning(f"Failed to analyze article: {title}\nError: {str(e)}")
            continue
    
    # Complete progress
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# Main execution flow
if analyze_button:
    # Clear previous results
    st.session_state.analysis_results = None
    
    # Fetch news
    news_titles = analyze_news(ticker, num_articles)
    
    if news_titles:
        # Perform analysis
        with st.spinner("Analyzing sentiment and future-looking insights..."):
            df = perform_sentiment_analysis(news_titles)
            
            if not df.empty:
                st.session_state.analysis_results = df
                
                # Display results
                st.subheader("Analysis Results")
                
                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Sentiment", 
                             f"{len(df[df['Sentiment'] == 'positive'])}/{len(df)}",
                             help="Number of articles with positive sentiment")
                
                with col2:
                    st.metric("Future-Looking Insights", 
                             f"{len(df[df['Future Looking'] == True])}/{len(df)}",
                             help="Number of articles with future-looking insights")
                
                # Show full results table
                st.dataframe(df, use_container_width=True)
                
                # Add visualizations
                st.subheader("Visualizations")
                
                tab1, tab2 = st.tabs(["Sentiment Distribution", "Future-Looking Insights"])
                
                with tab1:
                    sentiment_counts = df['Sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                
                with tab2:
                    future_counts = df['Future Looking'].value_counts()
                    st.bar_chart(future_counts)
            else:
                st.error("No valid analysis results were returned")

# Show cached results if available
if 'analysis_results' in st.session_state and not st.session_state.analysis_results.empty:
    st.subheader("Previous Analysis Results")
    st.dataframe(st.session_state.analysis_results, use_container_width=True)























# # %% [markdown]
# # # Structured Outputs: From Text to Tabular Data

# # %% [markdown]
# # ## Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

# # %% [markdown]
# # ### Import Libraries

# # %%
# from ollama import chat
# from pydantic import BaseModel
# import pandas as pd
# from gnews import GNews

# # %% [markdown]
# # ### Pull News Headline Data

# # %%

# # Fetch news articles
# google_news = GNews()
# news = google_news.get_news("LDG.L")

# # Extract top 6 news titles
# news_titles = [article['title'] for article in news[:6]]
# news_titles

# # %% [markdown]
# # ### LLM Model and Structured Outputs

# # %%
# # Define BaseModel for news analysis
# class NewsAnalysis(BaseModel):
#     sentiment: str  
#     future_looking: bool  

# # Initialize an empty list to store results
# results = []

# # Loop through the news titles and analyze each
# for title in news_titles:
#     response = chat(
#         messages=[
#             {
#                 'role': 'user',
#                 'content': f"""Analyze the following title for sentiment (positive, negative, or neutral) 
#                                and whether it provides future-looking financial insight, predictions, or 
#                                guidance on whether to buy/hold/sell the stock (True or False): {title}
#                 """,
#             }
#         ],
#         model='llama3.2',
#         format=NewsAnalysis.model_json_schema(),
#     )

#     # Parse the response into the NewsAnalysis model
#     sentiment_analysis = NewsAnalysis.model_validate_json(response['message']['content'])

#     # Append the results to the list
#     results.append({
#         'title': title,
#         'sentiment': sentiment_analysis.sentiment,
#         'future_looking': sentiment_analysis.future_looking
#     })

# # Convert the results to a DataFrame
# df = pd.DataFrame(results)
# print(df)




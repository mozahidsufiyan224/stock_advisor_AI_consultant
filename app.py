import streamlit as st
from ollama import chat
from pydantic import BaseModel
import pandas as pd
from gnews import GNews
import random
from sklearn.model_selection import train_test_split

# Set up the Streamlit page
st.set_page_config(
    page_title="Advanced News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Add title and description
st.title("📰 Advanced News Sentiment Analyzer")
st.markdown("""
Analyze financial news sentiment using both zero-shot and few-shot learning approaches.
[Deep Charts YouTube Channel](https://www.youtube.com/@DeepCharts)
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Stock Ticker", "INTC").upper()
    num_articles = st.slider("Number of Articles", 1, 20, 10)
    analysis_type = st.radio("Analysis Type", ["Zero-shot", "Few-shot", "Compare Both"])
    analyze_button = st.button("Analyze News", type="primary")

# Define BaseModel for news analysis
class NewsAnalysis(BaseModel):
    sentiment: str  
    future_looking: bool  

# Main news fetching function
def fetch_news(ticker, num_articles):
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
                
            # Filter news to only include articles with the ticker in title
            filtered_news = [
                article for article in news 
                if ticker.lower() in article['title'].lower() or 
                   ticker.upper() in article['title'].upper()
            ][:num_articles]
            
            if not filtered_news:
                st.error(f"No articles found containing {ticker} in the title")
                return None
                
            # Extract news titles and dates if available
            news_data = [{
                'Title': article['title'],
                'Date': article.get('published date', 'Unknown'),
                'Link': article.get('url', '#')
            } for article in filtered_news]
            
            # Create DataFrame
            news_df = pd.DataFrame(news_data)
            
            # Display raw news titles
            with st.expander("View Raw News Headlines"):
                st.dataframe(news_df[['Date', 'Title']], hide_index=True)
            
            return news_df
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return None

# Zero-shot analysis function
def zero_shot_analysis(news_df):
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in news_df.iterrows():
        # Update progress
        progress = int((i + 1) / len(news_df) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing article {i + 1} of {len(news_df)}...")
        
        try:
            # Call LLM for analysis
            response = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': f"""Analyze the following title for sentiment (positive, negative, or neutral) 
                                       and whether it provides future-looking financial insight, predictions, or 
                                       guidance on whether to buy/hold/sell the stock (True or False): {row['Title']}
                        """,
                    }
                ],
                model='llama3',
                format=NewsAnalysis.model_json_schema(),
            )

            # Parse the response
            sentiment_analysis = NewsAnalysis.model_validate_json(response['message']['content'])

            # Store results
            results.append({
                'Date': row['Date'],
                'Title': row['Title'],
                'Link': row['Link'],
                'Sentiment_zero': sentiment_analysis.sentiment,
                'Future Looking': sentiment_analysis.future_looking
            })
            
        except Exception as e:
            st.warning(f"Failed to analyze article: {row['Title']}\nError: {str(e)}")
            continue
    
    # Complete progress
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# Few-shot analysis function
def few_shot_analysis(news_df):
    # Create a few-shot training set
    if len(news_df) >= 5:
        # Randomly select 3 examples for few-shot training (keeping some for prediction)
        few_shot_df, predict_df = train_test_split(news_df, test_size=0.7, random_state=42)
        
        st.subheader("Few-shot Training Examples")
        st.write("Please label these examples for few-shot learning:")
        
        # Get user labels for the few-shot examples
        user_labels = []
        for i, row in few_shot_df.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**Example {i+1}:** {row['Title']}")
            with col2:
                label = st.radio(
                    f"Sentiment for example {i+1}",
                    ["positive", "negative", "neutral"],
                    key=f"label_{i}"
                )
                user_labels.append(label)
        
        if not st.button("Continue with Analysis"):
            return None
            
        # Add user labels to few-shot examples
        few_shot_df = few_shot_df.copy()
        few_shot_df['User_Sentiment'] = user_labels
        
        # Show the labeled examples
        with st.expander("View Labeled Examples"):
            st.dataframe(few_shot_df[['Title', 'User_Sentiment']])
        
        # Analyze the remaining articles using few-shot learning
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, row in predict_df.iterrows():
            # Update progress
            progress = int((i + 1) / len(predict_df) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing article {i + 1} of {len(predict_df)}...")
            
            try:
                # Prepare few-shot prompt
                examples_str = "\n".join(
                    [f"Title: {ex['Title']}\nSentiment: {ex['User_Sentiment']}" 
                     for _, ex in few_shot_df.iterrows()]
                )
                
                # Call LLM for analysis
                response = chat(
                    messages=[
                        {
                            'role': 'user',
                            'content': f"""Based on these examples, analyze the sentiment of the news title below.
Examples:
{examples_str}

New Title to Analyze: {row['Title']}
Sentiment:""",
                        }
                    ],
                    model='llama3',
                )

                # Extract sentiment from response
                sentiment = response['message']['content'].strip().lower()
                
                # Validate sentiment
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'  # fallback
                
                results.append({
                    'Date': row['Date'],
                    'Title': row['Title'],
                    'Link': row['Link'],
                    'Sentiment_few': sentiment,
                    'Training Example': False
                })
                
            except Exception as e:
                st.warning(f"Failed to analyze article: {row['Title']}\nError: {str(e)}")
                continue
        
        # Add the training examples to results (marked as such)
        for _, row in few_shot_df.iterrows():
            results.append({
                'Date': row['Date'],
                'Title': row['Title'],
                'Link': row['Link'],
                'Sentiment_few': row['User_Sentiment'],
                'Training Example': True
            })
        
        # Complete progress
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    else:
        st.warning("Not enough articles for few-shot learning (need at least 5)")
        return None

# Main execution flow
if analyze_button:
    # Clear previous results
    st.session_state.analysis_results = None
    
    # Fetch news
    news_df = fetch_news(ticker, num_articles)
    
    if news_df is not None:
        # Perform analysis based on selected type
        with st.spinner("Analyzing sentiment..."):
            if analysis_type == "Zero-shot":
                df = zero_shot_analysis(news_df)
                if df is not None:
                    st.session_state.analysis_results = df
                    st.session_state.analysis_type = "zero"
            elif analysis_type == "Few-shot":
                df = few_shot_analysis(news_df)
                if df is not None:
                    st.session_state.analysis_results = df
                    st.session_state.analysis_type = "few"
            elif analysis_type == "Compare Both":
                # Run both analyses
                with st.spinner("Running zero-shot analysis..."):
                    zero_df = zero_shot_analysis(news_df)
                with st.spinner("Running few-shot analysis..."):
                    few_df = few_shot_analysis(news_df)
                
                if zero_df is not None and few_df is not None:
                    # Merge results
                    merged_df = zero_df.merge(
                        few_df, 
                        on=['Date', 'Title', 'Link'],
                        how='outer'
                    )
                    st.session_state.analysis_results = merged_df
                    st.session_state.analysis_type = "compare"
            
            # Display results if available
            if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                df = st.session_state.analysis_results
                
                st.subheader("Analysis Results")
                
                # Show appropriate results based on analysis type
                if st.session_state.analysis_type == "zero":
                    # Zero-shot results
                    st.dataframe(df[['Date', 'Title', 'Sentiment_zero', 'Future Looking']], hide_index=True)
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Positive Sentiment", 
                                 f"{len(df[df['Sentiment_zero'] == 'positive'])}/{len(df)}")
                    with col2:
                        st.metric("Future-Looking Insights", 
                                 f"{len(df[df['Future Looking'] == True])}/{len(df)}")
                    
                    # Visualizations
                    st.subheader("Visualizations")
                    tab1, tab2 = st.tabs(["Sentiment Distribution", "Future-Looking Insights"])
                    with tab1:
                        st.bar_chart(df['Sentiment_zero'].value_counts())
                    with tab2:
                        st.bar_chart(df['Future Looking'].value_counts())
                
                elif st.session_state.analysis_type == "few":
                    # Few-shot results
                    st.dataframe(df[['Date', 'Title', 'Sentiment_few', 'Training Example']], hide_index=True)
                    
                    # Metrics
                    st.metric("Positive Sentiment", 
                             f"{len(df[df['Sentiment_few'] == 'positive'])}/{len(df)}")
                    
                    # Visualization
                    st.subheader("Sentiment Distribution")
                    st.bar_chart(df['Sentiment_few'].value_counts())
                
                elif st.session_state.analysis_type == "compare":
                    # Comparison results
                    st.dataframe(df[['Date', 'Title', 'Sentiment_zero', 'Sentiment_few']], hide_index=True)
                    
                    # Comparison metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Zero-shot Positive", 
                                 f"{len(df[df['Sentiment_zero'] == 'positive'])}/{len(df)}")
                    with col2:
                        st.metric("Few-shot Positive", 
                                 f"{len(df[df['Sentiment_few'] == 'positive'])}/{len(df)}")
                    
                    # Comparison visualization
                    st.subheader("Comparison of Approaches")
                    
                    # Create comparison DataFrame
                    compare_df = pd.DataFrame({
                        'Approach': ['Zero-shot', 'Few-shot'],
                        'Positive': [
                            len(df[df['Sentiment_zero'] == 'positive']),
                            len(df[df['Sentiment_few'] == 'positive'])
                        ],
                        'Negative': [
                            len(df[df['Sentiment_zero'] == 'negative']),
                            len(df[df['Sentiment_few'] == 'negative'])
                        ],
                        'Neutral': [
                            len(df[df['Sentiment_zero'] == 'neutral']),
                            len(df[df['Sentiment_few'] == 'neutral'])
                        ]
                    }).set_index('Approach')
                    
                    st.bar_chart(compare_df)

# Show cached results if available
if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
    st.subheader("Previous Analysis Results")
    st.dataframe(st.session_state.analysis_results, use_container_width=True)
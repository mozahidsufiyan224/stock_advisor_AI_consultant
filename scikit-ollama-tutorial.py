# %% [markdown]
# Source: Deep Charts Youtube Channel: https://www.youtube.com/@DeepCharts

# %% [markdown]
# # AI Sentiment Analysis with Ollama and Scikit-Ollama

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd
from finvizfinance.quote import finvizfinance
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
from skollama.models.ollama.classification.few_shot import FewShotOllamaClassifier

# %% [markdown]
# ## Pull Stock News Headline Data

# %%
# Initialize the finvizfinance object for INTC
stock = finvizfinance('INTC')

# Fetch the latest news articles
news_df = stock.ticker_news()

news_df.head()

# %% [markdown]
# Data Wrangling (Reorder dataframe, remove headlines without company name in headline)

# %%
# Reorder Columns
news_df = news_df[['Date','Link','Title']]

# Define the keywords to filter by
keywords = ['INTC', 'Intel']

# Create a regex pattern by joining keywords with '|'
pattern = '|'.join(keywords)

# Filter the DataFrame using str.contains
filtered_news_df = news_df[news_df['Title'].str.contains(pattern, case=False, na=False)]

filtered_news_df.head()

# %% [markdown]
# ## Run Zero Shot Classifier

# %%
# Initialize the ZeroShotOllamaClassifier
clf = ZeroShotOllamaClassifier(model='llama3')

# Define the candidate labels
candidate_labels = ['positive', 'negative', 'neutral']

# Fit the classifier (no training data needed for zero-shot)
clf.fit(None, candidate_labels)

# Predict the sentiment of each news title as a new colum in our DataFrame
filtered_news_df['Sentiment_zero'] = clf.predict(filtered_news_df['Title'])


# %%
filtered_news_df[['Title','Sentiment_zero']]

# %% [markdown]
# ## Train and Run Few Shot Classifier

# %% [markdown]
# 1. Start by randomly selecting a few training examples from the original dataset

# %%
# Randomly select 6 headlines for few-shot training and add a training indicator
few_shot_df = filtered_news_df.sample(n=7, random_state=1)
filtered_news_df['Few Shot Training Example'] = filtered_news_df.index.isin(few_shot_df.index)

# View training examples
list(few_shot_df['Title'])

# %% [markdown]
# 2. Manually review each training example and give human guided label assignment

# %%
# Manually assigned labels corresponding to the selected headlines
# Fill in below based on above headlines
user_labels = [
    'neutral',
    'negative',
    'neutral',
    'positive',
    'positive',
    'neutral',
    'positive'
]

# Add the user-provided labels to the few-shot DataFrame
few_shot_df['User_Sentiment'] = user_labels

# %% [markdown]
# 3. Initialize and run few shot classifier on the rest of the dataset

# %%
# Initialize the FewShotOllamaClassifier
few_shot_clf = FewShotOllamaClassifier(model='llama3')

# Fit the classifier with user-provided examples directly from the DataFrame columns
few_shot_clf.fit(few_shot_df['Title'], few_shot_df['User_Sentiment'])

# Predict the sentiment of all news titles in the filtered DataFrame
filtered_news_df['Sentiment_few'] = few_shot_clf.predict(filtered_news_df['Title'])


# %%
filtered_news_df_2 = filtered_news_df[['Title','Sentiment_zero','Sentiment_few','Few Shot Training Example']]
filtered_news_df_2



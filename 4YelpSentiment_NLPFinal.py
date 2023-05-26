#!/usr/bin/env python
# coding: utf-8

# ## Script 4, A Nambiar 
# 
# #### (Sentiment Analysis - Customized, Visualized Over Time)

# ### Sentiment Analysis: Explicitly Customized
# ##### Naive Bates Model Trained on Yelp Reviews + Fine Tuned on Annotated Dataset 

# In[ ]:


pip install eli5


# In[ ]:


pip install wordcloud


# In[2]:


import sklearn
import pandas as pd

import wordcloud
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
import numpy as np

import eli5


# In[5]:


directory = 'https://storage.googleapis.com/msca-bdp-data-open/yelp/'
fileName = 'yelp_train_sentiment.json'

path = directory + fileName


# In[6]:


get_ipython().run_cell_magic('time', '', "\nyelp = pd.read_json(path, orient='records', lines=True)\nyelp.shape\n")


# In[7]:


pd.set_option('display.max_colwidth', 200)


# In[8]:


yelp.head(15)


# In[9]:


# define X and y
X = yelp['text']
y = yelp['label']

print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")


# In[10]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(f"Training records, X_train: {X_train.shape} y_train: {y_train.shape}")
print(f"Testing records, X_test: {X_test.shape} y_test: {y_test.shape}")


# In[11]:


vect = CountVectorizer()

# vect = CountVectorizer(lowercase=False, stop_words='english',
#                                   max_df=0.8, min_df=0.2, max_features=10000, ngram_range=(1,3))

vect = CountVectorizer(lowercase=False, stop_words='english', ngram_range=(1,3))


# In[12]:


# instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()


# In[13]:


get_ipython().run_line_magic('time', '')
nb.fit(vect.fit_transform(X_train), y_train)


# In[14]:


# make class predictions
y_pred = nb.predict(vect.transform(X_test))


# In[15]:


# calculate accuracy of class predictions
print(f"Test Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.1f}%")


# In[16]:


# calculate precision and recall
print(classification_report(y_test, y_pred))


# In[17]:


df = pd.read_parquet('filtered_news.parquet')
df.head()


# In[18]:


df.shape


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer

X_test_news = df['important_words'] 
X_test_vect = vect.transform(X_test_news)
y_pred = nb.predict(X_test_vect)
df['predicted_sentiment_yelp'] = y_pred


# In[20]:


df.head()


# In[21]:


df.shape


# In[22]:


df['predicted_sentiment_yelp'].value_counts()


# In[23]:


topic_sentiment_distribution = df.groupby('topic')['predicted_sentiment_yelp'].value_counts(normalize=True)

for topic in df['topic'].unique():
    topic_data = df[df['topic'] == topic]
    
    sentiment_distribution = topic_sentiment_distribution[topic]
    
    print(f"Topic: {topic}")
    print(sentiment_distribution)
    print()


# In[24]:


element = 0
clf = nb

text = X_test_news.iloc[element]
prediction = np.where(clf.predict(vect.transform([text])) < 1, "Negative", "Positive").tolist()[element]
print('Text: >>> ' + text + '\n' + 'Sentiment: >>> ' + prediction)


# In[25]:


prediction


# In[26]:


df.shape


# In[27]:


clf = nb
df['predicted_sentiment_YELP2'] = clf.predict(vect.transform(X_test_news))
df['predicted_sentiment_YELP2'] = np.where(df['predicted_sentiment_YELP2'] < 1, "0", "1")


# In[28]:


df.head()


# In[29]:


df['predicted_sentiment_YELP2'].value_counts()


# In[30]:


df[['predicted_sentiment_yelp', 'predicted_sentiment_YELP2']].nunique().gt(1).any()


# In[31]:


df.info()


# In[32]:


df['predicted_sentiment_YELP2'] = df['predicted_sentiment_YELP2'].astype(int)


# ### Fine Tuning

# #### Manual Labeling

# In[33]:


df['new_sentiment_label'] = ''

from sklearn.utils import shuffle
shuffled_df = df.sample(frac=1).reset_index(drop=True)

num_instances = 50


# In[34]:


for index, row in shuffled_df.iterrows():
    if index >= num_instances:
        break

    text = row['text']

    sentiment = input(f"Data instance {index+1}/{num_instances}:\n{text}\nSentiment (positive/negative): ")

    shuffled_df.at[index, 'new_sentiment_label'] = sentiment.lower()


# #### Fine Tuning Model

# In[39]:


shuffled_df[(shuffled_df['new_sentiment_label'].notnull()) & (shuffled_df['new_sentiment_label'] != '')]


# In[58]:


sentiment_df =  shuffled_df[(shuffled_df['new_sentiment_label'].notnull()) 
                            & (shuffled_df['new_sentiment_label'] != '')]
len(sentiment_df)


# In[59]:


fine_tuning_reviews = sentiment_df['important_words']
fine_tuning_sentiments = sentiment_df['new_sentiment_label']


# In[60]:


fine_tuning_reviews = vect.transform(fine_tuning_reviews)


# In[61]:


nb.fit(fine_tuning_reviews, fine_tuning_sentiments)


# In[65]:


y_pred_test = nb.predict(fine_tuning_reviews)
sentiment_df['predicted_sentiment_yelp_new'] = y_pred_test


# In[67]:


#sentiment_df


# In[69]:


X_test_news = df['important_words'] 
X_test_vect = vect.transform(X_test_news)
y_pred_new = nb.predict(X_test_vect)
df['predicted_sentiment_yelp_new'] = y_pred_new


# In[70]:


df


# In[85]:


df['predicted_sentiment_yelp_new'].value_counts()


# In[78]:


#df[df['predicted_sentiment_yelp_new'] =='p']


# In[77]:


#df[df['predicted_sentiment_yelp_new'] =='n']


# In[76]:


#df[df['predicted_sentiment_yelp_new'] =='neu']


# ### Sentiment Over Time Analysis and Visualization

# In[79]:


import matplotlib.pyplot as plt


# In[80]:


df['year'] = pd.to_datetime(df['date']).dt.year


# In[88]:


year_range = df['year'].unique()
min_year = year_range.min()
max_year = year_range.max()

print("Year range:", min_year, "-", max_year)


# In[91]:


df['date'] = pd.to_datetime(df['date'])


# ###### Below approach calculates the frequency of positive sentiment for each month. It provides the proportion of positive sentiment occurrences for each month.

# In[138]:


import pandas as pd
import matplotlib.pyplot as plt

df['date'] = pd.to_datetime(df['date'])

df['year_month'] = df['date'].dt.to_period('M')
sentiment_by_month = df.groupby(['year_month', 'predicted_sentiment_yelp_new']).size().unstack()
positive_sentiment_by_month = sentiment_by_month['p'] / sentiment_by_month.sum(axis=1)

positive_sentiment_by_month.index = positive_sentiment_by_month.index.to_timestamp()

fig, ax = plt.subplots()
ax.plot(positive_sentiment_by_month.index, positive_sentiment_by_month.values)
plt.xticks(rotation=45)

plt.xlabel('Month')
plt.ylabel('Positive Sentiment Proportion')
plt.title('Positive Sentiment Proportion of AI in News Articles Over Time')
plt.tight_layout()
plt.show()


# ##### BELOW calculates the average sentiment (mean()) for each year by grouping the DataFrame df by the 'year' column and taking the mean of the 'sentiment_numeric' column. It provides an average sentiment value for each year.

# In[143]:


import matplotlib.dates as mdates

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
sentiment_by_month = df.groupby(['year', 'month'])['sentiment_numeric'].mean()  
sentiment_by_month.index = pd.to_datetime(sentiment_by_month.index.map(lambda x: f'{x[0]}-{x[1]}'))

fig, ax = plt.subplots()
ax.plot(sentiment_by_month.index, sentiment_by_month.values)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 6]))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b %Y'))

plt.xticks(rotation=45)
for tick in ax.xaxis.get_minor_ticks():
    tick.label.set_rotation(45)
    
plt.xlabel('Year')
plt.ylabel('Positive Sentiment Average')
plt.title('Overall Sentiment of AI in News Articles Over Time')
plt.tight_layout()
plt.show()


# ### LDA for Each Sentiment

# In[146]:


#df.head()


# In[151]:


import nltk
from gensim import corpora
from gensim.models import LdaModel
from nltk.util import ngrams


# In[153]:


df['important_words'] = df['important_words'].str.lower()
df['tokens'] = df['important_words'].apply(nltk.word_tokenize)


# In[154]:


df['ngrams'] = df['tokens'].apply(lambda tokens: list(ngrams(tokens, 3)))


# In[157]:


df['ngrams'] = df['ngrams'].apply(lambda ngrams: [' '.join(gram) for gram in ngrams])


# In[158]:


positive_data = df[df['predicted_sentiment_yelp_new'] == 'p']['ngrams']
negative_data = df[df['predicted_sentiment_yelp_new'] == 'n']['ngrams']
neutral_data = df[df['predicted_sentiment_yelp_new'] == 'neu']['ngrams']


# ##### Positive Topics

# In[167]:


dictionary = corpora.Dictionary(positive_data)


# In[168]:


corpus = [dictionary.doc2bow(tokens) for tokens in positive_data]
num_topics = 5


# In[169]:


lda_model_positive = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)


# In[170]:


for topic_id, topic_words in lda_model_positive.show_topics(num_topics=num_topics):
    print(f"Topic ID: {topic_id}")
    print(f"Keywords: {topic_words}")
    print()


# Topic ID: 0 - Technology and Business:
# 
# Opens new tab
# Shares company's stock
# Artificial intelligence technology
# Earn affiliate commission
# Artificial intelligence AI
# Search
# 
# Topic ID: 1 - Artificial Intelligence and Global Impact:
# 
# Artificial intelligence AI
# Global artificial intelligence
# In-beta experience
# Sign in-beta
# Natural language processing
# Facebook, Twitter, LinkedIn
# AI cyber security
# Large language models
# 
# Topic ID: 2 - Online Services and Agreements:
# 
# Car insurance quotes
# Accurate real info
# Agree terms of use
# GDPR cookie consent
# Set GDPR cookie
# User consent cookies
# 
# Topic ID: 3 - AI in Market Insights and News:
# 
# Matrix AI network
# AI machine learning
# Artificial intelligence AI
# Natural language processing
# Machine learning models
# Breaking news alerts
# Market insights
# 
# Topic ID: 4 - Decentralized Machine Learning and Trades:
# 
# Decentralized machine learning
# Lower dollar trades
# Lisk machine learning
# WFMZTV 69 news
# Higher dollar trades
# Permission to edit article
# Intended users located within European Economic Area
# 

# ##### Negative Topics

# In[171]:


dictionary2 = corpora.Dictionary(negative_data)


# In[172]:


corpus2 = [dictionary2.doc2bow(tokens) for tokens in negative_data]
num_topics = 5


# In[174]:


lda_model_neg = LdaModel(corpus=corpus2, id2word=dictionary2, num_topics=num_topics, random_state=42)


# In[175]:


for topic_id, topic_words in lda_model_neg.show_topics(num_topics=num_topics):
    print(f"Topic ID: {topic_id}")
    print(f"Keywords: {topic_words}")
    print()


# Topic ID: 0 - Internet Search and Technology:
# 
# Bing search engine
# Enter search term
# Argo AI autonomous
# Internet search giants
# Recently rolled-out bot
# Broadcast, rewritten, redistributed
# 
# Topic ID: 1 - Account and Login:
# 
# Password, forgot password
# Email, log link
# Email address
# Permission to edit article
# Recovery, recover password
# Username, password, forgot
# Facebook, Twitter, WhatsApp
# 
# Topic ID: 2 - User Profiles and Interaction:
# 
# Indicates user profile
# Often indicates user
# Icon, icon shape
# Shoulders often indicate
# Shape person's head
# Account icon, icon shape
# Way of close interaction
# 
# Topic ID: 3 - OpenAI and Technology:
# 
# Name, email, website
# OpenAI logo seen
# Front computer screen
# Phone front computer
# Mobile phone front
# Broadcast, rewritten, redistributed
# 
# Topic ID: 4 - Entertainment and Media:
# 
# Videos, music, movies, visual art, TV series, books
# Viral videos, performing arts, TV
# News breaks, sign videos, music, visual art, TV
# Books, literature, comics, theater
# Dance behind viral videos, performing arts
# Enter search term

# In[ ]:


# neg_topic_labels = ['0', 'AI Search Engine, 'Topic 3', 'Topic 4', 'Topic 5', 'Media and Art']


# ##### Neutral Topics

# In[176]:


dictionary3 = corpora.Dictionary(neutral_data)


# In[177]:


corpus3 = [dictionary3.doc2bow(tokens) for tokens in neutral_data]
num_topics = 5


# In[178]:


lda_model_neutral = LdaModel(corpus=corpus3, id2word=dictionary3, num_topics=num_topics, random_state=42)


# In[179]:


for topic_id, topic_words in lda_model_neutral.show_topics(num_topics=num_topics):
    print(f"Topic ID: {topic_id}")
    print(f"Keywords: {topic_words}")
    print()


# In[188]:


import matplotlib.pyplot as plt

positive_topic_labels = ['Positive Topic 1', 'Positive Topic 2', 'Positive Topic 3']
negative_topic_labels = ['Negative Topic 1', 'Negative Topic 2', 'Negative Topic 3']

# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Positive Sentiment
positive_topics = []
for topic_id, topic_words in lda_model_positive.show_topics(num_topics=num_topics):
    positive_topics.append(topic_words)
axs[0].bar(range(num_topics), positive_topics)
axs[0].set_title('Positive Sentiment')

# Negative Sentiment
negative_topics = []
for topic_id, topic_words in lda_model_neg.show_topics(num_topics=num_topics):
    negative_topics.append(topic_words)
axs[1].bar(range(num_topics), negative_topics)
axs[1].set_title('Negative Sentiment')

#fig.text(0.5, 0.04, 'Topic ID', ha='center')
#fig.text(0.04, 0.5, 'Keywords', va='center', rotation='vertical')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[197]:


positive_topic_counts = []
for topic_id in range(num_topics):
    count = sum(1 for doc_topics in lda_model_positive.get_document_topics(corpus) if doc_topics[0][0] == topic_id)
    positive_topic_counts.append(count)


# In[199]:


negative_topic_counts = []
for topic_id in range(num_topics):
    count = sum(1 for doc_topics in lda_model_neg.get_document_topics(corpus2) if doc_topics[0][0] == topic_id)
    negative_topic_counts.append(count)


# In[206]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Positive Sentiment
axs[0].bar(range(num_topics), positive_topic_counts, color='green')
axs[0].set_title('Positive Sentiment')
#axs[0].set_xlabel('Topic ID')
axs[0].set_ylabel('Frequency')

positive_topic_labels = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']
axs[0].set_xticklabels(positive_topic_labels, rotation=45)  # Set custom x-axis tick labels

# Negative Sentiment
axs[1].bar(range(num_topics), negative_topic_counts, color='red')
axs[1].set_title('Negative Sentiment')
#axs[1].set_xlabel('Topic ID')
axs[1].set_ylabel('Frequency')

neg_topic_labels = ['0', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', '6']
axs[1].set_xticklabels(neg_topic_labels, rotation=45)  # Set custom x-axis tick labels

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[207]:


df


# In[208]:


df2 = df.drop(['ngrams', 'month', 'year_month', 'year', 'new_sentiment_label', 
               'predicted_sentiment_YELP2', 'predicted_sentiment_yelp'], axis=1)


# In[210]:


#df2


# In[213]:


topic_sentiment_df = df[['topic', 'sentiment_numeric']]
topic_counts = topic_sentiment_df['topic'].value_counts()
average_sentiment = topic_sentiment_df.groupby('topic')['sentiment_numeric'].mean()
sentiment_counts = topic_sentiment_df['sentiment_numeric'].value_counts()


# In[214]:


topic_counts.plot(kind='bar')
plt.xlabel('Topic Number')
plt.ylabel('Count')
plt.title('Topic Counts')
plt.show()


# In[215]:


average_sentiment.plot(kind='bar')
plt.xlabel('Topic Number')
plt.ylabel('Average Sentiment')
plt.title('Average Sentiment by Topic')
plt.show()


# In[216]:


sentiment_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()


# In[217]:


df2.to_parquet('sentiment_filtered_news.parquet')


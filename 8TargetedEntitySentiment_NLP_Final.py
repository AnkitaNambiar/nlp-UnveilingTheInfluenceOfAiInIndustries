#!/usr/bin/env python
# coding: utf-8

# ## Script 8, A Nambiar 
# ### Targeted (entity) Sentiment Identification

# In[1]:


import pandas as pd
import spacy
from collections import Counter


# In[10]:


df = pd.read_parquet('entity_nlp.parquet')
df.head()


# In[11]:


df.info()


# In[12]:


df_exploded = df.assign(Entity=df['entities']).explode('Entity')


# In[13]:


df_exploded.head()


# In[15]:


df_exploded[['entity_name', 'entity_type']] = df_exploded['Entity'].apply(pd.Series)


# In[16]:


df_exploded.head()


# In[17]:


sentiment_distribution = df_exploded['predicted_sentiment_yelp_new'].value_counts(normalize=True) * 100
print("Sentiment Distribution:")
print(sentiment_distribution)


# In[33]:


positive_df_actual = df_exploded[df_exploded['predicted_sentiment_yelp_new'] == 'p']
negative_df_actual = df_exploded[df_exploded['predicted_sentiment_yelp_new'] == 'n']


# In[34]:


positive_df.head()


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


positive_df = df[df['predicted_sentiment_yelp_new'] == 'p']

monthly_positive_counts = positive_df.groupby(pd.Grouper(key='date', freq='M')).size()
monthly_total_counts = df.groupby(pd.Grouper(key='date', freq='M')).size()

monthly_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(10, 6))
plt.plot(monthly_sentiment_percent.index, monthly_sentiment_percent)
plt.title('General Sentiment Over Time')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')
plt.xticks(rotation=45)

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]
plt.axvline(pd.Timestamp('2022-12-01'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.xticks(tick_locations, tick_labels)

plt.show()


# #### General Sentiment For Entity Type

# PERSON - People, including fictional.
# 
# NORP - Nationalities or religious or political groups.
# 
# FAC - Buildings, airports, highways, bridges, etc.
# 
# ORG - Companies, agencies, institutions, etc.
# 
# GPE - Countries, cities, states.
# 
# LOC - Non-GPE locations, mountain ranges, bodies of water.
# 
# PRODUCT - Objects, vehicles, foods, etc. (Not services.)
# 
# EVENT - Named hurricanes, battles, wars, sports events, etc.
# 
# WORK_OF_ART - Titles of books, songs, etc.
# 
# LAW - Named documents made into laws.
# 
# LANGUAGE - Any named language.
# 
# DATE - Absolute or relative dates or periods.
# 
# TIME - Times smaller than a day.
# 
# PERCENT - Percentage, including "%".
# 
# MONEY - Monetary values, including unit.
# 
# QUANTITY - Measurements, as of weight or distance.
# 
# ORDINAL - "first", "second", etc.
# 
# CARDINAL - Numerals that do not fall under another type.
# 
# 

# In[37]:


entities = ['PERSON', 'ORG', 'PRODUCT', 'GPE']

plt.figure(figsize=(10, 6))

for entity in entities:
    filtered_df = df_exploded[(df_exploded['entity_type'] == entity) & (df_exploded['predicted_sentiment_yelp_new'] == 'p')]

    monthly_positive_counts = filtered_df.groupby(pd.Grouper(key='date', freq='M')).size()
    monthly_total_counts = df_exploded[df_exploded['entity_type'] == entity].groupby(pd.Grouper(key='date', freq='M')).size()

    monthly_positive_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

    plt.plot(monthly_positive_sentiment_percent.index, monthly_positive_sentiment_percent, label=entity)

plt.title('Positive Sentiment Over Time for Entity Types')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]
plt.xticks(tick_locations, tick_labels)

plt.axvline(pd.Timestamp('2022-11-30'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.legend()  

plt.xticks(rotation=45)
plt.show()


# In[40]:


import numpy as np


# In[41]:


entities = ['PERSON', 'ORG', 'PRODUCT', 'GPE']

positive_entities_df2 = positive_df_actual[positive_df_actual['entity_type'].isin(entities)]
negative_entities_df2 = negative_df_actual[negative_df_actual['entity_type'].isin(entities)]

positive_counts2 = positive_entities_df2['entity_type'].value_counts()

negative_counts2 = negative_entities_df2['entity_type'].value_counts()

combined_counts2 = pd.concat([positive_counts2, negative_counts2], axis=1, sort=False).fillna(0)
combined_counts2.columns = ['Positive', 'Negative']

plt.figure(figsize=(10, 6))
x = np.arange(len(entities))
width = 0.35

plt.bar(x - width/2, combined_counts2['Positive'], width, color='green', label='Positive')
plt.bar(x + width/2, combined_counts2['Negative'], width, color='red', label='Negative')

plt.title('Positive and Negative Entity Types')
plt.xlabel('Entity Type')
plt.ylabel('Count')

plt.xticks(x, entities, rotation=45)
plt.legend()

for i, count in enumerate(combined_counts2.iterrows()):
    positive_count = count[1]['Positive']
    negative_count = count[1]['Negative']
    total_count = positive_count + negative_count
    plt.text(x[i] - width/2, positive_count + 10, f"{(positive_count/total_count)*100:.1f}%", ha='center', color='black')
    plt.text(x[i] + width/2, negative_count + 10, f"{(negative_count/total_count)*100:.1f}%", ha='center', color='black')

plt.show()


# #### Plot ORG Entities Over Time

# In[42]:


entities = ['Microsoft', 'OpenAI']

plt.figure(figsize=(10, 6))

for entity in entities:
    filtered_df = df_exploded[(df_exploded['entity_name'] == entity) & (df_exploded['predicted_sentiment_yelp_new'] == 'p')]

    monthly_positive_counts = filtered_df.groupby(pd.Grouper(key='date', freq='M')).size()
    monthly_total_counts = df_exploded[df_exploded['entity_name'] == entity].groupby(pd.Grouper(key='date', freq='M')).size()

    monthly_positive_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

    plt.plot(monthly_positive_sentiment_percent.index, monthly_positive_sentiment_percent, label=entity)

plt.title('Positive Sentiment Over Time for Organizations')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]
plt.xticks(tick_locations, tick_labels)

plt.axvline(pd.Timestamp('2022-11-30'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.legend()  # Add a legend to distinguish between entities

plt.xticks(rotation=45)
plt.show()


# In[43]:


entities = [ 'Google', 'DeepMind', 'C3.ai']

plt.figure(figsize=(10, 6))

for entity in entities:
    filtered_df = df_exploded[(df_exploded['entity_name'] == entity) & (df_exploded['predicted_sentiment_yelp_new'] == 'p')]

    monthly_positive_counts = filtered_df.groupby(pd.Grouper(key='date', freq='M')).size()
    monthly_total_counts = df_exploded[df_exploded['entity_name'] == entity].groupby(pd.Grouper(key='date', freq='M')).size()

    monthly_positive_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

    plt.plot(monthly_positive_sentiment_percent.index, monthly_positive_sentiment_percent, label=entity)

plt.title('Positive Sentiment Over Time for Organizations')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]

plt.axvline(pd.Timestamp('2022-11-30'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.legend()  
plt.xticks(tick_locations, tick_labels)
plt.xticks(rotation=45)
plt.show()


# #### Plot Product Entities Over Time

# In[44]:


entities = ['Bard', 'Excel', 'TensorFlow', 'CRM', 'ChatGPT']

plt.figure(figsize=(10, 6))

for entity in entities:
    filtered_df = df_exploded[(df_exploded['entity_name'] == entity) & (df_exploded['predicted_sentiment_yelp_new'] == 'p')]

    monthly_positive_counts = filtered_df.groupby(pd.Grouper(key='date', freq='M')).size()
    monthly_total_counts = df_exploded[df_exploded['entity_name'] == entity].groupby(pd.Grouper(key='date', freq='M')).size()

    monthly_positive_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

    plt.plot(monthly_positive_sentiment_percent.index, monthly_positive_sentiment_percent, label=entity)

plt.title('Positive Sentiment Over Time for Products')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]

plt.axvline(pd.Timestamp('2022-11-30'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.legend()  
plt.xticks(tick_locations, tick_labels)
plt.xticks(rotation=45)
plt.show()


# In[45]:


entities = [ 'A100', 'H100']

plt.figure(figsize=(10, 6))

for entity in entities:
    filtered_df = df_exploded[(df_exploded['entity_name'] == entity) & (df_exploded['predicted_sentiment_yelp_new'] == 'p')]

    monthly_positive_counts = filtered_df.groupby(pd.Grouper(key='date', freq='M')).size()
    monthly_total_counts = df_exploded[df_exploded['entity_name'] == entity].groupby(pd.Grouper(key='date', freq='M')).size()

    monthly_positive_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

    plt.plot(monthly_positive_sentiment_percent.index, monthly_positive_sentiment_percent, label=entity)

plt.title('Positive Sentiment Over Time for Products')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]

plt.axvline(pd.Timestamp('2022-11-30'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.legend()  
plt.xticks(tick_locations, tick_labels)
plt.xticks(rotation=45)
plt.show()


# #### Plot GPE Entities Over Time

# China
# US
# India
# UK
# Canada
# Japan
# Russia
# California
# San Francisco
# New York
# US
# 

# In[46]:


entities = [ 'California', 'San Francisco', 'New York']

plt.figure(figsize=(10, 6))

for entity in entities:
    filtered_df = df_exploded[(df_exploded['entity_name'] == entity) & (df_exploded['predicted_sentiment_yelp_new'] == 'p')]

    monthly_positive_counts = filtered_df.groupby(pd.Grouper(key='date', freq='M')).size()
    monthly_total_counts = df_exploded[df_exploded['entity_name'] == entity].groupby(pd.Grouper(key='date', freq='M')).size()

    monthly_positive_sentiment_percent = (monthly_positive_counts / monthly_total_counts) * 100

    plt.plot(monthly_positive_sentiment_percent.index, monthly_positive_sentiment_percent, label=entity)

plt.title('Positive Sentiment Over Time for GPEs')
plt.xlabel('Month - Year')
plt.ylabel('Positive Sentiment %')

tick_locations = monthly_sentiment_percent.index[::6]
tick_labels = [date.strftime('%B %Y') for date in tick_locations]

plt.axvline(pd.Timestamp('2022-11-30'), color='red', linestyle='--')
plt.axvline(pd.Timestamp('2023-01-23'), color='green', linestyle='--')

plt.legend()  
plt.xticks(tick_locations, tick_labels)
plt.xticks(rotation=45)
plt.show()


# ### Entitites of Each Sentiment

# In[47]:


df_exploded.info()


# In[48]:


import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# In[49]:


positive_df = df_exploded[df_exploded['predicted_sentiment_yelp_new'] == 'p']
negative_df = df_exploded[df_exploded['predicted_sentiment_yelp_new'] == 'n']


# In[50]:


positive_entities = positive_df['entity_name'].value_counts().head(20)
negative_entities = negative_df['entity_name'].value_counts().head(20)

print("Most common entities in positive sentiment:")
print(positive_entities)

print("\nMost common entities in negative sentiment:")
print(negative_entities)


# In[51]:


entities = ['OpenAI', 'Microsoft', 'Google', 'Bing', 'DeepMind', 'C3.ai', 'Nvidia']  

positive_entities_df = positive_df[positive_df['entity_name'].isin(entities)]
negative_entities_df = negative_df[negative_df['entity_name'].isin(entities)]

positive_counts = positive_entities_df['entity_name'].value_counts()

negative_counts = negative_entities_df['entity_name'].value_counts()


# In[55]:


combined_counts = pd.concat([positive_counts, negative_counts], axis=1, sort=False).fillna(0)
combined_counts.columns = ['Positive', 'Negative']

plt.figure(figsize=(10, 6))
x = np.arange(len(entities))
width = 0.35

plt.bar(x - width/2, combined_counts['Positive'], width, color='green', label='Positive')
plt.bar(x + width/2, combined_counts['Negative'], width, color='red', label='Negative')

plt.title('Count of Organization Articles in Positive and Negative Sentiment')
plt.xlabel('Organization')
plt.ylabel('Count')

plt.xticks(x, entities, rotation=45)
plt.legend()
plt.show()


# In[56]:


entities = ['Bard', 'Excel', 'TensorFlow', 'CRM', 'ChatGPT']

positive_entities_df = positive_df[positive_df['entity_name'].isin(entities)]
negative_entities_df = negative_df[negative_df['entity_name'].isin(entities)]

positive_counts = positive_entities_df['entity_name'].value_counts()

negative_counts = negative_entities_df['entity_name'].value_counts()

combined_counts = pd.concat([positive_counts, negative_counts], axis=1, sort=False).fillna(0)
combined_counts.columns = ['Positive', 'Negative']

plt.figure(figsize=(10, 6))
x = np.arange(len(entities))
width = 0.35

plt.bar(x - width/2, combined_counts['Positive'], width, color='green', label='Positive')
plt.bar(x + width/2, combined_counts['Negative'], width, color='red', label='Negative')

plt.title('Count of Product Articles in Positive and Negative Sentiment')
plt.xlabel('Product')
plt.ylabel('Count')

plt.xticks(x, entities, rotation=45)
plt.legend()
plt.show()


# In[57]:


entities = [ 'California', 'San Francisco', 'New York', 'US', 'Russia', 'China']

positive_entities_df = positive_df[positive_df['entity_name'].isin(entities)]
negative_entities_df = negative_df[negative_df['entity_name'].isin(entities)]

positive_counts = positive_entities_df['entity_name'].value_counts()

negative_counts = negative_entities_df['entity_name'].value_counts()

combined_counts = pd.concat([positive_counts, negative_counts], axis=1, sort=False).fillna(0)
combined_counts.columns = ['Positive', 'Negative']

plt.figure(figsize=(10, 6))
x = np.arange(len(entities))
width = 0.35

plt.bar(x - width/2, combined_counts['Positive'], width, color='green', label='Positive')
plt.bar(x + width/2, combined_counts['Negative'], width, color='red', label='Negative')

plt.title('Count of GPE Articles in Positive and Negative Sentiment')
plt.xlabel('GPE')
plt.ylabel('Count')

plt.xticks(x, entities, rotation=45)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





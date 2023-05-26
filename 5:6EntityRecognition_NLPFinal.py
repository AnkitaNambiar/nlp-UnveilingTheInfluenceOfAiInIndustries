#!/usr/bin/env python
# coding: utf-8

# ## Script 6, A Nambiar 
# #### (Entity Identification)

# ### General Entities 

# In[1]:


import pandas as pd
import spacy
from collections import Counter


# In[2]:


from pandarallel import pandarallel
import multiprocessing

num_processors = multiprocessing.cpu_count()
print(f'Available CPUs: {num_processors}')

pandarallel.initialize(nb_workers=num_processors-3, use_memory_fs=False, progress_bar=True)


# In[3]:


df = pd.read_parquet('sentiment_filtered_news.parquet')
df.head()


# In[4]:


#pip install ipywidgets


# In[5]:


def extract_entities(text, nlp_package_name='en_core_web_sm'):
    nlp = spacy.load(nlp_package_name)
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities


# In[6]:


pandarallel.initialize(nb_workers=num_processors-3, use_memory_fs=False, progress_bar=True)


# In[7]:


nlp = spacy.load('en_core_web_sm')  

def extract_entities(texts):
    docs = nlp.pipe(texts, disable=['parser', 'tagger'])
    return [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]


# In[8]:


from tqdm import tqdm


# In[9]:


batch_size = 32  # Adjust this value to your needs.
entities = []
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['cleaned_text'].iloc[i:i+batch_size]
    batch_entities = extract_entities(batch_texts)
    entities.extend(batch_entities)


# In[10]:


df['entities'] = entities


# In[11]:


df['organization'] = df['entities'].apply(lambda x: [entity[0] for entity in x if entity[1] == 'ORG'])


# In[12]:


top_org_entities = df.explode('organization')['organization'].value_counts().nlargest(10)

print("Top 10 Most Frequent Organization Entities:")
for entity, count in top_org_entities.items():
    print(f"Entity: {entity}\tCount: {count}")


# In[13]:


df['products'] = df['entities'].apply(lambda entities: [entity[0] for entity in entities if entity[1] == 'PRODUCT'])


# In[14]:


top_product_entities = df.explode('products')['products'].value_counts().nlargest(20)

print("Top 10 Most Frequent Product Entities:")
for entity, count in top_product_entities.items():
    print(f"Product: {entity}\tCount: {count}")


# In[15]:


df['GPE'] = df['entities'].apply(lambda entities: [entity[0] for entity in entities if entity[1] == 'GPE'])


# In[16]:


top_GPE_entities = df.explode('GPE')['GPE'].value_counts().nlargest(20)

print("Top 10 Most Frequent GPE Entities:")
for entity, count in top_GPE_entities.items():
    print(f"Product: {entity}\tCount: {count}")


# In[17]:


from collections import Counter


# In[18]:


all_entities = [entity for entities in df['entities'] for entity in entities]


# In[19]:


entity_counts = Counter(all_entities)
top_entities = entity_counts.most_common(20)


# In[20]:


for entity, count in top_entities:
    print(f'Entity: {entity}\tCount: {count}')


# In[21]:


sentiment_groups = df.groupby('predicted_sentiment_yelp_new')


# In[22]:


df.to_parquet('entity_nlp.parquet')


# ### Targeted (entity) Sentiment Identification (Start)

# In[23]:


def targeted_sentiment(text, target_entity):
    doc = nlp(text)
    sentiment = None
    for ent in doc.ents:
        if ent.text.lower() == target_entity.lower():
            sentiment = ent._.polarity
            break
    return sentiment


# In[24]:


df.head()


# In[25]:


df_exploded = df.assign(Entity=df['entities'].str.split(';')).explode('entities')


# In[26]:


df_exploded


# In[27]:


df_exploded[['entity_name', 'entity_type']] = df_exploded['entities'].str.extract(r'\(([^,]+), ([^)]+)\)')
df_exploded.head()


# In[28]:


df_exploded['entity_type'] = df_exploded['entity_type'].str.strip("'")
df_exploded['entity_name'] = df_exploded['entity_name'].str.strip("'")
df_exploded.head()


# In[29]:


len(df)


# In[30]:


len(df_exploded)


# In[31]:


sentiment_distribution = df_exploded['predicted_sentiment_yelp_new'].value_counts(normalize=True) * 100
print("Sentiment Distribution:")
print(sentiment_distribution)


# In[32]:


df_exploded['entities'] = df_exploded['entities'].astype(str)


# In[36]:


date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
date_range


# In[ ]:





# In[ ]:





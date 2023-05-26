#!/usr/bin/env python
# coding: utf-8

# ## Script 9, A Nambiar 
# ## Summarization: Articles Positive and Negative

# In[4]:


import pandas as pd
import spacy
from collections import Counter


# In[5]:


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer 


# In[6]:


df = pd.read_parquet('entity_nlp.parquet')
df.head()


# In[7]:


df.info()


# In[8]:


type(df['entities'])


# In[9]:


df['entities'] = df['entities'].astype(str)


# In[10]:


p_df = df[df['predicted_sentiment_yelp_new'] == 'p']
n_df = df[df['predicted_sentiment_yelp_new'] == 'n']


# In[11]:


summarize_pos_df = p_df[p_df['entities'].str.contains('ChatGPT')]
summarize_pos_df.head()


# In[12]:


summarize_pos_df['cleaned_text']


# In[13]:


pos_text = str(summarize_pos_df['cleaned_text'].tolist())
pos_text[:390]


# In[14]:


type(pos_text)


# In[15]:


from summarizer import Summarizer


# In[17]:


type(pos_text[:390])


# In[18]:


model = Summarizer()


# In[19]:


result = model(pos_text[:300], ratio=0.1)  # Specified with ratio
result


# #### exciting advancements the industry is experiencing.

# In[20]:


filtered_df = summarize_pos_df[summarize_pos_df['cleaned_text'].str.contains('ChatGPT') & df['cleaned_text'].str.contains('exciting adv')]
filtered_df


# In[21]:


x = filtered_df.iloc[0]['url']
print(x)


# #### Finanace Yahoo Domain
# 
# #### Title
# Sensorium to Lead Conversation on AI Virtual Beings at LEAP 2023
# 
# #### Text
# It's fantastic to have the opportunity to lead the conversation on AI and AI-driven virtual beings at a stage like LEAP, which has become a global window into the state of emerging technology and the most exciting advancements the industry is experiencing

# In[22]:


len(pos_text)


# In[23]:


sliced_text = pos_text[9429503:]
sliced_text


# In[29]:


summarizer = Summarizer()
summary = summarizer( pos_text[9429603:])
print(summary)


# In[30]:


summarize_neg_df = n_df[n_df['entities'].str.contains('ChatGPT')]
summarize_neg_df.head()


# In[33]:


neg_text = str(summarize_neg_df['cleaned_text'].tolist())
neg_text[:300]


# In[34]:


len(neg_text)


# In[35]:


result = model(neg_text[:300], ratio=0.1)  # Specified with ratio
result


# #### hurriedly and desperately aiming to bring their generative AI apps into nan marketplace

# In[36]:


filtered_df = summarize_neg_df[summarize_neg_df['cleaned_text'].str.contains('ChatGPT') & df['cleaned_text'].str.contains('hurriedly and desperately')]
filtered_df


# In[37]:


x = filtered_df.iloc[0]['url']
print(x)


# #### Central Point News
# 
# #### Title: 
# Recent ChatGPT And Bard Predicament Raises Thorny Questions About Whether Using One AI To Train A Competing AI Can Be Fair And Square, Says AI Ethics And AI Law | CENTRALPOINTNEWS
# 
# #### Text:
# "One point that we cognize for judge is that location is simply a benignant of conflict aliases warfare taking spot among nan various Artificial Intelligence (AI) makers that are hurriedly and desperately aiming to bring their generative AI apps into nan marketplace."

# In[38]:


summarize_pos_df = p_df[p_df['entities'].str.contains('health')]
summarize_pos_df.head()


# In[39]:


summarize_pos_df['cleaned_text']


# In[40]:


pos_text = str(summarize_pos_df['cleaned_text'].tolist())
pos_text[:390]


# In[41]:


type(pos_text)


# In[42]:


from summarizer import Summarizer


# In[44]:


model = Summarizer()


# In[49]:


result = model(pos_text[:200], ratio=0.1)  # Specified with ratio
result


# In[58]:


summarize_neg_df = p_df[p_df['entities'].str.contains('business')]
summarize_neg_df.head()


# In[59]:


neg_text = str(summarize_neg_df['cleaned_text'].tolist())
neg_text[:390]


# In[60]:


model = Summarizer()


# In[61]:


result = model(neg_text[:200], ratio=0.1)  # Specified with ratio
result


# In[ ]:





# In[ ]:





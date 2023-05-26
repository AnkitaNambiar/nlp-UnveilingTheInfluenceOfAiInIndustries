#!/usr/bin/env python
# coding: utf-8

# ## Script 1, A Nambiar 
# 
# #### (Preprocessing)
# 
# ### Article Clean Up

# In[2]:


get_ipython().system('pip install pyarrow')


# In[2]:


import pandas as pd
import pyarrow
import re

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 500)


# In[3]:


pip install progress


# In[4]:


pip install pandarallel


# In[5]:


from pandarallel import pandarallel
import multiprocessing
from progress.bar import Bar

num_processors = multiprocessing.cpu_count()
print(f'Available CPUs: {num_processors}')

pandarallel.initialize(nb_workers=num_processors-1, use_memory_fs=False, progress_bar=Bar())


# ## 1. Preprocessing 

# #### Load Data

# In[6]:


df_news_final_project = pd.read_parquet('https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet', engine='pyarrow')


# In[7]:


df_news_final_project.head()


# #### Initial Data Analysis

# In[8]:


df = df_news_final_project[df_news_final_project['language'] == 'en']
df.shape


# In[9]:


df.nunique()


# In[10]:


df.shape


# In[11]:


df.info()


# #### Drop Duplicates

# In[12]:


df = df.drop_duplicates(subset=['title'], keep='first')


# In[13]:


df.nunique()


# 140,223 articles in sample. 

# #### Data Sampling

# In[14]:


#df = df.sample(n=1000, random_state=42)


# #### Find Average Word, Sentence, and Article Length

# In[14]:


text_doc = df['text']
text_doc_word_lengths = text_doc.str.len()
avg_word_len_before = text_doc_word_lengths.mean()


# In[17]:


sentences = text_doc.str.split('[.!?]')
sentence_lengths = sentences.apply(len)
avg_sent_len_before = sentence_lengths.mean()


# In[18]:


# Compute the mean article length
word_counts = text_doc.apply(lambda x: len(x.split()))
article_lengths = word_counts.sum()
article_len_before = article_lengths / len(text_doc)


# In[19]:


print(f"Average word length: {avg_word_len_before} characters")
print(f"Average sentence length: {avg_sent_len_before} words")
print(f"Average article length: {article_len_before} words")


# Average word length: 9664.156850160101 characters
# 
# Average sentence length: 61.27847072163625 words
# 
# Average article length: 1327.7224278470721 words

# #### Word Cloud

# In[15]:


from wordcloud import WordCloud


# In[16]:


initial_texts = df['text']


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(initial_texts))

plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# #### Text Frequency Over Time

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace=True)

texts_per_month = df.resample('M').count()


# In[36]:


plt.figure(figsize=(10, 6))
plt.plot(texts_per_month.index, texts_per_month['text'])
plt.title('Frequency of Articles Published Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.show()


# #### Clean Text

# ##### Clean-up the noise, by eliminating newlines, tabs, remnants of web crawls, and other irrelevant text

# In[22]:


def newline_tab(text):
    # Replace newline characters with spaces
    text = re.sub(r'\n', ' ', text)
    # Replace tab characters with spaces
    text = re.sub(r'\t', ' ', text)
    return text


# In[23]:


#def article_phrase(text, phrases, max_words_after_phrase=60):
#    for phrase in phrases:
#        pattern = re.compile(r'(' + re.escape(phrase) + r')\W.*', re.IGNORECASE | re.DOTALL)
 #       match = re.search(pattern, text)
        
  #      if match:
   #         pos = match.start(1)
    #        words_after_phrase = len(re.findall(r'\w+', text[pos:]))
            
     #       if words_after_phrase <= max_words_after_phrase:
      #          text = text[:pos].strip()
    #return text


# In[24]:


common_phrases = [
    "Skip to content",
    "Skip to main content",
    "Related Articles",
    "Related Stories",
    "Advertisement",
    "Sponsored Content",
    "Share this",
    "Share on",
    "Comments",
    "Leave a Reply",
    "Subscribe",
    "Sign up",
    "About the Author",
    "Author Bio",
    "Copyright",
    "All rights reserved",
    "Source",
    "Originally published on",
    "The above press release",
    "READ MORE",
    "Read more",
    "Contact Us",
    "Read More",
    "Copyright",
    "To see more",
    "For additional information",
    "For more information"
]


# In[25]:


def article_phrase(_text, endings, position_ratio=0.75):
    for ending in endings:
        pattern = re.compile(r'(' + re.escape(ending) + r')\W.*', re.IGNORECASE | re.DOTALL)
        match = re.search(pattern, _text)
        
        if match:
            pos = match.start(1)
            
            if pos >= len(_text) * position_ratio:
                _text = _text[:pos].strip()  # Remove the ending and everything after it
            else:
                _text = _text[:pos] + _text[pos+len(ending):]  # Remove only the ending
    return _text


# In[26]:


def noisy_lines(text):
    for phrase in common_phrases:
        text = re.sub(phrase, ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# In[27]:


def remove_words_sentences_by_length(text, max_word_length=25, min_sentence_length=5, max_sentence_length=45):
    # Filter by word length (did not remove small ones, want to keep acronyms like AI and IT)
    filtered_words = [word for word in text.split() if len(word) <= max_word_length]
    text = ' '.join(filtered_words)

    sentences = sent_tokenize(text)
    filtered_sentences = [s for s in sentences if min_sentence_length <= len(s.split()) <= max_sentence_length]

    return ' '.join(filtered_sentences)


# In[28]:


def remove_unncessary(text):
    # Remove special characters and stopwords
    words = [re.sub(r'[^\w\s]', '', word) for word in text.split() if word.lower() not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text.strip()


# In[29]:


df['cleaned_text'] = df['text'].apply(newline_tab)


# In[30]:


df['cleaned_text'] = df['cleaned_text'].apply(
    lambda t: article_phrase(t, endings=common_phrases, position_ratio=0.75))


# In[31]:


df['cleaned_text'] = df['cleaned_text'].apply(noisy_lines)


# In[32]:


df['cleaned_text'] = df['cleaned_text'].apply(
    lambda t: remove_words_sentences_by_length(t, max_word_length=25, 
                                               min_sentence_length=5, max_sentence_length=45))


# In[33]:


df['important_words'] = df['cleaned_text'].apply(remove_unncessary)


# In[34]:


df.head()


# #### Lemmatizer

# In[35]:


lemmatizer = WordNetLemmatizer()


# In[36]:


def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text


# In[37]:


#df['important_words'] = df['important_words'].apply(lemmatize_text)
df['important_words'] = df['important_words'].apply(remove_unncessary)


# ##### LATER: 
# ##### - Discard irrelevant articles by Loooking at Topics
# ##### - N-grams: consider for topic modeling

# In[38]:


df.to_parquet('preprocessed_news_final_1.parquet')


# In[39]:


df.shape


# In[40]:


df.head(15)


# In[36]:


import os

cwd = os.getcwd()
print("Current working directory:", cwd)


# ## 2. Initial Exploratory Data Analysis

# In[39]:


pip install wordcloud


# In[40]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

from nltk.probability import FreqDist
import seaborn as sns


# #### Average

# In[45]:


# Calculate average word length for important words
important_words = df['important_words']
important_word_lengths = important_words.str.len()
avg_word_len_after_imp = important_word_lengths.mean()


# In[46]:


# Calculate average word length for all words
cleaned_text = df['cleaned_text']
cleaned_word_lengths = cleaned_text.str.len()
avg_word_len_after = cleaned_word_lengths.mean()


# In[47]:


# Calculate average sentence length
sentences = cleaned_text.str.split('[.!?]')
sentence_lengths = sentences.apply(len)
avg_sent_len_after = sentence_lengths.mean()


# In[50]:


# Compute the mean article length
word_counts = cleaned_text.apply(lambda x: len(x.split()))
article_lengths = word_counts.sum()
article_len_after = article_lengths / len(cleaned_text)


# In[51]:


print(f"Average word length for more important words: {avg_word_len_after_imp} characters")
print(f"Average word length for all words: {avg_word_len_after} characters")
print(f"Average sentence length: {avg_sent_len_after} words")
print(f"Average article length: {article_len_after} words")


# #### Word Cloud

# In[60]:


cleaned_texts = df['important_words']


# In[61]:


wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(cleaned_texts))

plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


#!/usr/bin/env python
# coding: utf-8

# ## Script 2, A Nambiar 
# #### (Topic Detection, Article Filtering)
# ### Topic Detection: Latent Dirichlet Allocation (LDA)

# In[1]:


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

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import gensim.downloader as api

import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install textblob')


# In[3]:


get_ipython().system('pip install spacy')


# In[4]:


get_ipython().system('pip install pyLDAvis')


# In[5]:


get_ipython().system('pip install gensim')


# In[6]:


import os
import time
import math
import re
from pprint import pprint
from textblob import TextBlob
import pandas as pd
import numpy as np


import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import multiprocessing
import string


import gensim
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()


# In[7]:


get_ipython().system('pip install pandarallel')


# In[8]:


get_ipython().system('pip install progress')


# In[9]:


from pandarallel import pandarallel
import multiprocessing
from progress.bar import Bar

num_processors = multiprocessing.cpu_count()
print(f'Available CPUs: {num_processors}')

pandarallel.initialize(nb_workers=num_processors-1, use_memory_fs=False, progress_bar=Bar())


# In[10]:


import locale
locale.getpreferredencoding = lambda: "UTF-8"


# In[11]:


df = pd.read_parquet('preprocessed_news_final_1.parquet')
df.head()


# In[12]:


df.shape


# In[13]:


get_ipython().run_cell_magic('time', '', "df['important_words'] = df['important_words'].str.lower()\ndf['tokens'] = df['important_words'].apply(lambda x: nltk.word_tokenize(x))\ndata_tokens = df['tokens'].tolist()\n")


# In[14]:


bigram = gensim.models.Phrases(data_tokens, min_count=1, threshold=1)
trigram = gensim.models.Phrases(bigram[data_tokens], threshold=1)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[15]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[16]:


# Create n-grams
data_words_bigrams = make_bigrams(data_tokens)
data_words_trigrams = make_trigrams(data_tokens)

# Combine tokens and n-grams
# data_tokens_cobnined = data_tokens_nostops + data_words_bigrams + data_words_trigrams
data_tokens_combined = data_words_trigrams


# In[17]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[18]:


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


# In[19]:


get_ipython().run_cell_magic('time', '', '\n# Creating the term dictionary of our courpus, where every unique term is assigned an index. \ndictionary = corpora.Dictionary(data_tokens_combined)\n\n# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.\ndoc_term_matrix = [dictionary.doc2bow(doc) for doc in data_tokens_combined]\n')


# In[20]:


workers = num_processors-1
workers


# In[21]:


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = LdaMulticore(corpus=doc_term_matrix,
                       id2word=dictionary,
                       num_topics=k,
                       random_state=100,                  
                       passes=10,
                       alpha=a,
                       eta=b,
                       workers=workers)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_tokens_combined, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()


# In[22]:


start_time = time.time()

def tic():
    global start_time 
    start_time = time.time()

def tac():
    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print(f'Execution time to calculate for topic {k}: {t_hour}hour:{t_min}min:{t_sec}sec'.format(t_hour,t_min,t_sec))


# In[23]:


get_ipython().run_cell_magic('time', '', "\ngrid = {}\ngrid['Validation_Set'] = {}\n# Topics range\nmin_topics = 2\nmax_topics = 10\nstep_size = 1\ntopics_range = range(min_topics, max_topics+1, step_size)\n\n# Alpha parameter\n#alpha = list(np.arange(0.01, 1, 0.3))\n#alpha.append('symmetric')\n#alpha.append('asymmetric')\nalpha = ['symmetric'] # Run for number of topics only\n\n# Beta parameter\n#beta = list(np.arange(0.01, 1, 0.3))\n#beta.append('symmetric')\nbeta = ['auto'] # Run for number of topics only\n\n\n# Validation sets\nnum_of_docs = len(doc_term_matrix)\ncorpus_sets = [# gensim.utils.ClippedCorpus(doc_term_matrix, num_of_docs*0.25), \n               # gensim.utils.ClippedCorpus(doc_term_matrix, num_of_docs*0.5), \n#                gensim.utils.ClippedCorpus(doc_term_matrix, num_of_docs*0.75), \n               doc_term_matrix]\n# corpus_title = ['75% Corpus', '100% Corpus']\ncorpus_title = ['100% Corpus']\nmodel_results = {\n                 'Topics': [],\n                 'Alpha': [],\n                 'Beta': [],\n                 'Coherence': []\n                }\n\nitr = 0\nitr_total = len(beta)*len(alpha)*len(topics_range)*len(corpus_title)\nprint(f'LDA will execute {itr_total} iterations')\n\n    \n# iterate through hyperparameters\nfor i in range(len(corpus_sets)):\n    # iterate through number of topics\n    for k in topics_range:\n        # iterate through alpha values\n        tic()\n        for a in alpha:\n            # iterare through beta values\n            for b in beta:\n                # get the coherence score for the given parameters\n                itr += 1\n                cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary, \n                                              k=k, a=a, b=b)\n                # Save the model results\n                model_results['Topics'].append(k)\n                model_results['Alpha'].append(a)\n                model_results['Beta'].append(b)\n                model_results['Coherence'].append(cv)\n                pct_completed = round((itr / itr_total * 100),1)\n#                 print(f'Completed Percent: {pct_completed}%, Corpus: {corpus_title[i]}, Topics: {k}, Alpha: {a}, Beta: {b}, Coherence: {cv}')\n        print(f'Completed model based on {k} LDA topics. Finished {pct_completed}% of LDA runs')\n        tac()\n                    \nlda_tuning = pd.DataFrame(model_results)\n#lda_tuning.to_csv(os.path.join(path_lda, 'lda_tuning_results.csv'), index=False)\n")


# In[24]:


lda_tuning_best = lda_tuning.sort_values(by=['Coherence'], ascending=False).head(1)


tuned_topics = int(lda_tuning_best['Topics'].to_string(index=False))


# Since the values for Alpha and Beta can be float, symmetric and asymmetric, we will either strip or convert to float
try:
    tuned_alpha = float(lda_tuning_best['Alpha'].to_string(index=False))
except:
    tuned_alpha = lda_tuning_best['Alpha'].to_string(index=False).strip()
    

try:
    tuned_beta = float(lda_tuning_best['Beta'].to_string(index=False))
except:
    tuned_beta = lda_tuning_best['Beta'].to_string(index=False).strip()    
    
print(f'Best Parameters: Topics: {tuned_topics}, Alpha: {tuned_alpha}, Beta: {tuned_beta}')


# In[25]:


lda_tuning_best


# **Best Parameters: Topics: 10

# In[26]:


get_ipython().run_cell_magic('time', '', "\ntuned_lda_model = LdaMulticore(corpus=doc_term_matrix,\n                       id2word=dictionary,\n                       num_topics=tuned_topics,\n                       random_state=100,\n                       passes=10,\n                       alpha=tuned_alpha,\n                       eta=tuned_beta,\n                       workers = workers)\n\ncoherence_model_lda = CoherenceModel(model=tuned_lda_model, texts=data_tokens_combined, \n                                     dictionary=dictionary, coherence='c_v')\n")


# In[27]:


# Print the Keyword in the best topic model
pprint(tuned_lda_model.print_topics())
doc_lda = tuned_lda_model[doc_term_matrix]


# In[28]:


get_ipython().run_cell_magic('time', '', "\nlda_display = gensimvis.prepare(tuned_lda_model, doc_term_matrix, dictionary, sort_topics=False, mds='mmds')\npyLDAvis.display(lda_display)\n")


# **Potential Topic List**

# Topic 0: AI and Data
# 
# Topic 1: AI and ChatGPT
# 
# Topic 2: Market Analysis and Reports
# 
# Topic 3: Search Terms
# 
# Topic 4: AI and Insurance
# 
# Topic 5: AI in Air India
# 
# Topic 6: AI and Cryptocurrency Trades
# 
# Topic 7: AI in Business and Technology
# 
# Topic 8: AI and Information
# 
# Topic 9: AI and Social Media Moderation

# Topic 0: AI in Business
# 
# AI's impact on data utilization and analysis in businesses
# Technology-driven advancements in AI applications
# The role of AI in transforming business operations and strategies
# Companies leveraging AI to drive innovation and gain a competitive edge
# Emerging trends in AI adoption by businesses across industries
# 
# Topic 1: AI and ChatGPT
# 
# Applications of AI-powered chatbots like ChatGPT
# Enhancing user experience through AI-driven conversational agents
# Exploring the capabilities and limitations of AI language models like ChatGPT
# How AI chatbots are revolutionizing customer support and service interactions
# Ethical considerations in the development and deployment of AI chat systems
# 
# Topic 2: Market Analysis and Research
# 
# Market reports and their role in analyzing industry trends and forecasts
# Leveraging data analysis and market research for informed decision-making
# The impact of market size and share on business strategies and competition
# Key insights derived from market analysis and their significance for industries
# Growth opportunities and challenges identified through market research
# 
# Topic 4: AI in the Insurance Industry
# 
# AI applications in automating insurance processes and enhancing customer experiences
# AI-driven risk assessment and fraud detection in the insurance sector
# Impact of AI on insurance pricing, underwriting, and claims management
# Innovations in insurance technology powered by artificial intelligence
# Challenges and ethical considerations in integrating AI into insurance practices
# 
# Topic 5: AI in Air India (Airlines)
# 
# Role of AI in enhancing operational efficiency and passenger experience in airlines
# Adoption of AI technologies by Air India for various functions such as scheduling and maintenance
# AI-driven advancements in airline safety and security measures
# Utilizing AI to personalize services and improve customer satisfaction in the aviation industry
# Opportunities and challenges in implementing AI solutions in the airline sector
# 
# Topic 9: Social Media Moderation and Spam Detection
# 
# Techniques and tools for detecting and combating spam on social media platforms
# Ethical considerations in moderating user-generated content using AI
# Ensuring compliance and privacy in social media content moderation

# Filter Topic 3.

# In[29]:


df['topic'] = [sorted(doc, key=lambda x: x[1], reverse=True)[0][0] for doc in doc_lda]


# In[30]:


df.head()


# In[31]:


top_words_per_topic = []
for i, topic in tuned_lda_model.show_topics(num_topics=-1, num_words=10, formatted=False):
    top_words_per_topic.append([word[0] for word in topic])

for i, topic in enumerate(top_words_per_topic):
    plt.figure(figsize=(8, 4))
    plt.bar([word for word in topic], [tuned_lda_model.get_topic_terms(i)[j][1] for j in range(10)], alpha=0.5)
    plt.title(f'Topic {i}')
    plt.xlabel('Top 10 words')
    plt.ylabel('Topic weight')
    plt.xticks(rotation=90)
    plt.tight_layout()

plt.show()


# In[72]:


assigned_topics = df['topic']  

custom_labels = ['AI in Business', 'Chat GPT', 'Market Analysis', 'Search Terms', 'AI in Insurance', 
                 'AI in Airlines', 'AI in Cryptocurrency', 'Business Technology', 'AI in Information', 'AI in Social Media']

topic_counts = assigned_topics.value_counts()

sorted_topics = topic_counts.sort_values(ascending=False)

topics = sorted_topics.index
frequencies = sorted_topics.values

plt.bar(range(len(topics)), frequencies, tick_label=custom_labels)
plt.xlabel('Topic')
plt.ylabel('Frequency')
plt.title('Topic Frequency in News Articles')
plt.xticks(rotation=90)

for i, freq in enumerate(frequencies):
    plt.text(i, freq, str(freq), ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')

plt.show()


# ### Article Filtering

# In[85]:


df[df['topic'] == 3]


# In[81]:


df.shape


# In[86]:


filtered_df = df[df['topic'] != 3]
filtered_df.head()


# In[88]:


filtered_df.shape


# In[89]:


filtered_df.to_parquet('filtered_news.parquet')


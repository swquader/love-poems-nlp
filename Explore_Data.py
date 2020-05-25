# -*- coding: utf-8 -*-
"""
Created on Sun May 17, 2020

@author: wasilaq
"""

import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


col = MongoClient().poetry.love_poems

df = pd.DataFrame(list(col.find({},{'_id': 0, 'link': 0})))
df = df[df['year'] != '']
df.reset_index(inplace=True)

df['year'] = df['year'].map(lambda x: int(x))
plt.hist(df['year'], bins=50) 
df['year'].value_counts()

def bin_year(year):
    '''
    Categorizes to a time period, given a year

    Parameters
    ----------
    year : int

    Returns
    -------
    period : str
        Output is 'pre-2000' or 'post-2000'

    '''
    if year >= 2000:
        period = 'post-2000'
    else:
        period = 'pre-2000'
    
    return period

df['period'] = df['year'].map(lambda x: bin_year(x))

corpus = df['title'] + ' ' + df['poem']

def clean_text(text):
    '''
    Given a string, lowercase, remove punctuation and remove nonsense       characters
    
    Parameters
    ----------
    text : str
        String of text to be cleaned

    Returns
    -------
    final : str
        Cleaned string
    '''
    no_lines = text.replace('\n', '')
    no_x = no_lines.replace('\xa0', '')
    no_plus = re.sub('\+', ' ', no_x)
    final = re.sub('[^\w\s]', '', no_plus.lower())
    return final

cleaned_corpus = corpus.map(lambda x: clean_text(x))

# top words
cv = CountVectorizer(stop_words='english')
cv_dtm = cv.fit_transform(cleaned_corpus)
dtm = pd.DataFrame(cv_dtm.todense(), columns=cv.get_feature_names())
dtm.index = df['title']

def word_count(doc_term_matrix):
    '''
    Count the total number of words within a document-term matrix

    Parameters
    ----------
    doc_term_matrix : dataframe
        A document-term matrix (words as columns, documents as rows)

    Returns
    -------
    ordered : list
        Returns a list of tuples containing words and word frequencies, ordered by the word with the highest frequency to the word with the lowest frequency
    '''
    word_totals = {}
    ordered = []
    for word in doc_term_matrix.columns:
        word_totals[word] = sum(doc_term_matrix[word])
    for key, value in word_totals.items():
        ordered.append(tuple([value, key]))
        
    ordered.sort(reverse=1)
    
    return ordered
    
word_count(dtm)[:20]

# top words per period
period_words = {}

for period in df['period'].unique():
    index = df[df['period']==period].index
    period_dtm = dtm.iloc[index]
    period_words[period] = word_count(period_dtm)

# part of speech for love
df['POS'] = [pos_tag(word_tokenize(poem)) for poem in cleaned_corpus]

period_pos = {'post-2000': [], 'pre-2000': []}

for poem, period in list(zip(df['POS'],df['period'])):
    for word in poem:
        if word[0] in ('Love','love'):
            period_pos[period].append(word[1])

pos_count_past = Counter(period_pos['pre-2000'])
# {'NN': 89, 'VB': 25, 'VBP': 15, 'IN': 7, 'RBR': 1, 'VBN': 1}
pos_count_pres = Counter(period_pos['post-2000'])
# {'VB': 80, 'NN': 227, 'IN': 11, 'VBP': 87, 'RB': 4, 'JJ': 4, 'VBN': 1, 'JJR': 2, 'RBR': 1}

pos_count_past['NN'] # 89, 68%
pos_count_past['VB'] + pos_count_past['VBP'] + pos_count_past['VBN'] # 41, 32%

pos_count_pres['NN'] # 227, 57%
pos_count_pres['VB'] + pos_count_pres['VBP'] + pos_count_pres['VBN'] # 168, 43%


# save findings in dataframe & pickle
df['poem_clean'] = cleaned_corpus
df.drop(columns = ['index'], inplace=True)
df.to_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/poems_df')
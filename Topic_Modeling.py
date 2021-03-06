#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:27:13 2020

@author: wasilaq
"""

import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from topic_modeling_class import DTM

df = pd.read_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/poems_df')
cleaned_corpus = df['poem_clean']

def repeat(model, start, stop, corpus, form='BOW', sw='english'):
    docs = DTM(df['title'], corpus, form, sw)
    for num in range(start, stop):
        if model == 'LDA':
            print(docs.LDA_topics(num))
        if model == 'NMF':
            print(docs.NMF_topics(num))
        print()

# LDA
repeat(2, 6, cleaned_corpus)

# remove words 'love' and 'like'
sw = stopwords.words('english') + ['love','Love','like','Like']
repeat('LDA', 2, 4, cleaned_corpus, sw=sw)


# stem
stemmed_corpus = []
for poem in cleaned_corpus:
    stemmed_poem = []
    for word in poem:
        stemmed_poem.append(LancasterStemmer().stem(word))
    stemmed_corpus.append(''.join(stemmed_poem))

repeat('LDA', 2, 4, stemmed_corpus, sw=sw)    
repeat('LDA', 3, 5, stemmed_corpus, sw=(sw+['one','know','would']))


# nouns only
nouns_corpus = []
for poem in df['POS']:
    poem_nouns = []
    for word in poem:
        if word[1] == 'NN':
            poem_nouns.append(word[0]+' ')
    nouns_corpus.append(''.join(poem_nouns))

repeat('LDA', 2, 5, nouns_corpus, sw=sw)


#try TF-IDF
repeat('LDA', 2, 9, nouns_corpus, sw=sw, form='TFIDF')


# verbs only
verbs_corpus = []
for poem in df['POS']:
    poem_verbs = []
    for word in poem:
        if 'VB' in word[1]:
            poem_verbs.append(word[0]+' ')
    verbs_corpus.append(''.join(poem_verbs))

repeat('LDA', 2, 5, verbs_corpus, sw=sw, form='TFIDF')


# new stop words
sw2 = sw + ['know','shall','heart','eyes','loves','light']

repeat('LDA', 2, 5, cleaned_corpus, sw=sw2, form='TFIDF')
repeat('LDA', 2, 6, nouns_corpus, sw=sw2, form='TFIDF')
repeat('LDA', 2, 4, verbs_corpus, sw=sw2, form='TFIDF')


#NMF
repeat('NMF', 2, 7, verbs_corpus, sw=sw2, form='TFIDF')

# new stop words
sw3 = sw2 + ['thou','thee','thy','yo']
repeat('NMF', 3, 11, verbs_corpus, sw=sw3, form='TFIDF')

repeat(2, 6, nouns_corpus, sw=sw3, form='TFIDF')
# best model: 4 topics

for poem in df.poem_clean:
    if 'derness' in poem:
        print(poem)
# 'derness' from 'tenderness'


# Final Model
'''
Topic 1 - metaphysical, spiritual
['night', 'life', 'world', 'sea', 'body', 'soul', 'wind', 'god', 'face', 'hand']

Topic 2 - affectionate
['way', 'day', 'kiss', 'home', 'breath', 'derness', 'face', 'morning', 'hope', 'front']

Topic 3 - nature, celestial
['moon', 'morning', 'sun', 'song', 'night', 'noon', 'side', 'beautiful', 'faint', 'watch']

Topic 4 - concrete, specific
['man', 'time', 'mother', 'poem', 'room', 'door', 'someone', 'father', 'everything', 'body']
'''

# add topics to dataframe of poems
doc_topic = DTM(df['title'], nouns_corpus, stopwords=sw3, form='TFIDF').NMF_df(4)

for topic in ['topic_1', 'topic_2', 'topic_3', 'topic_4']:
    df[topic] = doc_topic[topic].values

df.groupby('period').mean()[['topic_1', 'topic_2', 'topic_3', 'topic_4']]


# save findings from topic modeling
df.to_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/poems_w_topics_df')
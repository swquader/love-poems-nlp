#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:27:13 2020

@author: wasilaq
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from gensim import models, matutils

df = pd.read_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/poems_df')
cleaned_corpus = df['poem_clean']

# LDA
def LDA_topics(corpus, num_topics, stop_words_list = 'english', TFIDF = None):
    '''
    Return the topics of an LDA model, given the documents, the desired number of topics, stop words, and the type of document-term matrix
    
    Parameters
    ----------
    corpus : iterable (e.g. series, list)
        Contains all documents
    num_topics : int
        Number of topics
    stop_words_list : list
        List of all stop words
    TFIDF
        Assigned no value for bag of words document-term matrix (specifically, function will use CountVectorizer). Assign value for TF-IDF document-term matrix (function will use TfidfVectorizer)

    Returns
    -------
    Topics from LDA model

    '''
    if TFIDF == None:
        LDA_cv = CountVectorizer(stop_words=stop_words_list)
    else:
        LDA_cv = TfidfVectorizer(stop_words=stop_words_list)
        
    LDA_cv_dtm = LDA_cv.fit_transform(corpus)
    gensim_corpus = matutils.Sparse2Corpus(LDA_cv_dtm.transpose())
    id2word = dict((v,k) for k,v in LDA_cv.vocabulary_.items())

    LDA_model = models.LdaModel(corpus=gensim_corpus, num_topics=num_topics, id2word=id2word, passes=5)
    
    return LDA_model.print_topics()

for num in range(2,6):
    LDA_topics(cleaned_corpus, num)

# remove words 'love' and 'like'
sw = stopwords.words('english') + ['love','Love','like','Like']
LDA_topics(cleaned_corpus, 2, sw)
LDA_topics(cleaned_corpus, 3, sw)

# stem, then do topic modeling?
stemmed_corpus = []
for poem in cleaned_corpus:
    stemmed_poem = []
    for word in poem:
        stemmed_poem.append(LancasterStemmer().stem(word))
    stemmed_corpus.append(''.join(stemmed_poem))
    
LDA_topics(stemmed_corpus, 2, sw)
LDA_topics(stemmed_corpus, 3, sw)
LDA_topics(stemmed_corpus, 3, (sw+['one','know','would']))
LDA_topics(stemmed_corpus, 4, (sw+['one','know','would']))

# nouns only
nouns_corpus = []
for poem in df['POS']:
    poem_nouns = []
    for word in poem:
        if word[1] == 'NN':
            poem_nouns.append(word[0]+' ')
    nouns_corpus.append(''.join(poem_nouns))

for num in range(2,5):
    LDA_topics(nouns_corpus, num, sw)

#try TF-IDF
LDA_topics(nouns_corpus, 2, sw, 1)
LDA_topics(nouns_corpus, 3, sw, 1)
LDA_topics(nouns_corpus, 4, sw, 1)
LDA_topics(nouns_corpus, 8, sw, 1)

# verbs only
verbs_corpus = []
for poem in df['POS']:
    poem_verbs = []
    for word in poem:
        if 'VB' in word[1]:
            poem_verbs.append(word[0]+' ')
    verbs_corpus.append(''.join(poem_verbs))

for num in range(2,5):
    LDA_topics(verbs_corpus, 2, sw, 1)

LDA_topics(cleaned_corpus, 2, sw, 1)


# new stop words
sw2 = sw + ['know','shall','heart','eyes','loves','light']

for num in range(2,5):
    LDA_topics(cleaned_corpus, num, sw2, 1)

LDA_topics(nouns_corpus, 2, sw2, 1)
LDA_topics(nouns_corpus, 3, sw2+['thee'], 1)
LDA_topics(nouns_corpus, 5, sw2+['thee'], 1)

LDA_topics(verbs_corpus, 2, sw2+['let'], 1)
LDA_topics(verbs_corpus, 3, sw2, 1)


#NMF
def NMF_topics(corpus, num_topics, stop_words_list='english', TFIDF=None):
    '''
    Return the top words for each topic in an NMF model
    
    Parameters
    ----------
    corpus : iterable (e.g. series, list)
        Contains all documents
    num_topics : int
        Number of topics
    stop_words_list : list
        List of all stop words
    TFIDF
        Assigned no value for bag of words document-term matrix (specifically, function will use CountVectorizer). Assign value for TF-IDF document-term matrix (function will use TfidfVectorizer)

    Returns
    -------
    Topics from NMF model

    '''
    if TFIDF == None:
        NMF_cv = CountVectorizer(stop_words=stop_words_list)
    else:
        NMF_cv = TfidfVectorizer(stop_words=stop_words_list)
        
    NMF_cv_dtm = NMF_cv.fit_transform(corpus)
    model = NMF(num_topics)
    fitted_model = model.fit_transform(NMF_cv_dtm)
    
    global doc_topic
    columns = []
    for num in range(1,num_topics+1):
        columns.append('topic_' + str(num))
    doc_topic = pd.DataFrame(fitted_model.round(3), index = df['title'], columns = columns)
    
    topic_words = model.components_
    for ix, topic in enumerate(topic_words):
        print('Topic {}'.format(ix+1))
        word_list = []
        for i in topic.argsort()[::-1][:10]:
            word = NMF_cv.get_feature_names()[i]
            word_list.append(word)
        print(word_list)
    
for num in range(2,7):
    NMF_topics(verbs_corpus, num, sw2, 1)

sw3 = sw2 + ['thou','thee','thy','yo']
NMF_topics(verbs_corpus, 10, sw3, 1)
NMF_topics(verbs_corpus, 5, sw3, 1)
NMF_topics(verbs_corpus, 3, sw3, 1)

for num in range(2,6):
    NMF_topics(nouns_corpus, num, sw3, 1)
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
for topic in ['topic_1', 'topic_2', 'topic_3', 'topic_4']:
    df[topic] = doc_topic[topic].values

df.groupby('period').mean()[['topic_1', 'topic_2', 'topic_3', 'topic_4']]


# save findings from topic modeling
df.to_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/poems_w_topics_df')
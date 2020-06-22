#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:02:07 2020

@author: wasilaq
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from gensim import models, matutils


class DTM:
    def __init__(self, document_titles, documents, form, stopwords='english'):
        self.titles = document_titles
        self.documents = documents
        self.form = form
        self.stopwords = stopwords
        if form=='TFIDF':
            self.vectorizer = TfidfVectorizer(
                stop_words = stopwords
                )
        if form=='BOW':
            self.vectorizer = CountVectorizer(
                stop_words = stopwords
                )
        
    def fit_vectorizer(self):
        return (self.vectorizer).fit_transform(self.documents)
        
    def fit_LDA(self, num_topics, passes=5):
        fitted = self.fit_vectorizer()
        gensim_corpus = matutils.Sparse2Corpus(fitted.transpose())
        id2word = dict(
            (v,k) for k,v in (self.vectorizer).vocabulary_.items()
            )
        LDA_model = models.LdaModel(
            corpus=gensim_corpus, num_topics=num_topics, id2word=id2word, passes=passes
            )
        self.model = LDA_model
    
    def LDA_topics(self, num_topics, passes=5):
        self.fit_LDA(num_topics, passes)
        return (self.model).print_topics()
    
    def fit_NMF(self, num_topics):
        fitted = self.fit_vectorizer()
        model = NMF(num_topics, max_iter=1000)
        model.fit_transform(fitted)
        self.model = model
    
    def NMF_df(self, num_topics):
        self.fit_NMF(num_topics)
        fitted = self.fit_vectorizer()
        fitted_model = (self.model).fit_transform(fitted)
        columns = []
        for num in range(1,num_topics+1):
            columns.append('topic_' + str(num))
        doc_topic_df = pd.DataFrame(
                fitted_model.round(3), index = self.titles, columns = columns
                )
        return doc_topic_df
    
    def NMF_topics(self, num_topics):
        self.fit_NMF(num_topics)
        topic_words = (self.model).components_
        for i, topic in enumerate(topic_words):
            print('Topic {}'.format(i+1))
            word_list = []
            for i in topic.argsort()[::-1][:10]:
                word = (self.vectorizer).get_feature_names()[i]
                word_list.append(word)
            print(word_list)
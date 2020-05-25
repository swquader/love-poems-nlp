#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:21:31 2020

@author: wasilaq
"""


import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/poems_w_topics_df')

df['pol'] = df['poem_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)

df['subj'] = df['poem_clean'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

df['positive'] = df['poem_clean'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['pos'])

df['compound'] = df['poem_clean'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

df['negative'] = df['poem_clean'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['neg'])


df.groupby('period').mean()[['pol','subj','positive','compound','negative']]


df.to_pickle('/Users/wasilaq/Metis/love-poems-nlp/pickled/final_poems_df')
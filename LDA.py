# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:51:34 2021

@author: anton
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition, ensemble

tweet_df =  pd.read_csv(r'tweet_data_0630.csv')

# transforming data 
    #train test split
#lda_train_x, lda_test_x, lda_train_y, lda_test_y = model_selection.train_test_split(dense_dfND['tweet'],dense_dfND['vaccine_type'])

    #feature engineering
cv_lda = TfidfVectorizer(max_df = 0.9, ngram_range = (1,2), max_features=5000, min_df=100)
# transforming label variables
# lda_train_y = cv_lda.fit_transform(lda_train_y)
# lda_test_y =cv_lda.fit_transform(lda_test_y)
# #transforming set
# lda_train_x = cv_lda.fit_transform(lda_train_x)
# lda_test_x =cv_lda.fit_transform(lda_test_x)

lda_text = cv_lda.fit_transform(tweet_df.tweet)


# specific bigram to express ownership 


#model
lda_model = decomposition.LatentDirichletAllocation(n_components = 3, verbose=1, max_iter= 20 )
#mode fit/transform
X_topics = lda_model.fit_transform(lda_text)
topic_word = lda_model.components_
vocab = cv_lda.get_feature_names()


n_top_words = 10
topic_summaries = []
for topic_dist in topic_word:
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
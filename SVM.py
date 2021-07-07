# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:45:03 2021

@author: anton
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


#data upload
df = pd.read_csv(r'tweet_data.csv')


#train test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(df.tweet,df.vaccine_type,test_size=0.3)


# vectorizer

tf_vec = TfidfVectorizer()
tf_vec.fit(df.tweet)

X_test_tfidf = tf_vec.transform(X_test)
X_train_tfidf = tf_vec.transform(X_train)

#encode
le= LabelEncoder()

y_train_le = le.fit_transform(y_train)
y_test_le = le.fit_transform(y_test)


## SVM
SVM = svm.SVC(C=1.0, kernel = 'linear')

SVM.fit(X_train_tfidf,y_train_le)

predict_svm = SVM.predict(X_test_tfidf)

accuracy = accuracy_score(predict_svm,y_test_le)

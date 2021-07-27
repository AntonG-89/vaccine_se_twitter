# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:45:03 2021

@author: anton
"""

import pandas as pd
import numpy as np
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#data upload
df = pd.read_csv(r'ready_to_go_1020_220721.csv')


#train test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(df.tweet,df.vaccine_type,test_size=0.3)


# vectorizer

tf_vec = TfidfVectorizer(stop_words='english')
tf_vec.fit(df.tweet)

X_test_tfidf = tf_vec.transform(X_test)
X_train_tfidf = tf_vec.transform(X_train)

#encode
le= LabelEncoder()

y_train_le = le.fit_transform(y_train)
y_test_le = le.fit_transform(y_test)


## SVM
SVM = svm.SVC(C=1, kernel = 'rbf', class_weight='balanced')

SVM.fit(X_train_tfidf,y_train_le)

predict_svm = SVM.predict(X_test_tfidf)

mtrs = confusion_matrix(predict_svm,y_test_le)
accuracy = accuracy_score(predict_svm,y_test_le)
precision = precision_score(predict_svm,y_test_le, average='weighted')
recall = recall_score(predict_svm,y_test_le, average='weighted')

## add mtrs visalization and ROC curve?
label = ['azn','moderna','mod-pfizer','pfizer']
sns.heatmap(mtrs.T,square=True,annot=True, fmt='d', cbar = False,xticklabels=label, yticklabels =label)
plt.xlabel('true')
plt.ylabel('predicted')




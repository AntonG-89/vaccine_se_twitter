# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:40:27 2021

@author: anton
"""


import pandas as pd
from sklearn import model_selection,preprocessing,metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



tweet_df = pd.read_csv(r'tweet_data_0705.csv')
clean_tweets = tweet_df.tweet.copy()


#getting dict
cv_dict = TfidfVectorizer(ngram_range = (1,2))

terms = cv_dict.fit_transform(clean_tweets)
vec = np.transpose(terms.toarray())
index = cv_dict.get_feature_names()
df  = pd.DataFrame(vec, index = index)

freq_dict = {}

for i in range(0,len(df)-1):
    freq_dict[df.index[i]] = sum(df.iloc[i,])

#splitting by class labels (vaccine types)
pfizer_df = tweet_df.loc[tweet_df['vaccine_type'] == 'pfizer'].copy()
moderna_df = tweet_df.loc[tweet_df['vaccine_type'] == 'moderna'].copy()
azn_df = tweet_df.loc[tweet_df['vaccine_type'] == 'astrazeneca'].copy()
#need to add stats for each DF

# 2x2
pfi_mod = [pfizer_df, moderna_df]
pfi_azn = [azn_df,pfizer_df]
mod_azn =[moderna_df, azn_df]

pfi_mod_df = pd.concat(pfi_mod)
pfi_azn_df = pd.concat(pfi_azn)
mod_azn_df = pd.concat(mod_azn)
#
tse_df = tweet_df[['tweet','vaccine_type']]

#train test split needs to be done before vectorization

#rebalance data/

#train_x, test_x , train_y, test_y 
train, test = model_selection.train_test_split(tse_df,test_size = 0.2, random_state = 10)


# tfidif, stop words remove, no ngrams

## TF-IDF: peforms at 60; min_df takes it to 77; Interesting to note that ngram_range (2,2) without min_df provides a boost that is deminished by min_df over 100
model = make_pipeline(TfidfVectorizer(min_df = 300, ngram_range=(1,2)), MultinomialNB(alpha = 2))


model.fit(train.tweet,train.vaccine_type)

labels = model.predict(test.tweet)

mtrs = metrics.confusion_matrix(test.vaccine_type,labels)
accuracy = metrics.accuracy_score(test.vaccine_type,labels)

# The confusion matrix was labeled wrong, as you noted in our last meeting. I have since made changes to fix that issue.
sns.heatmap(mtrs.T,square=True,annot=True, fmt='d', cbar = False,xticklabels=model.classes_, yticklabels = model.classes_)
plt.xlabel('true')
plt.ylabel('predicted')

# fresh data for classification/ all three search tags present
new_tweets = pd.read_csv(r'tweet_data_0705.csv')

#vectorizing new data


new_labels = model.predict(new_tweets.tweet)
accuracy1 = metrics.accuracy_score(new_tweets.vaccine_type,new_labels)
mtrs1 = metrics.confusion_matrix(new_tweets.vaccine_type,new_labels)

sns.heatmap(mtrs1.T,square=True,annot=True, fmt='d', cbar = False,xticklabels=model.classes_, yticklabels = model.classes_)
plt.xlabel('true')
plt.ylabel('predicted')

#df+predictions



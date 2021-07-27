
"""
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



tweet_df = pd.read_csv(r'ready_to_go_1020_220726.csv')
clean_tweets = tweet_df.tweet.copy()


#getting dict
cv_dict = TfidfVectorizer(stop_words = 'english')

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

#
tse_df = tweet_df[['tweet','vaccine_type']]

#train test split needs to be done before vectorization

#rebalance data/

#train_x, test_x , train_y, test_y 
train, test = model_selection.train_test_split(tse_df,test_size = 0.3, random_state = 10)


# tfidif, stop words remove, no ngrams

## pipeline model
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())


model.fit(train.tweet,train.vaccine_type)

labels = model.predict(test.tweet)

mtrs = metrics.confusion_matrix(test.vaccine_type,labels)
accuracy = metrics.accuracy_score(test.vaccine_type,labels)

# The confusion matrix was labeled wrong, as you noted in our last meeting. I have since made changes to fix that issue.
sns.heatmap(mtrs.T,square=True,annot=True, fmt='d', cbar = False,xticklabels=model.classes_, yticklabels = model.classes_)
plt.xlabel('true')
plt.ylabel('predicted')


# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:44:30 2021

@author: anton
"""

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer


#loading scraped df
scrape_df = pd.read_csv(r'dDf_102020_to_2021-07-26.csv')


# dtype conversion
scrape_df['id'] = scrape_df['id'].astype('int64')
scrape_df['conversation_id'] = scrape_df['conversation_id'].astype('int64')
scrape_df['date'] = scrape_df['date'].astype('datetime64[ns]')

# relabeling tweets to aggregate all vaccine type labels/groupby takes care of duplicate tweets by id
labeled_df = scrape_df.groupby(['id','tweet','date'])['vaccine_type'].apply(', '.join).reset_index()

# adjusting label column dtype and formating
labeled_df.vaccine_type = [tweet.split() for tweet in labeled_df.vaccine_type]


#using cosine_similarity to find near duplicates:
    #near duplicates will be given an index, which then will be used in a groupby.apply combo to aggregate
    ## Filtering tweets by cosine similarity
vectorizer = TfidfVectorizer()
tfidf_tweet = vectorizer.fit_transform(labeled_df.tweet)

#creating an empty column to hold cos_sim index
labeled_df['cs_id'] = ''

def cs_dedup():
    cs_index = 0
    for tweet in tfidf_tweet:
        #finding cos similarity arrat per tweet
        cos_array = cosine_similarity(tweet,tfidf_tweet)
        #finding indexes of similar tweets to query tweet/ 0.7 measure is arbitrary
        sim_array = np.where(cos_array > 0.7)
        #plugging in the index array to labeled_df to re-index tweets

        labeled_df.loc[sim_array[1],'cs_id'] = cs_index
        cs_index +=1 

cs_dedup()
#inspecting duplicates
cs_dups = labeled_df[labeled_df.duplicated(subset='cs_id',keep = False)]
        
#dropping near duplicates
labeled_df = labeled_df.drop_duplicates(subset='cs_id').reset_index()



# text cleaning

tweetSeries = labeled_df.tweet.copy()



wl = WordNetLemmatizer()
ps = PorterStemmer()
#add rule to filter 'no{keyword}'

clean_tweets = []

#need to filter out actual labels words (pfizer, moderna, astrazeneca)
def cleaner(tweet_text, clean_list):
    for i in range(0,len(tweet_text)):
        #filter out @handle pattern
        no_at = filter(lambda x: x[0] != '@',tweet_text[i].split())
        # remove emojis/TBD
        
        #replace '2nd'/'1st' token for more consistency
        shot_num = [word.replace('2nd','second').replace('1st','first') for word in no_at]
    

        
        
        #tokens = [token.strip('.,![]{}()\'\"#') for token in shot_num if token.isalnum() and token.isascii()]
        
        tokens = [token.strip('.,![]{}()\'\"#/') for token in shot_num]
        more_cl = [token.replace('(','').replace(':','') for token in tokens]
        #isalnum removes important info because of brackets and stuff: Had a first shot (Moderna) 
        isasc_low = [token.lower() for token in more_cl if token.isascii()]
        isaln = [token for token in isasc_low if token.isalnum()]
        no_label = [token for token in isaln if not token in ['pfizer','moderna','astrazeneca']]
        
        #stemming/ TBD implement POS tagger to replace with lemmatizer
        #lstem_tweets = [ps.stem(token) for token in tokens]
        
        #converting back to string          
        c_tweets = ' '.join(map(str,no_label))

        clean_list.append(c_tweets)

cleaner(tweetSeries,clean_tweets)

labeled_df['tweet'] = clean_tweets
labeled_df = labeled_df.drop('index', axis = 1)

# Working with labels

all_labels = [vaccines for vaccines in labeled_df.vaccine_type]

for idx,label in enumerate(all_labels):
    cl_labels = [vaccine.strip(',') for vaccine in label]
    all_labels[idx] = sorted(set(cl_labels))

labeled_df['vaccine_type'] = all_labels

# sort by label len/ delete all len 3 as they are generic. keep1, split 2 into duplicate tweets/diff labels

#dropping all tweets w/ 3 labels 
labeled_df = labeled_df[labeled_df.vaccine_type.map(len)<3]

#isolating and checking len2
double_df = labeled_df[labeled_df.vaccine_type.map(len)==2]

#double_df.to_csv('2tweet.csv')

#splitting double categories into duplicate tweets with separate labels 
# labeled_df.vaccine_type = [tuple(vaccine) for vaccine in labeled_df.vaccine_type]
# labeled_df = labeled_df.explode(column='vaccine_type')

## NEW LABELING SCHEMA: cast everything as str, let 2labels be actual labels
#recasting as str
labeled_df.vaccine_type = labeled_df.vaccine_type.astype('str')

clean_labs = [label.strip("[]").replace("'",'') for label in labeled_df.vaccine_type]
labeled_df.vaccine_type = clean_labs

#dropping labels of low counts
v_counts = labeled_df.vaccine_type.value_counts().reset_index()

for i in range(0,len(v_counts)):
    #identifying labels w/ low counts | 100 count is arbitrary, based on data
    if v_counts.iloc[i,1] < 100:
        #finding index of rows w/ low vaccine_types
        drop_idx = labeled_df[labeled_df['vaccine_type'] == v_counts.iloc[i,0]].index
        #dropping the rows by index
        labeled_df = labeled_df.drop(drop_idx)




#next best dataset
labeled_df.to_csv('ready_to_go_1020_220726.csv')

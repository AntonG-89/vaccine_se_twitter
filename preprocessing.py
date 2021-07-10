# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:44:30 2021

@author: anton
"""

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer


#loading scraped df
scrape_df = pd.read_csv(r'dDf_2021-07-10.csv')


# dtype conversion
scrape_df['id'] = scrape_df['id'].astype('int64')
scrape_df['conversation_id'] = scrape_df['conversation_id'].astype('int64')
scrape_df['date'] = scrape_df['date'].astype('datetime64[ns]')

# relabeling tweets to aggregate all vaccine type labels/groupby takes care of duplicate tweets by id
labeled_df = scrape_df.groupby(['id','tweet'])['vaccine_type'].apply(', '.join).reset_index()

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
        #add try except for no matches
        if len(sim_array[1]) > 0:
            labeled_df.loc[sim_array[1],'cs_id'] = cs_index
            cs_index +=1 
        else:
            continue
        
        
#dropping near duplicates
labeled_df = labeled_df.drop_duplicates(subset='cs_id').reset_index()



# text cleaning

tweetSeries = labeled_df.tweet.copy()



wl = WordNetLemmatizer()
ps = PorterStemmer()
#add rule to filter 'no{keyword}'

clean_tweets = []
def cleaner(tweet_text, clean_list):
    for i in range(0,len(tweet_text)):
        #filter out @handle pattern
        no_at = filter(lambda x: x[0] != '@',tweet_text[i].split())
        # remove emojis/TBD
        
        #replace '2nd'/'1st' token for more consistency
        shot_num = [word.replace('2nd','second').replace('1st','first') for word in no_at]
    
        #strip , . ! ( ) {} [] '' "" if alnum/ isacii to filter non -latinic words
        tokens = [token.strip('.,![]{}()\'\"#') for token in shot_num if token.isalnum() and token.isascii()]
        

        #stemming/ TBD implement POS tagger to replace with lemmatizer
        lstem_tweets = [ps.stem(token) for token in tokens]
        
        #converting back to string          
        c_tweets = ' '.join(map(str,lstem_tweets))

        clean_list.append(c_tweets)

cleaner(tweetSeries,clean_tweets)

labeled_df['tweet'] = clean_tweets

# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:59:01 2021

@author: anton
"""
import twint 
import nest_asyncio
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


tag_list = ['covid','covid19','covid_19',]
vaccine_list = ['pfizer', 'moderna', 'astrazeneca']
side_effect_list = ['fatigue','tired','headache','chills', 'sore', 'fever', 'runny','diarrhea']


# async to fix an twin lib issue
nest_asyncio.apply()
#creating scraper instance

def topic_scrape(tags, vaccines, side_effects):
    df = pd.DataFrame()
    for tag in tags:
        for vaccine in vaccines:
            
            for effect in side_effects:
                
                c= twint.Config()
   
    #break down search by #vaccinename 1 side effect from the common list (up to 5) -> compare BagofWords results, mine new words?
    
    #can the ratio of same tweets coming up for different search words used statistically?
                print(' \n This is the search string: {0},{1},{2}'.format(tag,vaccine, effect))
                c.Search ='{0},{1},{2}'.format(tag,vaccine,effect) # super set per vaccine: subset per 1 common side effect
                c.Limit = 1000
                c.Store_csv = True
                
                #c.Since = '2021-06-02'
                #c.Until = '2021-06-03'
                c.Pandas = True
                c.Lang = 'en'
                c.Count = True
##
#.Resume = 'covid_moderna_fever1_today.txt'
                
                twint.run.Search(c)
                effect_df = twint.storage.panda.Tweets_df
                #adding class labels
                
                # need better rule to label/
                effect_df['covid_tag'] = tag
                effect_df['vaccine_type'] = vaccine
                effect_df['side_effect'] = effect
                
                if len(df) == 0:
                    df = effect_df
                    
                else:     
                          
                    df = df.append(effect_df)
                    
                    
        #effect_df append to df of vaccine name
    #vaccine name df appended to a full df?
    #full df return
    return df
        
# indexing is based on amount of results returned for each combination. I might use it for labeling, 
#but need to have a separate index column

full_df = topic_scrape(tag_list, vaccine_list, side_effect_list)

#full_df.to_csv('tweet_df.csv')

# setting a regular index, 'index' column corresponds to search combinations
#full_df = full_df.reset_index()
# data exploration
full_df.info(verbose = True)
# entries in df need to be filtered to reflect general tweet vs side-effect description by users
# extracting id0, conversation_id 1,created_at 2,date 3, tweet 6, hashtags 8,nlikes 22
dense_df = full_df[['id', 'conversation_id','date','tweet','hashtags','nlikes','vaccine_type']].copy()
#analysis
dense_df.dtypes

#dtype conversion
dense_df['id'] = dense_df['id'].astype('int64')
dense_df['conversation_id'] = dense_df['conversation_id'].astype('int64')
dense_df['date'] = dense_df['date'].astype('datetime64[ns]')

#dropping tweets with the same tweet id/ using bitwise negation for speed as this might need to run online
dense_dfND = dense_df[~dense_df.id.duplicated(keep = 'first')]
#reindexing densedfND
dense_dfND = dense_dfND.reset_index()

# Tweet series
tweetSeries = dense_dfND.tweet.copy()


#preprocessing steps
# Tokenization/ Lemmatization


wl = WordNetLemmatizer()
ps = PorterStemmer()
#add rule to filter 'no{keyword}'

clean_tweets = []
def cleaner(tweet_text, clean_list):
    for i in range(0,len(tweet_text)):
        #filter out @handle pattern
        no_at = filter(lambda x: x[0] != '@',tweet_text[i].split())
        # remove emojis
        
        no_stop = [word for word in no_at if not word in stopwords.words()]
        
        #strip , . ! ( ) {} [] '' "" if alnum/ isacii to filter non -latinic words
        tokens = [token.strip('.,![]{}()\'\"#') for token in no_stop if token.isalnum() and token.isascii()]
        
        #lemmatizaton
        #llemma_tweets = [wl.lemmatize(token) for token in tokens]
        #stemming
        lstem_tweets = [ps.stem(token) for token in tokens]
        
        #converting back to string          
        c_tweets = ' '.join(map(str,lstem_tweets))
        #need to convert back to string after clean and before append
        clean_list.append(c_tweets)

cleaner(tweetSeries,clean_tweets)

dense_dfND['tweet'] = clean_tweets

dense_dfND.to_csv('tweet_data_0705.csv')




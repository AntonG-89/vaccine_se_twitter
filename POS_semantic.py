# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:58:15 2021

@author: anton
"""

import twint 
import nest_asyncio
import pandas as pd

nest_asyncio.apply()
#creating scraper instance

tag_list = ['covid','covid19','covid_19',]
vaccine_list = ['pfizer', 'moderna', 'astrazeneca']
side_effect_list = ['fatigue','tired','headache','chills', 'sore', 'fever', 'runny','diarrhea']

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
                c.Limit = 100
                c.Store_csv = True
                
                c.Since = '2021-07-03'
                c.Until = '2021-07-05'
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

tw_df = topic_scrape(tag_list,vaccine_list,side_effect_list)

tw_df =tw_df.reset_index()
tw_df = tw_df.drop('index', axis=1)

main_df = tw_df.copy()

from nltk.tokenize import word_tokenize
tw_df['clean_tweet'] = [word.split() for word in tw_df['tweet']]

# for some stupid reason this needs to run several times
for tweet in tw_df['clean_tweet']:
    for idx, word in enumerate(tweet):      
        if word[0] == '@':
            tweet.pop(idx)
            #print(word)
            
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

#this drops 1st/2nd tokens. Needs to be changed to first/second

for index,entry in enumerate(tw_df['clean_tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    tw_df.loc[index,'tweet_tags'] = str(Final_words)

        
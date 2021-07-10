# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:44:28 2021

@author: anton
"""

import twint 
import nest_asyncio
import pandas as pd
from datetime import date


pn_list = ['I','me','my']
tag_list = ['covid','covid19','covid_19',]
vaccine_list = ['pfizer', 'moderna', 'astrazeneca']
side_effect_list = ['tired','headache','chills', 'sore', 'fever']


# async to fix an twin lib issue
nest_asyncio.apply()
#creating scraper instance


#accumulating all tag combination search results into a df
def topic_scrape(prns,tags, vaccines, side_effects):
    df = pd.DataFrame()
    for pn in pn_list:
        for tag in tags:
            for vaccine in vaccines:
                
                for effect in side_effects:
                    
                    c= twint.Config()
       
        
                       
                    print(' \n This is the search string: {0},{1},{2}'.format(tag,vaccine, effect))
                    c.Search ='{0},{1},{2}'.format(tag,vaccine,effect) # super set per vaccine: subset per 1 common side effect
                    c.Limit = 1000
                    c.Store_csv = True
                    
                    c.Since = '2021-05-01'
                    c.Until = '2021-07-01'
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
                    
                    

    return df
        

# dumping all scrape results into a df
full_df = topic_scrape(pn_list,tag_list, vaccine_list, side_effect_list)


# setting a regular index, 'index' column corresponds to search combinations
full_df = full_df.reset_index()

# extracting id0, conversation_id 1,created_at 2,date 3, tweet 6, hashtags 8,nlikes 22
dense_df = full_df[['id', 'conversation_id','date','tweet','hashtags','nlikes','vaccine_type']].copy()

#this is a bit misleading since this date inst tied in with search params, which can be done, if needed
today = str(date.today())
dense_df.to_csv('dDf_{0}.csv'.format(today))


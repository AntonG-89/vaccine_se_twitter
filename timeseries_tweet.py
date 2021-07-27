# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:05:25 2021

@author: anton
"""

import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer
##reading in the df
base_df = pd.read_csv(r'ready_to_go_1020_220726.csv')


#cut date to YYmmDD
base_df.date = [date for date in base_df.date.dt.date]

#converting date dtype to date
base_df.date = base_df.date.astype('datetime64[ns]')

# get month and date range
m_range = pd.date_range(start=min(base_df.date), end=max(base_df.date),freq='M')
d_range = pd.date_range(start=min(base_df.date), end=max(base_df.date),freq='D')
#date organizing logic

def term_timeline(index,df):
    # creating an empty dataframe  
    ts_term_df = pd.DataFrame()
    #for date in day_index
    for date in index:
        #date filter/new df for each day
        daily_df = df[df['date'] == date]
         #implement an if statement to skip days with no data
        if len(daily_df) > 0:
       
            #count vectorizer instance
            cv = CountVectorizer(stop_words = 'english',ngram_range=(1,2))
            terms = cv.fit_transform(daily_df.tweet)
            vec = np.transpose(terms.toarray())
            f_names = cv.get_feature_names()
            vec_df = pd.DataFrame(vec,index=f_names)
            #new_df: index, date col, term col, freq col
              
            #creating a frequency dict
            frequency_dict = {}
    
            for i in range(0,len(vec_df)-1):
                frequency_dict[vec_df.index[i]] = sum(vec_df.iloc[i,])
                
            #converting f dict to df    
            dterms = pd.DataFrame.from_dict(frequency_dict, orient = 'index').reset_index()
            #adding date for each df
            dterms['date'] = date

            #changin column names
                
            dterms = dterms.rename(columns={'index':'terms', 0:'t_freq'})

            #sorting and isolating top terms            
            dterms = dterms.sort_values(by = ['t_freq'], ascending = False).head(20)
            
            #controling for first/next df
            if len(ts_term_df) == 0:
                 ts_term_df = dterms
            else:
                ts_term_df = ts_term_df.append(dterms)
    #0:t_freq renaming only works outside of the function for some reason            
    
    return ts_term_df

# monthy tweet counts
from collections import Counter
#using Counter function to aggregate duplicate month as each entry == tweet
tweet_counts = Counter(base_df.date.dt.month)


     
ts_df = term_timeline(d_range,base_df)
## use month range to group days by, aggregate all token sums? Can Tableu can be used?
ts_df.to_csv('ts_df_1020_220726.csv')









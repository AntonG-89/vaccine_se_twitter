# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:01:31 2021

@author: anton
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# sns.set(font_scale=1.5)
# sns.set_style("whitegrid")




df = pd.read_csv(r'ready_to_go_1020_220721.csv')

all_labels = df.vaccine_type.unique()
lab_counts = df.vaccine_type.value_counts()

# assigning different labels
df_pfizer = df[df['vaccine_type'] == 'pfizer'].copy()
df_moderna = df[df['vaccine_type'] == 'moderna'].copy()
df_azn = df[df['vaccine_type'] == 'astrazeneca'].copy()
df_pfi_mod = df[df['vaccine_type'] == 'moderna, pfizer'].copy()
# df_mod_azn = df[df['vaccine_type'] == 'astrazeneca, moderna'].copy() 
# df_pfi_azn = df[df['vaccine_type'] == 'pfizer, astrazeneca'].copy() 



def fr_dict(dframe):
    #ngrams 1-2, min_df 1% -  provides a strong way to normalize the term dictionaries size/ log normalization needed for statistical inf/plotting?
    cv = CountVectorizer(stop_words = 'english',ngram_range=(1,2),min_df= 0.01)
    terms = cv.fit_transform(dframe.tweet)
    vec = np.transpose(terms.toarray())
    index = cv.get_feature_names()
    
    #replace column index with words?
    df_cv  = pd.DataFrame(vec, index = index)
    
    frequency_dict = {}
    
    for i in range(0,len(df_cv)-1):
        frequency_dict[df_cv.index[i]] = sum(df_cv.iloc[i,])
    
    return frequency_dict, df_cv

# Pfizer stats
pfizer_dict, pfizer_cv = fr_dict(df_pfizer)

#Moderna Stats
moderna_dict, moderna_cv=fr_dict(df_moderna)

# Astrazeneca Stats
azn_dict, azn_cv = fr_dict(df_azn)

# ModPfi
mod_pfi_dict, mf_cv = fr_dict(df_pfi_mod)


# Plot horizontal bar graph
#dict to df for ease of plotting
# pfizer_dict.sort_values(by='count').plot.barh(x='words',
#                       y='count',
#                       ax=ax,
#                       color="purple")

# ax.set_title("Common Words Found in ")


# plt.show()
# import csv

# dict_list = [mod_pfi_dict]
# dict_names = ['mod_pfi_dict']
# def dict_writer(dict_list,dict_name):
#     for idx,t_dict in enumerate(dict_list):
#         with open('{0}.csv'.format(dict_names[idx]), 'w') as csv_file:  
#             writer = csv.writer(csv_file)
#             for key, value in t_dict.items():
#                 writer.writerow([key, value])
            
            
# dict_writer(dict_list,dict_names)            

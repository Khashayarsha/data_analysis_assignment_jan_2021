# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:02:58 2021

@author: XHK
"""


"""
NB the naming of this module is misleading: it does not visualize much. Mainly just sorts data and makes 
it accessible for other modules.

This module takes the data as supplied by the course (.txt files) from the specified relative locations and 
puts it all in three different (one for initial campaign data,   one for our hand-in allocations and results,   
                                and one which is padded with 'good' observations that we've attained. 
                                
It has one 'get_data()'  function at the end so other modules can invoke this module and then get the relevant data. 
It also .pkls the data.         """










import numpy as np 
import pandas as pd 
import os

#loading the data. Assuming you have folder-structure similar to how it is on Canvas

initial_data = r"C:\Users\XHK\Desktop\VU2020\Periode3\Case in Econometrics and Data Science\Initial_Data"
submitted_data_path =  r"C:\Users\XHK\Desktop\VU2020\Periode3\Case in Econometrics and Data Science\previous_days"
results_path = r"C:\Users\XHK\Desktop\VU2020\Periode3\Case in Econometrics and Data Science\Polls"
init_camps =  [r"initial_campaign_data_party_"+str(i)+".txt" for i in range(1,22)]






#puts all relevant .txt files in arrays of pd.DataFrame
demographics = pd.read_csv(initial_data+r"\demographics.txt",  header="infer")
initial_results = pd.read_csv(initial_data+r"\initial_campaign_results.txt", header = 'infer')
initial_campaigns = [pd.read_csv(initial_data+ r"\initial_campaigns\\" +  campaign, header = 'infer' ) for campaign in init_camps]
submitted_data = [pd.read_csv(submitted_data_path +r"\\"+data, names = ['A1','A2','A3']) for data in os.listdir(submitted_data_path)]
results = [pd.read_csv(results_path +r"\\"+poll, header = 'infer') for poll in os.listdir(results_path)]



for day_i in range(len(results)):
    results[day_i]['Rank'] = results[day_i][results[day_i].columns[1:17]].rank(axis=1, method = 'min', pct = False)['Team 13']
    results[day_i]['RankPct'] = results[day_i][results[day_i].columns[1:17]].rank(axis=1, method = 'min', pct = True)['Team 13']
    
print("Loaded all relevant data \n")
print("Lengths of  intial_campaigns, submitted_data, results respectively ") #for debugging:
print([len(array) for array in (initial_campaigns, submitted_data, results)])

#loading data done. Analysis: 
import sklearn 
import matplotlib.pyplot as plt




#Following block checks the demographics data for normality using Shapiro Wilks and QQ plots

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
#Checking for normality with a QQ-plot
for i in range(1,16):
    data = demographics.iloc[:,i] 
    qqplot(data, line='s')
    plt.show()
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Sample looks Gaussian (fail to reject H0)')
    else:
    	print('Sample does not look Gaussian (reject H0)')
    #Conclusion:  All demographics data looks Gaussian
    
dem_corr = demographics.corr()
corr_heat = sns.heatmap(dem_corr,annot=True)

import itertools
permutations = list(itertools.product([0,1], repeat = 3))

 
sums_to_one = [np.array(i)/sum(i) if sum(i)>0 else np.array(i) for i in permutations ]
sums_to_one = np.array(sums_to_one)

allocation = np.array([0.3,0.3,0.4])
def closest_vec(allocation, a = sums_to_one):
    #not used, really. Artifact of earlier stuff, but scared to remove last-minute. 
    distances = [np.linalg.norm(allocation-row) for row in a]
    return np.argmin(distances)


print(closest_vec(allocation))

initial_results_data = initial_results.iloc[:,1:22] #without abstention and city-strings. 
Y_init = 100*initial_results_data.div(initial_results_data.sum(axis=1), axis=0) #percentages of the non-absentee votes

demo_data = demographics[demographics.columns[1:]]  
import demog_clusters
labels, centers = demog_clusters.get_cluster_labels_and_centres()
demo_data['Cluster'] = labels
num_embed = 3
demog_embeddings = demog_clusters.get_TSNE_embeddings(n = num_embed)
#remove the (column) mean from the embeddings. 
demog_embeddings = demog_embeddings - demog_embeddings.mean(axis = 0)
col_names = ['E'+str(i) for i in range(num_embed) ]

for name,column in zip(col_names, demog_embeddings.T):
    demo_data[name] = column
        

#I think the following block was for testing purposes, so just ignore it :-)

Y_20 = Y_init['Party 20']
X_20 = initial_campaigns[19][['A1','A2','A3']] #the campaigns of party 20
X_20['dummies'] = X_20.apply(closest_vec, axis = 1)
df1 = pd.get_dummies(X_20['dummies'].astype(str))    #converts the integer dummies into vector representation, dropping one to avoid dummy trap
#X_20 = X_20.append(df1) 
#X_20 = X_20.drop('dummies')
dummies = [str(i) for i in range(1,8)]
X_20[dummies] = df1
X_20 = X_20.drop(['dummies'], axis = 1) #drop the integer-dummies, because we use vectorized




#                  process initial_data into one big DataFrame:
df = pd.DataFrame()


useRelativePercentage = False     # set to True if one wants to consider the percentage of votes a party got of actual voters
                                  # When set to False: percentages of POTENTIAL votes (so including abstention) are used.

#This for-loop puts all the initial-campaign data and results in 1 pd.DataFrame() object for easy analysis 
for i, dataframe in enumerate(initial_campaigns):
    party_nr = i+1
    dataframe['Party'] = party_nr
    #dataframe['Dummy'] = dataframe[['A1','A2','A3']].apply(closest_vec,axis=1)  
    results_column = 'Party '+str(party_nr)
    dataframe['Results'] = initial_results_data[results_column]
    dataframe['Abstention'] = initial_results['Abstention']
    dataframe['Day'] = 0
    dataframe[list(demo_data.columns.values)] = demo_data
    df = df.append(dataframe)

submitted_data_and_results = pd.DataFrame()
municipalities = initial_campaigns[0]['Municipality']
submitted_data[0].index = submitted_data[1].index     #this caused problems if not done. 



#This for-loop puts all the submitted campaign data and results in 1 pd.DataFrame() object for easy analysis 
for i, dataframe in enumerate(submitted_data):
    print('processing day '+str(i+1))
    
    temp_df = pd.DataFrame()
    temp_df['Municipality'] = municipalities.values
    day_nr = i+1   
    #temp_df[['A1','A2','A3']] = 0
    temp_df[['A1','A2','A3']] = dataframe[['A1','A2','A3']] 
    #temp_df['Dummy'] = temp_df[['A1','A2','A3']].apply(closest_vec,axis = 1)
    temp_df['Party'] = 13
    
    vote_shares = results[day_nr-1][results[day_nr-1].columns[1:17]]
    #vote_shares = 100*vote_shares.div(vote_shares.sum(axis=1), axis=0)
    temp_df['Results'] = vote_shares['Team 13']
    temp_df[list(demo_data.columns.values)] = demo_data
    temp_df['Abstention'] = results[day_nr-1]['Abstention']
    temp_df['Rank'] = results[day_nr-1]['Rank'] 
    temp_df['RankPct'] = results[day_nr-1]['RankPct']
    temp_df['Cluster'] = labels
    temp_df['Day'] = day_nr
    submitted_data_and_results = submitted_data_and_results.append(temp_df)

def rank_init_data(y, percentage = True):
    y.index = range(len(y)) #removes duplicate_indexes
    y['Rank'] = y.groupby(by = 'Municipality')['Results'].rank(method = 'min', pct = False)
    y['RankPct'] = y.groupby(by = 'Municipality')['Results'].rank(method = 'min', pct = True)

init_data = df
rank_init_data(init_data)
other_data = submitted_data_and_results




#appends submitted campaigns and results to initial campaigns and results:
total_data = init_data.append(other_data)





# below code-block pads one of the output dataframes with those ovbservations that have attained high rank.
# our hope was to bias the Neural Net to learn the higher-ranking allocations more.
top_init = init_data[init_data['RankPct'] >0.80]
top_init.index = range(len(top_init))
b= pd.concat([top_init] *2, ignore_index = True)
top_init = b
top_init.index = range(len(top_init))

temp = pd.concat([other_data[other_data['RankPct'] >0.80]]*4, ignore_index = True)
total = top_init.append(temp)
total = total.append(other_data)
total.index = range(len(total))






def get_data(initial_data = init_data, othr_data = other_data, total=total):
    """This function basically let's other modules initialize the data-wrangling process of this module
    i.e. (putting all data in a couple of dataframes, after processing """
    initial_data.to_pickle('data_init.pkl')
    othr_data.to_pickle('data_other.pkl')
    total.to_pickle('data_total.pkl')
    return init_data, othr_data, total 



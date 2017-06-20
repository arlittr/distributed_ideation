#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:09:45 2017

@author: arlittr
"""

import pandas as pd

def setCorrect(k,cluster,this_df):
    if list(this_df[this_df['k']==k]['cluster']).count(cluster) >= 2:
        return True
    else:
        return False

def setQuestionList(k,this_df):
    return list(this_df[this_df['k']==k]['idea'])

if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/topic_clustering_results/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/'
    basename = 'res-n1400_dn'
#    basename = 'res-n1400_ge'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    
    df = pd.read_csv(path)
    
    #TODO: Weave in one(?) control question per HIT
    
    #make keyfile
    kf = pd.DataFrame(columns=df.columns)
    cluster_min_size = 5
    cluster_max_size = 50

    cluster_ids = set(df[(df['n'] >= cluster_min_size) & (df['n'] <= cluster_max_size)]['cluster'])
        
    k=0
    for cluster_id in cluster_ids:
        kftemp = df[df['cluster']==cluster_id].sample(n=2)
        kftemp = kftemp.append(df[(df['cluster']!=cluster_id) & (df['cluster'].isin(cluster_ids))].sample(n=1))
        kftemp['k'] = k
        kf = kf.append(kftemp)
        k+=1
    
    #Added improved keyfile later
#    outpath = outputbasepath + basename + '_mturk_keyfile' + '.csv'
#    kf.to_csv(outpath,encoding='utf-8')
       
    
    #make inputfile
    #Use headers to control how many questions per HIT
    headers = ['idea1','idea1a','idea1b','idea2','idea2a','idea2b','idea3','idea3a','idea3b']
    #Use max_HITs to control how many HITs to generate
    max_HITs = 2
    mtin = pd.DataFrame()
    for k in set(kf['k']):
        mtintemp = kf[kf['k']==k].iloc[[0]]
        #randomize the order of the other ideas
        mtintemp = mtintemp.append(kf[kf['k']==k].sample(n=2,weights=[0,1,1]))
        mtin = mtin.append(mtintemp)
    
    #add question order to each item for improved keyfile
    mtin['question_list'] = mtin.apply(lambda row: setQuestionList(row['k'],mtin),axis=1)
    #add correct bool to each item for improved keyfile
    mtin['correct'] = mtin.apply(lambda row: setCorrect(row['k'],row['cluster'],mtin),axis=1)
    
    #Improved keyfile with addition columns
    outpath = outputbasepath + basename + '_mturk_keyfile' + '.csv'
    mtin.to_csv(outpath,encoding='utf-8')
    
    idea_data = mtin['idea'].as_matrix()
    #hack to drop extra data if it won't fit in the given header matrix
    idea_data = idea_data[0:(int(len(idea_data) / len(headers))) * len(headers)]
    idea_data = idea_data.reshape((int(len(idea_data) / len(headers))),len(headers))
    idea_data = idea_data[0:max_HITs]
    
    mturk_infile = pd.DataFrame(data=idea_data,
                                columns=headers)
    
    outpath = outputbasepath + basename + '_mturk_infile' + '.csv'
    mturk_infile.to_csv(outpath,encoding='utf-8',index=False)
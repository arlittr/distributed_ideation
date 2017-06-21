#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:09:45 2017

@author: arlittr
"""

import pandas as pd
import random
import csv

def setCorrect(k,cluster,this_df):
    if list(this_df[this_df['k']==k]['cluster']).count(cluster) >= 2:
        return True
    else:
        return False

def setQuestionList(k,this_df):
    return list(this_df[this_df['k']==k]['idea'])

def getControlStatement(control_df):
    return control_df.sample(n=1)
    

if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/topic_clustering_results/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/'
    basename = 'res-n1400_dn'
#    basename = 'res-n1400_ge'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    
    df = pd.read_csv(path)
    
    #Weave in N control questions per HIT (prob max 4 total Qs per HIT)
    control_questions_per_HIT = 1
    
    #Set min desired number of unique questions in keyfile
    total_questions = 1000
    
    #%%
    #make keyfile
    kf = pd.DataFrame(columns=df.columns)
    
    #restrict cluster size
    cluster_min_size = 5
    cluster_max_size = 50
    cluster_ids = set(df[(df['n'] >= cluster_min_size) & (df['n'] <= cluster_max_size)]['cluster'])
    
    
    #stratified sample of selected clusters
    k=0
    while k < total_questions:
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
    headers = ['idea1','idea1a','idea1b','idea2','idea2a','idea2b','idea3','idea3a','idea3b','idea4','idea4a','idea4b']
    questions_per_HIT = int(len(headers) / 3)
    items_per_question = 3
    #Use max_HITs to control how many HITs to generate
    max_HITs = 100
    mtin = pd.DataFrame()
    for k in set(kf['k']):
        #grab the first idea as the baseline
        mtintemp = kf[kf['k']==k].iloc[[0]]
        #randomize the order of the other ideas
        mtintemp = mtintemp.append(kf[kf['k']==k].sample(n=items_per_question-1,weights=[0]+(items_per_question-1)*[1]))
        mtin = mtin.append(mtintemp)
    
    #add correctness bool to each item for improved keyfile
    mtin['correct'] = mtin.apply(lambda row: setCorrect(row['k'],row['cluster'],mtin),axis=1)

    #randomly shuffle around each question triad
    shuffled_inds = random.sample(range(int(max(mtin['k'])+1)),int(max(mtin['k'])+1))
    mtin['shuffled_ks'] = [item+i for item in shuffled_inds for i in [0.0,0.1,0.2]]
    mtin.sort_values('shuffled_ks',inplace=True)
    mtin['k_new'] = mtin['shuffled_ks'].apply(lambda x: int(x))

    #%%
    #weave in control text
    control_path = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/moby_dick.csv'
    with open(control_path,'r') as controlTextFile:
        reader = csv.reader(controlTextFile, dialect='excel')
        control_text = [r for r in reader][0]
    
    mtin['is_control'] = False
    mtin = mtin.reset_index()    
    seedvec = range(len(mtin[mtin['correct']==False]))[0::questions_per_HIT]
    replace_inds=[]
    false_answer_indices = list(mtin[mtin['correct']==False].index)
    for low,high in zip(seedvec,[s+(questions_per_HIT-1) for s in seedvec]):
        low = low + 1 #skip first question for weaving control question
        replace_ind = random.sample(false_answer_indices[low:high+1],control_questions_per_HIT)
        replace_inds.append(replace_ind)
        mtin.set_value(replace_ind,'idea',random.sample(control_text,1))
        mtin.set_value(replace_ind,'is_control',True)
    
    #%%
    
    #add question order to each item for improved keyfile
    mtin['question_list'] = mtin.apply(lambda row: setQuestionList(row['k'],mtin),axis=1)
    
    #%%
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
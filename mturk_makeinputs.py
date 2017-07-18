#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:09:45 2017

@author: arlittr
"""

import pandas as pd
import random
import csv
import numpy as np


def setCorrect(k,cluster,this_df):
    if list(this_df[this_df['k']==k]['cluster']).count(cluster) >= 2:
        return True
    else:
        return False

def setQuestionList(k,this_df):
    return list(this_df[this_df['k_new']==k]['idea'])

def getControlStatement(control_df):
    return control_df.sample(n=1)

def sampleClusters_stratified(this_df,this_kf,cluster_ids):
    #stratified random sample of selected clusters ------
    k=0
    cluster_size_dict = {}
    for cluster in list(set(this_df['cluster'])):
        cluster_size_dict[cluster] = this_df[this_df['cluster']==cluster]['n'].iloc[0]
        cluster_size_dict = {k:v for k,v in cluster_size_dict.items() if k in cluster_ids}
    while k < total_questions:
        cluster_ids = list(cluster_size_dict.keys())
        selection_probability = list(cluster_size_dict.values())
        normalized_selection_probability = [p/sum(selection_probability) for p in selection_probability]
        cluster_id = np.random.choice(list(cluster_size_dict.keys()),1,p=normalized_selection_probability)[0]
        kftemp = this_df[this_df['cluster']==cluster_id].sample(n=2)
        kftemp = kftemp.append(this_df[(this_df['cluster']!=cluster_id) & (this_df['cluster'].isin(cluster_ids))].sample(n=1))
        kftemp['k'] = k
        this_kf = this_kf.append(kftemp)
        k+=1
    return this_kf
    
def sanitizeInputs(this_df,col):
    this_df[col] = this_df[col].str.replace(r'\n\n', ' ')
    this_df[col] = this_df[col].str.replace(r'\r\r', ' ')
    #TODO: Delete weird(?) characters
    return this_df

def checkForUnprintable(this_df):
    #Prints all of the entries in your source file with unprintable chars
    import string
    print('Checking for unprintable characters...')
    printable = set(string.printable)
    this_df['printable'] = this_df['idea'].apply(lambda s: ''.join(filter(lambda x: x in printable, s)))
    print(this_df[this_df['printable']!=this_df['idea']])

if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/topic_clustering_results/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/'
    basename = 'res-n1400_dn'
#    basename = 'res-n1400_ge'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    
    df = pd.read_csv(path)
    #Sanitize inputs
    df = sanitizeInputs(df,'idea')
#    df['idea'] = df.idea.str.replace(r'\n\n', ' ')
#    df['idea'] = df.idea.str.replace(r'\r\r', ' ')
    
    #Weave in N control questions per HIT (prob max 4 total Qs per HIT)
    control_questions_per_HIT = 1
    
    #Set min desired number of unique questions in keyfile
    #Sampling will stop after sufficient passes
    total_questions = 1001
    
    #%%
    #make keyfile
    kf = pd.DataFrame(columns=df.columns)
    
    #restrict cluster size
    cluster_min_size = 5
    cluster_max_size = 50
    cluster_ids = set(df[(df['n'] >= cluster_min_size) & (df['n'] <= cluster_max_size)]['cluster'])
    
#    #uniform sample of selected clusters ------
#    k=0
#    while k < total_questions:
#        for cluster_id in cluster_ids:
#            kftemp = df[df['cluster']==cluster_id].sample(n=2)
#            kftemp = kftemp.append(df[(df['cluster']!=cluster_id) & (df['cluster'].isin(cluster_ids))].sample(n=1))
#            kftemp['k'] = k
#            kf = kf.append(kftemp)
#            k+=1 
    #-----
    
    
#    #stratified random sample of selected clusters ------
#    k=0
#    cluster_size_dict = {}
#    for cluster in list(set(df['cluster'])):
#        cluster_size_dict[cluster] = df[df['cluster']==cluster]['n'].iloc[0]
#        cluster_size_dict = {k:v for k,v in cluster_size_dict.items() if k in cluster_ids}
#    while k < total_questions:
#        cluster_ids = list(cluster_size_dict.keys())
#        selection_probability = list(cluster_size_dict.values())
#        normalized_selection_probability = [p/sum(selection_probability) for p in selection_probability]
#        cluster_id = np.random.choice(list(cluster_size_dict.keys()),1,p=normalized_selection_probability)[0]
#        kftemp = df[df['cluster']==cluster_id].sample(n=2)
#        kftemp = kftemp.append(df[(df['cluster']!=cluster_id) & (df['cluster'].isin(cluster_ids))].sample(n=1))
#        kftemp['k'] = k
#        kf = kf.append(kftemp)
#        k+=1 
#    #-----
    
    kf = sampleClusters_stratified(df,kf,cluster_ids)
    checkForUnprintable(kf)
    #make inputfile
    #Use headers to control how many questions per HIT
    headers = ['idea1','idea1a','idea1b','idea2','idea2a','idea2b','idea3','idea3a','idea3b']
    questions_per_HIT = int(len(headers) / 3)
    items_per_question = 3
    #Use max_questions to control how many unique HITs to generate
    #If max_questions is greater than the number of HITs supported by the keyfile, error
    #Right now this needs to be a little less than total_questions because of how control text is weaved in
    max_questions = 200
#    assert max_questions<=total_questions-2, "max_HITs must be less than total_questions generated in keyfile"
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
    replace_inds_control=[]
    false_answer_indices = list(mtin[mtin['correct']==False].index)
    for low,high in zip(seedvec,[s+(questions_per_HIT-1) for s in seedvec]):
        low = low + 1 #skip first question for weaving control question
        print(low,high)
        replace_ind = random.sample(false_answer_indices[low:high+1],control_questions_per_HIT)
        replace_inds_control.append(replace_ind)
        mtin.set_value(replace_ind,'idea',random.sample(control_text,1))
        mtin.set_value(replace_ind,'is_control',True)
    
    #%%
    #Choose one of the non-control questions, replace all 3 with separate treatment
    control_question_inds = [int(ind/items_per_question) for i in replace_inds_control for ind in i]
    #get indices for each element split into questions
    replaceable_inds = [[i for i in range(low,high+1) if i not in control_question_inds] for low,high in zip(seedvec,[s+(questions_per_HIT-1) for s in seedvec])]
    replace_inds_otherTreatment = [random.sample(i,1) for i in replaceable_inds]
    
    #Read in treatment dataframe
    spectral_path = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/spectral_clusters.csv'
    spectral_df = pd.read_csv(path)
    spectral_df = sanitizeInputs(spectral_df,'idea')
    
    #generate replacement questions
    #strategy: follow same process as generating main df questions
    #   generate full set of alternate treatment questions
    #   slot them in one at a time
    spectral_kf = pd.DataFrame(columns=df.columns)
    spectral_df['n'] = spectral_df.groupby('cluster')['cluster'].transform('count')    #restrict cluster size
    cluster_min_size = 5
    cluster_max_size = 50
    cluster_ids = set(spectral_df[(spectral_df['n'] >= cluster_min_size) & (spectral_df['n'] <= cluster_max_size)]['cluster'])
    spectral_kf = sampleClusters_stratified(spectral_df,spectral_kf,cluster_ids)
    spectral_kf['correct'] = spectral_kf.apply(lambda row: setCorrect(row['k'],row['cluster'],spectral_kf),axis=1)
    spectral_kf['is_spectral'] = True
    
    #for each subframe in main dataframe
    #strategy: match on k_new
    #slot replacement question into selected indices of main df
    #strategy: add new is_spectral column and overwrite df on matching indices
    mtin['is_spectral'] = False
    for k in replace_inds_otherTreatment:
        sub_mtin = mtin[mtin['k_new']==k]
        sub_mtin_spectral = spectral_kf[spectral_kf['k']==k]
        sub_mtin_spectral.index = sub_mtin.index
        mtin.update(sub_mtin_spectral)
        
    #cleanup
    #remap 'cluster'
    #clear 'keywords'
    mtin.loc[mtin['is_spectral']==True,'keywords'] = None
    
    
    #%%
    
    #add question order to each item for improved keyfile
    mtin['question_list'] = mtin.apply(lambda row: setQuestionList(row['k_new'],mtin),axis=1)
    
    #%%
    #Improved keyfile with additional columns
    outpath = outputbasepath + basename + '_mturk_keyfile' + '.csv'
    mtin.to_csv(outpath,encoding='utf-8')
    
    idea_data = mtin['idea'].values
    #hack to drop extra data if it won't fit in the given header matrix
    idea_data = idea_data[0:(int(len(idea_data) / len(headers))) * len(headers)]
    idea_data = idea_data.reshape((int(len(idea_data) / len(headers))),len(headers))
    idea_data = idea_data[0:max_questions]
    
    mturk_infile = pd.DataFrame(data=idea_data,
                                columns=headers)
    
    outpath = outputbasepath + basename + '_mturk_infile' + '.csv'
    mturk_infile.to_csv(outpath,encoding='utf-8',index=False)
    
    print(mturk_infile.iloc[0])
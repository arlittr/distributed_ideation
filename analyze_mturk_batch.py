#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:34:13 2017

@author: arlittr
"""

import pandas as pd
from ast import literal_eval

def setCorrect(k,cluster):
    if list(key_df[key_df['k']==k]['cluster']).count(cluster) >= 2:
        return True
    else:
        return False
    
def setQuestionList(k):
    return list(key_df[key_df['k']==k]['idea'])

def setBatchCorrect(answer,ideas):
    #For each question, reverse lookup correctness of answer in keyfile
    ideas = list(ideas)
    selected_response = ideas[answer]
    is_correct = key_df[(key_df['idea']==selected_response) & (key_df['question_list_str'] == str(ideas))]['correct'].iloc[0]
    return is_correct
    
    
if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk_batch_results/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/mturk_batch_analysis/'
    basename = 'Batch_DUMMY_batch_results'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    
    #get batch results
    batch_df = pd.read_csv(path)
    
    #get corresponding keyfile
    keyfile_basename = 'DUMMY res-n1400_dn_mturk_keyfile'
    keyfile_path = inputbasepath + keyfile_basename + fileextension
    key_df = pd.read_csv(keyfile_path)
    
    #mapping between answer field and selectable answers
    #in batch results, values for answers must be in sequential order (eg idea1a = 1, idea1b = 2) 
    answer_input_mapping = {'Answer.Q1Answer':['Input.idea1','Input.idea1a','Input.idea1b'],
                            'Answer.Q2Answer':['Input.idea2','Input.idea2a','Input.idea2b'],
                            'Answer.Q3Answer':['Input.idea3','Input.idea3a','Input.idea3b']
                            }
 
    #prepare question list as lists rather than strings
    key_df['question_list'] = key_df['question_list'].apply(lambda x: literal_eval(x))
    key_df['question_list_str'] =  key_df['question_list'].apply(lambda x: str(x))

    for k,v in answer_input_mapping.items():
        batch_df[k+'_correct'] = batch_df.apply(lambda row: setBatchCorrect(row[k],row[v]),axis=1)
        
     #if list(row[v]) == list(question_df['idea'])
        
#    for k,v in answer_input_mapping.items():
#        batch_df[k+'_correct'] = batch_df.apply(lambda row: setBatchCorrect(row['Answer.Q1Answer'],row[['Input.idea1','Input.idea1a','Input.idea1b']]),axis=1)
    

#    for k,v in answer_input_mapping.items():
#        baseline_idea = v[0]
#        key_df[key_df['idea']==key_df['idea'][0]]['cluster']
#        batch_df[k+'_expected'] = key_df['idea'].apply(lambda idea: True if key_df[key_df['idea']==idea]['cluster'] == key_df[key_df['idea']==baseline_idea]['cluster']   batch_df['Answer.Q1Answer']     row['cluster'] == key_df
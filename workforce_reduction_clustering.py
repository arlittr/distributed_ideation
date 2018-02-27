#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:42:31 2017

@author: arlittr
"""

import pandas as pd
import numpy as np
import csv

def sanitizeInputs(this_df,col):
    this_df[col] = this_df[col].str.replace(r'\n\n', ' ')
    this_df[col] = this_df[col].str.replace(r'\r\r', ' ')
    #TODO: Delete weird(?) characters
    return this_df

def splitMultipleIdeas(this_df,col,delim):
    this_df[col].split('\r\n')
    
    
def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    https://stackoverflow.com/questions/36272992/numpy-random-shuffle-by-row-independently
    """
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled

if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/dsta/ideas/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/dsta/infiles/'
    basename = 'Batch_2908479_batch_results_noGarbage'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    
    df = pd.read_csv(path)
    #Sanitize inputs
    df = sanitizeInputs(df,'Answer.ideas')
    
    #split multiple ideas
    df['ideas'] = df['Answer.ideas'].apply(lambda x: x.split('\r\n'))
    
    #merge ideas into single list
    ideas = [idea for idea_group in df['ideas'] for idea in idea_group]
    df['ideas'] = ideas
    
    #write out
    outpath = outputbasepath + basename + '_ideas' + '.csv'
    df['ideas'].to_csv(outpath,encoding='utf-8')
    
    #get control question text
    control_path = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/moby_dick.csv'
    with open(control_path,'r') as controlTextFile:
        reader = csv.reader(controlTextFile, dialect='excel')
        control_text = [r for r in reader][0]
        
    #sample
    total_questions_per_hit = 5
    control_questions_per_hit = 1
    experimental_questions_per_hit = total_questions_per_hit - control_questions_per_hit
    
    total_hits_goal = 200
    total_hits = 0
    
       
    ideas = np.array(np.random.permutation(df['ideas']))
    while np.shape(ideas)[0] / total_questions_per_hit < total_hits_goal:
        ideas = np.concatenate((ideas,
                                np.array(np.random.permutation(df['ideas']))),
                                axis=0)
    print(np.shape(ideas))
    
    while len(ideas) % total_questions_per_hit != 0:
        ideas = np.append(ideas,(np.random.choice(df['ideas'])))
        print(len(ideas))
    ideas = ideas.reshape(-1,5)
    
    #add control question
    control = np.random.choice(control_text,np.shape(ideas)[0])
    control = control.reshape(np.shape(control)[0],1)
    ideas2 = np.concatenate((ideas,control),axis=1)
    ideas2 = np.concatenate(
            (ideas2[:,0].reshape(np.shape(ideas2[:,0])[0],1), 
                            scramble(ideas2[:,1:])),
                            axis=1)
    df2 = pd.DataFrame(ideas2)
    
    
     #write out
    outpath = outputbasepath + basename + '_ideas_infile' + '.csv'
    df2.to_csv(outpath,encoding='utf-8')
        
        
        
        
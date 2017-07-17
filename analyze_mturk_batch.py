#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:34:13 2017

@author: arlittr
"""

import pandas as pd
from ast import literal_eval
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def setCorrect(k,cluster):
    if list(key_df[key_df['k']==k]['cluster']).count(cluster) >= 2:
        return True
    else:
        return False
    
#def setQuestionList(k):
#    return list(key_df[key_df['k']==k]['idea'])

def setBatchCorrect(answer,ideas):
    #For each question, reverse lookup correctness of answer in keyfile
    ideas = list(ideas)
    selected_response = ideas[answer]
#    print('ideas: ',ideas)
#    print('selected: ',selected_response)
#    print(key_df['idea']==selected_response)
#    print(key_df['question_list_str'] == str(ideas).replace('\\r','\\n'))
#    print(key_df['question_list_str'].iloc[0])
#    print(str(ideas).replace('\\r','\\n'))
    
    #uncomment these lines to debug issues related to malformed text
#    print('selected response:')
#    print(key_df[key_df['idea']==selected_response])
#    print('correct?:')
#    print(key_df[key_df['question_list_str'] == str(ideas).replace('\\r','\\n')]['correct'])
    is_correct = key_df[(key_df['idea']==selected_response) & (key_df['question_list_str'] == str(ideas).replace('\\r','\\n'))]['correct'].iloc[0]

    return is_correct

def setBatchIsControl(answer,ideas):
    #For each question, reverse lookup whether it was a control question in keyfile
    ideas = list(ideas)
    is_control = True in list(key_df[key_df['question_list_str'] == str(ideas)]['is_control'])
    return is_control

def setBatchIsNaive(answer,ideas):
    #For each question, reverse lookup if it was the naive algorithm
    ideas = list(ideas)
    #TODO: generalize all 'is_spectral' to 'is_naive' need to mod other file too
    is_naive = True in list(key_df[key_df['question_list_str'] == str(ideas)]['is_spectral'])
    return is_naive

def fitBinomialDist(df):
    N = len(list(df['is_correct']))
    p = list(df['is_correct']).count(True) / N
    
    return N,p

def getSuccessFailBinomialDist(df):
    successes = list(df['is_correct']).count(True)
    failures = list(df['is_correct']).count(False)
    
    return successes,failures
    
if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk_batch_results/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/mturk_batch_analysis/'
#    basename = 'Batch_DUMMY_batch_results'
#    basename = 'Batch_2844899_batch_results'
    basename = 'Batch_2870121_batch_results'

    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    
    #get batch results
    batch_df = pd.read_csv(path,encoding='utf-8')
#    batch_df.replace({'\\r': '\\n'}, regex=True, inplace=True)
    
    #get corresponding keyfile
#    keyfile_basename = 'DUMMY res-n1400_dn_mturk_keyfile'
#    keyfile_basename = 'Pilot 1 7 res-n1400_dn_mturk_keyfile'
    keyfile_basename = 'Pilot_2_fixed_questionlist_res-n1400_dn_mturk_keyfile'
    keyfile_path = inputbasepath + keyfile_basename + fileextension
    key_df = pd.read_csv(keyfile_path,encoding='utf-8')
#    key_df.replace({'\r': '','\n':'','\\r': '','\\n':''}, regex=True, inplace=True)
    
    #mapping between answer field and selectable answers
    #in batch results, values for answers must be in sequential order (eg idea1a = 1, idea1b = 2) 
    answer_input_mapping = {'Answer.Q1Answer':['Input.idea1','Input.idea1a','Input.idea1b'],
                            'Answer.Q2Answer':['Input.idea2','Input.idea2a','Input.idea2b'],
                            'Answer.Q3Answer':['Input.idea3','Input.idea3a','Input.idea3b']
                            }
 
    #prepare question list as lists rather than strings, might be redundant
    key_df['question_list_literal'] = key_df['question_list'].apply(lambda x: literal_eval(x))
    key_df['question_list_str'] =  key_df['question_list_literal'].apply(lambda x: str(x))

    #set correct
    for k,v in answer_input_mapping.items():
        batch_df[k+'_correct'] = batch_df.apply(lambda row: setBatchCorrect(row[k],row[v]),axis=1)
    
    #set experimental vs control
    for k,v in answer_input_mapping.items():
        batch_df[k+'_is_control'] = batch_df.apply(lambda row: setBatchIsControl(row[k],row[v]),axis=1)
    
    #set naive algorithm
    for k,v in answer_input_mapping.items():
        batch_df[k+'_is_naive_algorithm'] = batch_df.apply(lambda row: setBatchIsNaive(row[k],row[v]),axis=1)
    
    #split experimental and control into separate dataframes
    stacked_df = pd.DataFrame()
    question_ids = ['1','2','3']
    for question_id in question_ids:
        temp_df = batch_df.filter(like=question_id) #get questions containing question_id
        temp_df.rename(columns = lambda x: x.replace(question_id,''),inplace=True) #generalize by removing question_id (lets us stack homogenous columns)
        stacked_df = stacked_df.append(temp_df)
     
    stacked_df.columns=['idea','idea_a','idea_b','answer','is_correct','is_control','is_naive']
    control_df = stacked_df[stacked_df['is_control']==True]
    experimental_df = stacked_df[stacked_df['is_control']==False]
   
#    #fit bionomial dist to control
#    N_control,p_control = fitBinomialDist(control_df)
#    N_experimental,p_experimental = fitBinomialDist(experimental_df)
    
    #fisher's exact test
    contingency = [getSuccessFailBinomialDist(control_df),getSuccessFailBinomialDist(experimental_df)]
    odds_ratio_fisher,p_fisher = scipy.stats.fisher_exact(contingency)
    
    #bootstrapping experimental vs control
    experimental_freqs = []
    control_freqs = []
    random_freqs = []
    odds_ratios=np.array([])
    ps=np.array([])
    nsamples=100
    for n in range(nsamples):
        sampled_control = list(control_df['is_correct'].sample(frac=1,replace=True))
        sampled_experimental = list(experimental_df['is_correct'].sample(frac=1,replace=True))    
        sampled_random = list(np.random.choice([True,False],size=len(list(experimental_df['is_correct']))))
        sampled_contingency = [[sampled_experimental.count(True),sampled_control.count(True)],
                                [sampled_experimental.count(False),sampled_control.count(False)]]
        odds_ratio,p = scipy.stats.fisher_exact(sampled_contingency)
        odds_ratios = np.append(odds_ratios,odds_ratio)
        ps = np.append(ps,p)
        experimental_freqs = np.append(experimental_freqs,sampled_experimental.count(True))
        control_freqs = np.append(control_freqs,sampled_control.count(True))
        random_freqs = np.append(random_freqs,sampled_random.count(True))
    odds_ratios = odds_ratios[odds_ratios<1e10] #hack for testing with small sample sizes, remove infs
    print('Experimental vs Control Bootstrapping with ',str(nsamples),'samples')
    print('Is experimental distinguishable from control? (Where control is best possible outcome)')
    print(odds_ratios.mean(),odds_ratios.std())
    print(ps.mean(),ps.std(),ps.max())
    odds_ratio_CI = (odds_ratios.mean()-1.96*odds_ratios.std(),odds_ratios.mean()+1.96*odds_ratios.std())
    print('odds ratio 95% CI: (',odds_ratios.mean()-1.96*odds_ratios.std(),'--',odds_ratios.mean()+1.96*odds_ratios.std(),')')
    plt.hist(odds_ratios,bins='auto')
    plt.title('Odds Ratio Frequency Distribution - Experimental vs Control')
    plt.xlabel('Odds Ratio')
    plt.ylabel('Frequency')
    line = plt.axvline(x=odds_ratio_CI[0])
    line.set_linestyle('--')
    line = plt.axvline(x=odds_ratio_CI[1])
    line.set_linestyle('--')
    plt.grid()
    plt.show()
    
    print('1')
    plt.hist(ps,range=(0,0.2))#,bins='auto')#,range=(0,0.2))
    plt.title('p-value Frequency Distribution')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    print('2')
    plt.hist(experimental_freqs,range=(0,int(experimental_freqs.max())),bins=int(experimental_freqs.max()+1))
    plt.title('Bootstrapped Experimental Binomial Probability Density')
    plt.xlabel('Number Successes')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    print('3')
    plt.hist(control_freqs,range=(0,int(control_freqs.max())),bins=int(control_freqs.max()+1))
    plt.title('Bootstrapped Control Binomial Probability Density')
    plt.xlabel('Number Successes')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    normalized_experimental = [e/len(list(experimental_df['is_correct'])) for e in experimental_freqs]
    plt.hist(normalized_experimental,bins=sorted(list(set(normalized_experimental))),normed=True) 
    plt.xlim(0,1)
    plt.title('Bootstrapped Experimental Binomial Probability Density (Normalized)')
    plt.xlabel('Number Successes')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    normalized_control = [c/len(list(control_df['is_correct'])) for c in control_freqs]
    plt.hist(normalized_control,bins=sorted(list(set(normalized_control))),normed=True) 
    plt.xlim(0,1)
    plt.title('Bootstrapped Control Binomial Probability Density (Normalized)')
    plt.xlabel('Number Successes')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    normalized_random = [r/len(sampled_random) for r in random_freqs]
    plt.hist(normalized_random,bins=sorted(list(set(normalized_random))),normed=True)
    plt.xlim(0,1)
    plt.title('Bootstrapped Random Binomial Probability Density (Normalized)')
    plt.xlabel('Number Successes')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    plt.title('Bootstrapped Probability Density (Normalized)')
    plt.xlabel('Number Successes (Normalized)')
    plt.ylabel('Frequency (Normalized)')
    _,_,hatch1 = plt.hist(normalized_experimental,bins=sorted(list(set(normalized_experimental))),normed=True,alpha=0.5) 
    _,_,hatch2 = plt.hist(normalized_control,bins=sorted(list(set(normalized_control))),normed=True,alpha=0.5) 
    _,_,hatch3 = plt.hist(normalized_random,bins=sorted(list(set(normalized_random))),normed=True,alpha=0.5)
    plt.legend(['Experimental','Control','Random'])
    plt.xlim(0,1.3)
    for h in hatch1:
        h.set_hatch('x')
    for h in hatch2:
        h.set_hatch('|')
    for h in hatch3:
        h.set_hatch('-')
    plt.show()
    
#    noise = np.random.normal(0, 1, (1000, ))
#    density = scipy.stats.gaussian_kde(noise)
#    plt.plot(binse,density(binse))
#    plt.plot(binsc,density(binsc))
#    plt.plot(binsr,density(binsr))
#    plt.grid()
#    plt.show()    
    
    print('============================')
    
    #bootstrapping experimental vs random
    odds_ratios=np.array([])
    ps=np.array([])
    nsamples=100
    for n in range(nsamples):
        sampled_control = list(control_df['is_correct'].sample(frac=1,replace=True))
        sampled_experimental = list(experimental_df['is_correct'].sample(frac=1,replace=True))
        sampled_contingency = [[sampled_experimental.count(True),int(len(sampled_experimental)/2)],
                                [sampled_experimental.count(False),int(len(sampled_experimental)/2)]]
        odds_ratio,p = scipy.stats.fisher_exact(sampled_contingency)
        odds_ratios = np.append(odds_ratios,odds_ratio)
        ps = np.append(ps,p)
    odds_ratios = odds_ratios[odds_ratios<1e10] #hack for testing with small sample sizes, remove infs
    print('Experimental vs 50/50 Random Bootstrapping with',str(nsamples),'samples')
    print('Does experimental outperform chance?')
    print(odds_ratios.mean(),odds_ratios.std())
    print(ps.mean(),ps.std(),ps.max())
    odds_ratio_CI = (odds_ratios.mean()-1.96*odds_ratios.std(),odds_ratios.mean()+1.96*odds_ratios.std())
    print('odds ratio 95% CI: (',odds_ratios.mean()-1.96*odds_ratios.std(),'--',odds_ratios.mean()+1.96*odds_ratios.std(),')')
    plt.hist(odds_ratios,bins='auto')
    plt.title('Odds Ratio Frequency Distribution - Experimental vs 50/50 Random')
    plt.xlabel('Odds Ratio')
    plt.ylabel('Frequency')
    line = plt.axvline(x=odds_ratio_CI[0])
    line.set_linestyle('--')
    line = plt.axvline(x=odds_ratio_CI[1])
    line.set_linestyle('--')
    plt.grid()
    plt.show()
    plt.hist(ps,range=(0,0.2))#,bins='auto')#,range=(0,0.2))
    plt.title('p-value Frequency Distribution')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    print('============================')
    
    #bootstrapping control vs random
    odds_ratios=np.array([])
    ps=np.array([])
    nsamples=100
    for n in range(nsamples):
        sampled_control = list(control_df['is_correct'].sample(frac=1,replace=True))
        sampled_experimental = list(experimental_df['is_correct'].sample(frac=1,replace=True))
        sampled_contingency = [[sampled_control.count(True),int(len(sampled_experimental)/2)],
                                [sampled_control.count(False),int(len(sampled_experimental)/2)]]
        odds_ratio,p = scipy.stats.fisher_exact(sampled_contingency)
        odds_ratios = np.append(odds_ratios,odds_ratio)
        ps = np.append(ps,p)
    odds_ratios = odds_ratios[odds_ratios<1e10] #hack for testing with small sample sizes, remove infs
    print('Control vs 50/50 Random Bootstrapping with',str(nsamples),'samples')
    print('Does control outperform chance? (if the two are not distinguishable, we are not getting good data from mturk)')
    print(odds_ratios.mean(),odds_ratios.std())
    print(ps.mean(),ps.std(),ps.max())
    odds_ratio_CI = (odds_ratios.mean()-1.96*odds_ratios.std(),odds_ratios.mean()+1.96*odds_ratios.std())
    print('odds ratio 95% CI: (',odds_ratios.mean()-1.96*odds_ratios.std(),'--',odds_ratios.mean()+1.96*odds_ratios.std(),')')
    plt.hist(odds_ratios,bins='auto')
    plt.title('Odds Ratio Frequency Distribution - Control vs 50/50 Random')
    plt.xlabel('Odds Ratio')
    plt.ylabel('Frequency')
    line = plt.axvline(x=odds_ratio_CI[0])
    line.set_linestyle('--')
    line = plt.axvline(x=odds_ratio_CI[1])
    line.set_linestyle('--')
    plt.grid()
    plt.show()
    plt.hist(ps,range=(0,0.2))#,bins='auto')#,range=(0,0.2))
    plt.title('p-value Frequency Distribution')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    print('============================')
    
    

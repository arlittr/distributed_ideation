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

def getQuestionSuccessRates(this_df,prompt_columns):
    #returns a list of tuples of question success rates 
    #(one tuple per unique question formulation)
    #only useful if we're replicating the same question
    #this_df: dataframe containing single treatment
    #prompt_columns: list containing column headers of all statements that make up a question
    return list(this_df.groupby(prompt_columns)['is_correct'].aggregate(lambda x: tuple(x)))

def bootstrapMultimodalQuestionDistribution(success_rates,nsamples,normalize=True):
    #success rates: list of tuples where each tuple contains True and False responses to a replicated question
    # each tuple represents a different question
    #nsamples: number of times to resample with replacement
    sampled_freqs = []
    for n in range(nsamples):
        for this_question_rates in success_rates:
            #sample, make a single flattened list of resampled success rates for each question
            this_sample = list(np.random.choice(this_question_rates,size=len(this_question_rates)))
            if normalize:
                sampled_freqs = np.append(sampled_freqs,np.mean(this_sample))
            else:
                sampled_freqs = np.append(sampled_freqs,this_sample.count(True))
    
    return sampled_freqs
    
if __name__ == '__main__':
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk_batch_results/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/mturk_batch_analysis/'
#    basename = 'Batch_DUMMY_batch_results'
#    basename = 'Batch_2844899_batch_results'
#    basename = 'Batch_2870121_batch_results'
#    basename = 'Batch_2872697_batch_results'
    
    fileextension = '.csv'
   
    #main dataset
#    basenames = ['Batch_2870121_batch_results',
#                 'Batch_2872697_batch_results'
#                 ]
    
    #10 reps trial
    basenames = ['Run4 10replicates Batch_2874207_batch_results','Batch_2877587_batch_results'
                 ]
    
    #get batch results
    batch_df = pd.DataFrame()
    for basename in basenames:
        path = inputbasepath + basename + fileextension
        batch_df = batch_df.append(pd.read_csv(path,encoding='utf-8'),ignore_index=True)
    
    #get corresponding keyfile
#    keyfile_basename = 'DUMMY res-n1400_dn_mturk_keyfile'
#    keyfile_basename = 'Pilot 1 7 res-n1400_dn_mturk_keyfile'
#    keyfile_basename = 'Pilot_2_fixed_questionlist_res-n1400_dn_mturk_keyfile'
#    keyfile_basename = 'Run_3_fixed_questionlist_res-n1400_dn_mturk_keyfile'
    
    #main dataset
#    keyfile_basenames = ['Pilot_2_fixed_questionlist_res-n1400_dn_mturk_keyfile',
#                         'Run_3_fixed_questionlist_res-n1400_dn_mturk_keyfile']
    
    #10 reps trial
    keyfile_basenames = ['Run4 10replicates res-n1400_dn_mturk_keyfile','R5 res-n1400_dn_mturk_keyfile']
                          
    #get keyfiles
    key_df = pd.DataFrame()
    for keyfile_basename in keyfile_basenames:
        keyfile_path = inputbasepath + keyfile_basename + fileextension
        key_df = key_df.append(pd.read_csv(keyfile_path,encoding='utf-8'),ignore_index=True)
            
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
        temp_df = batch_df.filter(regex='(Input|Answer).*'+question_id) #get questions containing question_id
        temp_df.rename(columns = lambda x: x.replace(question_id,''),inplace=True) #generalize by removing question_id (lets us stack homogenous columns)
        stacked_df = stacked_df.append(temp_df)
     
    stacked_df.columns=['idea','idea_a','idea_b','answer','is_correct','is_control','is_naive']
    control_df = stacked_df[stacked_df['is_control']==True]
    naive_df = stacked_df[stacked_df['is_naive']==True]
    experimental_df = stacked_df[(stacked_df['is_control']==False) & (stacked_df['is_naive']==False)]
    
    #%% Analysis of replicates
    #success rates for each unique question in each treatment
    control_success_rates = getQuestionSuccessRates(control_df,['idea','idea_a','idea_b'])
    naive_success_rates = getQuestionSuccessRates(naive_df,['idea','idea_a','idea_b'])
    experimental_success_rates = getQuestionSuccessRates(experimental_df,['idea','idea_a','idea_b'])
    
    #bootstrap
    nsamples=1000
    bootstrapped_control_rates = bootstrapMultimodalQuestionDistribution(control_success_rates,nsamples)
    bootstrapped_naive_rates = bootstrapMultimodalQuestionDistribution(naive_success_rates,nsamples)
    bootstrapped_experimental_rates = bootstrapMultimodalQuestionDistribution(experimental_success_rates,nsamples)

    #plot
    plt.title('Bootstrapped Probability Density (Normalized)')
    plt.xlabel('Number Successes (Normalized)')
    plt.ylabel('Frequency (Normalized)')
    _,_,hatch1 = plt.hist(bootstrapped_control_rates,bins=20+1,normed=True,alpha=0.5) 
    _,_,hatch2 = plt.hist(bootstrapped_naive_rates,bins=20+1,normed=True,alpha=0.5) 
    _,_,hatch3 = plt.hist(bootstrapped_experimental_rates,bins=20+1,normed=True,alpha=0.5)

    plt.legend(['Control','Naive','Experimental',])
    plt.xlim(0,1.0)
    for h in hatch1:
        h.set_hatch('\\')
    for h in hatch2:
        h.set_hatch('|')
    for h in hatch3:
        h.set_hatch('-')
    plt.show()
    
    plt.hist(bootstrapped_control_rates,range=(0,int(bootstrapped_control_rates.max())),normed=True,bins=10+1)
    plt.title('Control Binomial Probability Density\n Hierarchical Bootstrap w/ replicated questions')
    plt.xlabel('Number Successes (normalized)')
    plt.ylabel('Frequency (normalized)')
    plt.grid()
    plt.show()
    
    plt.hist(bootstrapped_naive_rates,range=(0,int(bootstrapped_naive_rates.max())),normed=True,bins=10+1)
    plt.title('Naive Binomial Probability Density\n Hierarchical Bootstrap w/ replicated questions')
    plt.xlabel('Number Successes (normalized)')
    plt.ylabel('Frequency (normalized)')
    plt.grid()
    plt.show()
    
    plt.hist(bootstrapped_experimental_rates,range=(0,int(bootstrapped_experimental_rates.max())),normed=True,bins=10+1)
    plt.title('Experimental Binomial Probability Density\n Hierarchical Bootstrap w/ replicated questions')
    plt.xlabel('Number Successes (normalized)')
    plt.ylabel('Frequency (normalized)')
    plt.grid()
    plt.show()
    
    #%%Analysis of full clusterings
    
    #fisher's exact test
    contingency = [getSuccessFailBinomialDist(control_df),getSuccessFailBinomialDist(experimental_df)]
    odds_ratio_fisher,p_fisher = scipy.stats.fisher_exact(contingency)
    
    #bootstrapping experimental vs control
    experimental_freqs = []
    control_freqs = []
    random_freqs = []
    naive_freqs = []
    odds_ratios=np.array([])
    ps=np.array([])
    nsamples=5000
    for n in range(nsamples):
        sampled_control = list(control_df['is_correct'].sample(frac=1,replace=True))
        sampled_experimental = list(experimental_df['is_correct'].sample(frac=1,replace=True))    
        sampled_random = list(np.random.choice([True,False],size=len(list(experimental_df['is_correct']))))
        sampled_naive = list(naive_df['is_correct'].sample(frac=1,replace=True))  
        sampled_contingency = [[sampled_experimental.count(True),sampled_control.count(True)],
                                [sampled_experimental.count(False),sampled_control.count(False)]]
        odds_ratio,p = scipy.stats.fisher_exact(sampled_contingency)
        odds_ratios = np.append(odds_ratios,odds_ratio)
        ps = np.append(ps,p)
        experimental_freqs = np.append(experimental_freqs,sampled_experimental.count(True))
        control_freqs = np.append(control_freqs,sampled_control.count(True))
        random_freqs = np.append(random_freqs,sampled_random.count(True))
        naive_freqs = np.append(naive_freqs,sampled_naive.count(True))
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
    
    normalized_naive = [r/len(sampled_naive) for r in naive_freqs]
    plt.hist(normalized_naive,bins=sorted(list(set(normalized_naive))),normed=True)
    plt.xlim(0,1)
    plt.title('Bootstrapped Naive Binomial Probability Density (Normalized)')
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
    _,_,hatch4 = plt.hist(normalized_naive,bins=sorted(list(set(normalized_naive))),normed=True,alpha=0.5)

    plt.legend(['Experimental','Control','Random','Naive'])
    plt.xlim(0,1.0)
    for h in hatch1:
        h.set_hatch('\\')
    for h in hatch2:
        h.set_hatch('|')
    for h in hatch3:
        h.set_hatch('-')
    for h in hatch4:
        h.set_hatch('/')
    plt.show()
      
    
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
    
    

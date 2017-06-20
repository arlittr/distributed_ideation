#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:56:31 2017

@author: arlittr
"""

import nltk
import csv

def join_punctuation(seq, characters="\\'.,;?!"):
    #https://stackoverflow.com/questions/15950672/join-split-words-and-punctuation-with-punctuation-in-the-right-place
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current

if __name__ == "__main__":
    dick = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')
    dicks = [' '.join(join_punctuation(d)) for d in dick]
    dicks = [d.replace("' ","'") for d in dicks]
    dicks = [d for d in dicks if len(d) > 100]
    
    
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/mturk/'
    basename = 'moby_dick'
    fileextension = '.csv'
    path = outputbasepath + basename + fileextension
    with open(path,'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows([dicks[1000:1100]])
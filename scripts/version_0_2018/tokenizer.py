# -*- coding: utf-8 -*-
# --------------------------------------------------------------------
# DSA: Run Tokenizer  								 			 -
#                                                                    -
# Author  : Amir HAZEM                                               -     
# Created : 14/12/2017                                               -  
# Updated : 14/12/2017                                               -
#                                                                    -  
# --------------------------------------------------------------------



from __future__ import division
import argparse
import numpy as np
import sys
import unicodecsv
import math
import os
import io
import re
import encodings
import nltk
from nltk.corpus import wordnet as wn
from operator import add



import os.path 
  
path_in="/home/hazem-a/Bureau/LS2N_2018/data/wikipedia_test/raw/"

path_out="/home/hazem-a/Bureau/LS2N_2018/data/wikipedia_test/tok/"

def tokenize(path_in,path_out):  
    fichier=[]  
    sep=" "	
    id_=0
    for root, dirs, files in os.walk(path_in):  
        for i in files:  
            #fichier.append(os.path.join(root, i))  
            id_+=1
            print root
            print i
            
            file_in=root+"/"+i

            ch=root.split("/")

            file_out=path_out+i+"."+ch[len(ch)-1]+".txt"
            print file_in
            print file_out
            
            with  open(file_out,'w') as f1:

				f = open(file_in,'r')
				
				for line in f:
					print line

					ch_tok=sep.join(nltk.word_tokenize((line.rstrip().lower()).decode('utf-8')))
					print ch_tok	
					ch_tok=sep.join(nltk.word_tokenize(ch_tok))
					print ch_tok	

            sys.exit()
    






#----------------------------------------- 
# Main 
#-----------------------------------------

if __name__=='__main__':


	tokenize(path_in,path_out)
    
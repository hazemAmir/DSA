# -*- coding: utf-8 -*-
#________________________________________ *** DSA *** _____________________________________________-
#																								   -
# 							  Random subcorpus selection										   -
# 																							       - 
# --------------------------  	N% of data is selected  --------------------------------------------
# 									                                                               -
# Author  : Amir HAZEM  			                                                               -
# Created : 24/01/2019				                       		                                   -
# Updated :	24/01/2018						                                                       -
# source  : 									 												   -
# 																							       - 
#									                                                               -
#---------------------------------------------------------------------------------------------------
#!/usr/bin/env python


# Libraries ----------------------------------------------------------------------------------------
from __future__ import division
import sys
import os 
from os import listdir
from os.path import isfile, join
import numpy as np
import random as rn
import nltk
import math
import treetaggerwrapper
from operator import itemgetter, attrgetter
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import jaccard_similarity_score
from scipy import spatial

# Parameters:---------------------------------------------------------------------------------------
corpus 		= sys.argv[1]
lang 		= sys.argv[2]	# Language : en/fr/...	
corpus_type = sys.argv[3]	# Flag     : tok/lem/postag
percent 	= int(sys.argv[4])

corpus_dir  	= "../data/train/corpora/" + corpus  + '/' + corpus_type + '/' + lang

path_out	    = "../data/train/corpora/" + corpus + str(percent)  + '/' + corpus_type + '/' + lang + "/train_"+ corpus + str(percent) +"_"+lang+".txt"
# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--


def load_corpus(corpus_dir):

	corpus = {}
	onlyfiles = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]
	
	for filename in onlyfiles:
		print filename


		f = open(corpus_dir+"/"+filename,'r')
		i = 0
		for line in f:
			line = ((line.decode('utf-8'))).strip()

			#print line
			
			corpus [i] = line 
			i+=1
			

	return corpus


def write_corpus(path_out,corpus_all,corpus_selected):

	tab_res = []

	with  open(path_out,'w') as f1 :
		
		for ind in corpus_selected:
		
			f1.write (corpus_all[ind].encode('utf-8') + '\n')

	

#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
	
	corp = {}
	

	corp = load_corpus(corpus_dir)	

	print len(corp)

	limit = round(((len(corp)  * percent) /100))

	print limit	

	cpt = 0
	tab = {}
	while cpt < limit:

		ind = np.random.randint(0,limit)

		if tab.has_key(ind):
			print "error" + " "+ str(ind)	
		else:	
			tab[ind] = ind
			cpt += 1	

	print len(tab)	

	print limit	


	write_corpus(path_out,corp,tab)

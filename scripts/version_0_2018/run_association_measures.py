# -*- coding: utf-8 -*-
#________________________________________ *** DSA *** _____________________________________________-
#																								   -
# 							  Distributional Standard Approach									   -
# 																							       - 
# * Standard Approach: Step 3 -----> Association Measures ------------------------------------------
# 									                                                               -
# Author  : Amir HAZEM  			                                                               -
# Created : 19/11/2018				                       		                                   -
# Updated :	19/11/2018						                                                       -
# source  : 									 												   -
# 		   python run_association_measures.py breast_cancer en mi 3 5						       - 
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
#---------------------------------------------------------------------------------------------------

# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--


# Parameters:---------------------------------------------------------------------------------------
corpus 				= sys.argv[1]	   # Corpus	
lang 				= sys.argv[2]	   # Language : en/fr/...	
assoc       		= sys.argv[3]	   # MI / JAC / ODDS 
w 					= int(sys.argv[4]) # : window size 1/2/3... number of words before and after the center word
min_occ				= int(sys.argv[5]) # : filtering tokens with number of occurence less than min_occ

path_vocab      	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+lang + ".csv"
path_ctxvec			= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang + '_w' + str(w) + ".vect"

path_ctxvec_assoc	= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc"
path_termlist_csv	= "../data/train/termlists/en_fr_" + corpus +"_248.csv"
# ----------------------------------------------------------------------------------------------------


def load_context_vectors(path_ctxvec):

	context_vectors = {}
	f = open(path_ctxvec,'r')
	for line in f:
		line = ((line.decode('utf-8'))).strip()
		vect = line.split(':')
		context_vectors[(vect[0].split('#'))[0]]=line
	return context_vectors	

def load_occurrence_vectors(path_occvec):

	occ = {}
	f = open(path_occvec,'r')
	for line in f:
		
		line = ((line.decode('utf-8'))).strip()
		
		vect = line.split('\t')
		
		if len(vect)>1: # avoid '' # to be solved beforehand
			occ [vect[0]]= int(vect[1])
	return occ


def load_termlist(path_termlist):

	termlist = {}
	termlist_inv = {}
	f = open(path_termlist,'r')
	for line in f:
		line = ((line.decode('utf-8'))).strip()
		vect = line.split('\t')
		termlist[vect[0]] = vect[1]
		termlist_inv[vect[1]] = vect[0]
	print ('%s \tTermlist size' % len(termlist))		
	return termlist,termlist_inv

def write_context_vectors(path_ctxvec, context_vectors_assoc, occ, min_occ,path_termlist_csv):

	
	termlist,termlist_inv = load_termlist(path_termlist_csv)
		
	with  open(path_ctxvec,'w') as f1 :		

		for x in context_vectors_assoc:
			count = occ[x] 


			if count >= min_occ : 
				f1.write(context_vectors_assoc[x].encode('utf-8') + '\n')
			else:

				if termlist.has_key(x)  and lang == "en":
					# keep the reference list
					line = context_vectors_assoc[x].split(':')
					token0 = line[0].split('#')
					result = str(token0[0]) + '#' + str(min_occ)+ ':'.join(line[1:]) 
					f1.write(result.encode('utf-8') + '\n')
					

				else:
					if termlist_inv.has_key(x) and 	lang == "fr":
						line = context_vectors_assoc[x].split(':')
						token0 = line[0].split('#')
						result = str(token0[0]) + '#' + str(min_occ)+ ':'.join(line[1:]) 
						f1.write(result.encode('utf-8') + '\n')

	#sys.exit()				

def compute_contingency_table(coocc):

	Tab_occ_X ={}
	Tab_cooc_XY ={}
	Tab_cooc_X_ALL ={}
	Tab_cooc_ALL_Y ={}
	Total = 0

	for word in coocc:

		line = coocc[word].split(':')

		x_word = word  
		x_freq = (line[0].split('#'))[1]
		Tab_occ_X[x_word] = int(x_freq)

		for i in range(1,len(line)):


			y_word = (line[i].split('#'))[0]  
			y_freq = int((line[i].split('#'))[1])

			Tab_cooc_XY[x_word+" "+y_word] = y_freq
			
			if Tab_cooc_X_ALL.has_key(x_word): 
				Tab_cooc_X_ALL[x_word]+=y_freq
			else:
				Tab_cooc_X_ALL[x_word]=y_freq	

			if Tab_cooc_ALL_Y.has_key(y_word): 
				Tab_cooc_ALL_Y[y_word]+=y_freq
			else:
				Tab_cooc_ALL_Y[y_word]=y_freq		

		Total += Tab_cooc_X_ALL[x_word]		

	return 	Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total
			 	

def compute_MI(coocc,Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total):

	Tab_MI = {}
	for x_word in coocc:
		
		line = coocc[x_word].split(':') 

		vec = line[0] 
		for i in range(1,len(line)):

			y_word = (line[i].split('#'))[0]  
			y_freq = int((line[i].split('#'))[1])		

			a = Tab_cooc_XY[x_word+" "+y_word]
			b = Tab_cooc_X_ALL[x_word]
			c = Tab_cooc_ALL_Y[y_word]

			result = Total * a 

			result = math.log(result / (b * c))
			#print x_word+ " "+ y_word + " "+ str(result)
			vec = vec + ':' + y_word + "#"+ str(result) 

		Tab_MI [x_word] = vec

	return Tab_MI		

def compute_ODDS(coocc,Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total):

	Tab_ODDS = {}
	for x_word in coocc:
		
		line = coocc[x_word].split(':') 

		vec = line[0] 
		for i in range(1,len(line)):

			y_word = (line[i].split('#'))[0]  
			y_freq = int((line[i].split('#'))[1])		

			a = Tab_cooc_XY[x_word+" "+y_word]
			b = Tab_cooc_X_ALL[x_word] - a
			c = Tab_cooc_ALL_Y[y_word] - a 
			N = Total
			d = N - a - b -c

			result = math.log( ((a + 0.5) * (d + 0.5))  / ((b + 0.5) * (c + 0.5)))
			#print x_word+ " "+ y_word + " "+ str(result)
			vec = vec + ':' + y_word + "#"+ str(result) 

		Tab_ODDS [x_word] = vec

	return Tab_ODDS


def compute_LL(coocc,Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total):

	Tab_LL = {}
	for x_word in coocc:
		
		line = coocc[x_word].split(':') 

		vec = line[0] 
		for i in range(1,len(line)):

			y_word = (line[i].split('#'))[0]  
			y_freq = int((line[i].split('#'))[1])		

			a = Tab_cooc_XY[x_word+" "+y_word]
			b = Tab_cooc_X_ALL[x_word] - a
			c = Tab_cooc_ALL_Y[y_word] - a 
			N = Total
			d = N - a - b -c

			if a>0 :
				result  = a * math.log(a)
			if b>0 :
				result += b * math.log(b) 
			if c>0:
				result += c * math.log(c) 
			if d>0 :	
				result += d * math.log(d) 
			if N>0:	
				result += N * math.log(N)
			
			result +=  - (a+b) * math.log(a+b) - (a+c) * math.log(a+c) - (b+d) * math.log(b+d) - (c+d) * math.log(c+d)

			
			#print x_word+ " "+ y_word + " "+ str(result)
			vec = vec + ':' + y_word + "#"+ str(result) 

		Tab_LL [x_word] = vec

	return Tab_LL
			
#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':

	context_vectors = {}
	Tab_occ_X ={}
	Tab_cooc_XY ={}
	Tab_cooc_X_ALL ={}
	Tab_cooc_ALL_Y ={}
	Total = 0

	context_vectors_assoc = {} 
	occ = {}

	# Load occurrence vectors
	occ = load_occurrence_vectors(path_vocab)

	# Loasd cooccurrence vectors
	context_vectors = load_context_vectors(path_ctxvec)

	# Compute COntingency Table
	Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total = compute_contingency_table(context_vectors)	

	if assoc == "mi":

		#Compute Mutual Information
		 context_vectors_assoc = compute_MI(context_vectors,Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total)

	if assoc == "odds":	
		# Compute Mutual Information
		context_vectors_assoc = compute_ODDS(context_vectors,Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total) 

	if assoc == "ll":	
		# Compute Mutual Information
		context_vectors_assoc = compute_LL(context_vectors,Tab_occ_X,Tab_cooc_XY,Tab_cooc_X_ALL,Tab_cooc_ALL_Y,Total)		

	# Save Association context vectors
	write_context_vectors(path_ctxvec_assoc, context_vectors_assoc, occ, min_occ,path_termlist_csv)

	


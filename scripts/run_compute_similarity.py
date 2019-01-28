# -*- coding: utf-8 -*-
#________________________________________ *** DSA *** _____________________________________________-
#																								   -
# 							  Distributional Standard Approach									   -
# 																							       - 
# ---------------- Step 5 --> Compute Context Vectors Similarity -----------------------------------
# 									                                                               -
# Author  : Amir HAZEM  			                                                               -
# Created : 19/11/2018				                       		                                   -
# Updated :	19/11/2018						                                                       -
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
from operator import itemgetter, attrgetter
from sklearn.metrics.pairwise import pairwise_distances
import codecs
import DSA
import time
import argparse
#---------------------------------------------------------------------------------------------------

# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--

# Assign id integers to target vocabulary
def get_id_vocab(sorted_vocab,target_vec):
	# union of target vocab (from context vectors) and vocabulary (all the vocubulary after stopwords and occurrence filtering)
	for x in sorted_vocab:
		word = x[0]
		freq = x[1]
		vocab_id[word] = freq
	for x in target_vec:
		vocab_id[word] = target_vec[x]
	for x in vocab_id:	
		tab_res.append((x,int(vocab_id[x])))	
	new_sorted_vocab = sorted(tab_res,key=itemgetter(1),reverse=True)	

	return new_sorted_vocab	
# ----------------------------------------------------------------------------------------------------
# Assign id integers to target words for which there is a context vector
def assign_target_vocab_id(sorted_vocab,target_vec,vocab): # id_start after id target_vectors 
	word_to_id = {}
	id_to_word = {}
	word_to_id_tgtvocab = {}
	id_to_wordtgtvocab = {}
	
	_id_target_cand_tgtvocab = 0
	new_sorted_vocab = []
	vocab_size = 0
	vocab_id = {}
	tab_res = []

	#print vocab
	#print len(target_vec)
	#print len(sorted_vocab)
	for x in target_vec:

		if vocab.has_key(x) :
			word = x
			freq = vocab[x]
			word_to_id_tgtvocab[word] = _id_target_cand_tgtvocab
			id_to_wordtgtvocab[_id_target_cand_tgtvocab] = word
			_id_target_cand_tgtvocab+=1		
		else:
			print "target id error !!!"
			print x

	# assign a unique id to the target vocabulary
	_id_target_cand = 0
	for x in sorted_vocab:
		word = x[0]
		freq = x[1]
		word_to_id[word] = _id_target_cand
		id_to_word[_id_target_cand] = word
		vocab_size += 1
		_id_target_cand += 1

	return word_to_id,id_to_word,_id_target_cand,sorted_vocab,word_to_id_tgtvocab,id_to_wordtgtvocab
# ----------------------------------------------------------------------------------------------------
# Assign termlis integer ids
def assign_termlist_vocab_id(source_vec):

	src_word_to_id = {}
	src_id_to_word = {}
	_id_terms = 0	
	termlist_size = 0

	for x in source_vec: # use a different table to avoid source/target homonyms
		vect = source_vec[x].split(':')
		word = (vect[0].split('#'))[0]
		freq = (vect[0].split('#'))[1]
		src_word_to_id[word] = _id_terms
		src_id_to_word[_id_terms] = word
		_id_terms+=1
		termlist_size +=1

	return 	src_word_to_id,src_id_to_word,termlist_size
# ----------------------------------------------------------------------------------------------------
# Build termlist matrix
def get_termlist_matrix(source_vec, sorted_vocab ,src_word_to_id,W_termlist):

	for x in source_vec:
		vect = source_vec[x].split(':')
		tmp = {}
		for j in range(1,len(vect)):
			word = (vect[j].split('#'))[0]
			freq = (vect[j].split('#'))[1]
			tmp[word] = freq
		line = ""	
		flag = 0
		count = 0
		for y in sorted_vocab:	
			if tmp.has_key(y[0]):
	
				line = line + ' ' + str(tmp[y[0]])
				flag +=1
				count += 1
			else:
				line = line + ' ' + '0'	

		#if flag == 0:
		#	print " aaaaaaaaaa" + x		
		values=line.split(' ')	
		coefs=np.asarray(values[1:],dtype='float32')	
		W_termlist[src_word_to_id[x], :] = coefs

	return 	W_termlist
# ----------------------------------------------------------------------------------------------------
# Build target matrix
def get_target_matrix(target_vec, sorted_vocab, tgt_word_to_id,W_target,tgt_word_to_id_tgtvocab):

	cpt_target_cand = 0
	for x in target_vec:

		vect = target_vec[x].split(':')
		tmp = {}
		for j in range(1,len(vect)):
			word = (vect[j].split('#'))[0]
			freq = (vect[j].split('#'))[1]
			tmp[word] = freq

		line = ""	
		flag = 0
		cpt_sorted = 0
		for y in sorted_vocab:	
			cpt_sorted += 1
			if tmp.has_key(y[0]):
				line = line + ' ' + str(tmp[y[0]])
				flag +=1
			else:
				line = line + ' ' + '0'	

		if flag == 0:
			print " aaaaaaaaaa" + x		
		values=line.split(' ')	

		coefs=np.asarray(values[1:],dtype='float32')	
		if tgt_word_to_id_tgtvocab.has_key(x):
			W_target[tgt_word_to_id_tgtvocab[x], :] = coefs

	return 	W_target
# ----------------------------------------------------------------------------------------------------
# 
def from_vec_to_matrix(sorted_vocab,source_vec,target_vec,termlist,cpt_target_cand,vocabulary):

	tgt_word_to_id  = {}
	tgt_id_to_word  = {}
	src_word_to_id  = {}
	src_id_to_word  = {}
	vocab_size 		= 0
	word_to_id_tgtvocab = {}
	id_to_wordtgtvocab  = {}
	termlist_size   = 0
	new_sorted_vocab = []

	# assign a unique id to the target vocabulary
	(tgt_word_to_id,tgt_id_to_word,vocab_size,new_sorted_vocab,word_to_id_tgtvocab,id_to_wordtgtvocab) = assign_target_vocab_id(sorted_vocab,target_vec,vocabulary)
	
	# assign a unique id to to evaluation list terms	
	(src_word_to_id,src_id_to_word,termlist_size) = assign_termlist_vocab_id(source_vec)	
	#print "vocab_size "    + str(vocab_size)
	#print "termlist_size " + str(termlist_size)	 

	# Get termlist vectors matrix
	W_termlist = np.zeros((termlist_size, vocab_size))	
	W_termlist = get_termlist_matrix(source_vec, new_sorted_vocab,src_word_to_id,W_termlist)	
	# Get target vectors matrix
	W_target = np.zeros((len(word_to_id_tgtvocab), vocab_size))
	W_target = get_target_matrix(target_vec, new_sorted_vocab,tgt_word_to_id,W_target,word_to_id_tgtvocab)	
	
	#print "-------------------- Compute Similarity Measure --------------------"	
	
	start_time = time.time()

	# Initialise similarity matrix: cos  = [ [0 for x in range(columns) ]  for y in range(rows)]
	cos = [ [0 for i in range(len(W_target)) ]  for j in range(len(W_termlist))]

	if sim.lower() == "cos":
		# Comput cosine similarity
		cos = 1 - pairwise_distances(W_termlist, W_target, metric='cosine')

	else:
		if sim.lower() == "jac":
			cpt_i = -1
			for i in W_termlist:
				cpt_i +=1
				cpt_j = 0
				#print cpt_i 
				for j in W_target:
					
					min_= np.minimum(i,j)
					max_= np.maximum(i,j)
					a = np.sum(min_) 
					b = np.sum(max_)
					if  b != 0 :
						jacc= a / b 
					else: 
						jacc = -9999	
			
					
					cos[cpt_i][cpt_j] = jacc
			
					cpt_j +=1
	
	# Execution time
	print "Similarity time cost: "
	print("--- %s seconds ---" % round((time.time() - start_time),3))			

	#print len(cos)
	# generate pairs similarity
	# Compute Mean Average Precision

	Rank={}
	for i in range(1,6000): #  6000 is an arbitrary number 100 is enough but for matter of map precision, greater is better 
		Rank[i]=0
	map_=0
	count = 0
	for i in range (0,termlist_size):
		tab_res=[]
 		for j in range(0,len(cos[i])):
 			tab_res.append((id_to_wordtgtvocab[j],cos[i][j]))
						
 		result=sorted(tab_res,key=itemgetter(1),reverse=True )
 		cpt=0
		for cand in result:
			cpt+=1
			if cand[0] == termlist[src_id_to_word[i]]:
				count+=1
				#print str(i)  + " "+ str(src_id_to_word[i]) + " " + (termlist[src_id_to_word[i]]) + " "+ str(cand[1]) + " " + " Rang : " + str(cpt)
				map_+=(1/cpt)
				for k in range(cpt,101):
					Rank[k]+=1  


	return 	Rank,termlist_size,map_			
	

# ----------------------------------------------------------------------------------------------------
# Load Occurence vectors	
def load_occurrence_vectors(path_occvec,min_occ): 

	occ 	= {}
	tab_res = []
	with  codecs.open(path_occvec,'r',encoding = 'utf-8') as f :

		for line in f:
			line = (line).strip()
			vect = line.split('\t')
			if len(vect)>1 and int(vect[1]) >= min_occ:# and int(vect[1]) >= min_occ: # avoid '' # to be solved beforehand and filter min_occ
				occ[vect[0]] = int(vect[1])
				tab_res.append((vect[0].rstrip(),int(vect[1])))	
		# Sort vocabulary	
		#print occ
		sorted_vocab = sorted(tab_res,key=itemgetter(1),reverse=True)		
		#print ('%s \tVocabulary size' % len(occ))		
	return occ,sorted_vocab
# ----------------------------------------------------------------------------------------------------
# Load context vectors
def load_context_vectors(path_vec):

	context_vectors = {}
	with  codecs.open(path_vec,'r',encoding = 'utf-8') as f :
		cpt = 0
		for line in f:
			cpt+=1
			#line = ((line.decode('utf-8'))).strip()
			line = ((line)).strip()
			vect = line.split(':')
			context_vectors[((vect[0].split('#'))[0]).rstrip()] = line
		#print ('%s \tTermlist size' % len(context_vectors))		
	return context_vectors,cpt
# ----------------------------------------------------------------------------------------------------
# Write matrix (not used for the moment)
def write_vect_matrix(path,matrix):
	#with  open(path,'w') as f1 :		
	with  codecs.open(path,'w',encoding = 'utf-8') as f :	
		for x in matrix:
			f1.write (np.array2string(x)  +  '\n')
# ----------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':


	source_vec      = {}
	target_vec      = {}
	termlist        = {}
	termlist_inv	= {}
	occ 	        = {}	  
	sorted_vocab    = []
	cpt_src_terms   = 0
	cpt_target_cand = 0
    

	# Load arguments ---------------------------------------------------------------------------------
	args = DSA.load_args()
	# ----------------------------------------------------------------------------------------------------


	# Parameters:---------------------------------------------------------------------------------------
	corpus 				= args.corpus 	    	# sys.argv[1]	   # Corpus	
	source_lang			= args.source_lang		# sys.argv[2]	   # 
	target_lang			= args.target_lang		# sys.argv[3]	   # 
	assoc       		= args.assoc			# sys.argv[4]	   # MI / JAC / ODDS 
	sim 				= args.sim              # cos / jac 	
	w 					= int(args.w)			# int(sys.argv[5]) # : window size 1/2/3... number of words before and after the center word
	min_occ				= int(args.min_occ)		# int(sys.argv[6]) # : filtering tokens with number of occurence less than min_occ
	termlist_name		= args.termlist_name	# sys.argv[7]	   # Term list name (evaluation list name)					   
	dictionary_name		= args.dictionary_name  # sys.argv[8] 	   # Bilinguam dictionary


	path_vocab_src     	  = "../data/train/corpora/" + corpus + '/tmp/vocab_'+source_lang + ".csv"
	path_vocab_tgt     	  = "../data/train/corpora/" + corpus + '/tmp/vocab_'+target_lang + ".csv"
	path_ctxvec_assoc_tgt = "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+target_lang + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc"
	path_ctxvec_trad      = "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+source_lang + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc.trad"
	path_dict 			  = "../data/train/dictionaries/" + dictionary_name	#dicfrenelda-utf8.txt"
	path_termlist_csv	  = "../data/train/termlists/"+ termlist_name
	# ----------------------------------------------------------------------------------------------------


	try: 
		print "Compute "+ DSA.Similarity_MAP(sim) +" Similarity..."

		# Load occurence vectors
		occ,sorted_vocab   			   = load_occurrence_vectors(path_vocab_tgt,min_occ)
		# Load source context vectors
		source_vec,cpt_src_terms	   = load_context_vectors(path_ctxvec_trad)
		# Load target Context vectors
		target_vec,cpt_target_cand	   = load_context_vectors(path_ctxvec_assoc_tgt)
		# Load termlist
		termlist,termlist_inv		   =  DSA.load_termlist(path_termlist_csv)	
		# Compute similarity
		Rank,termlist_size,map_ 	   = from_vec_to_matrix(sorted_vocab,source_vec,target_vec,termlist,cpt_target_cand,occ)
		print "Done."
		# print final result :
		DSA.print_MAP_scores(Rank,termlist_size,map_)	
	except:
		print "Unexpected error ", sys.exc_info()[0]














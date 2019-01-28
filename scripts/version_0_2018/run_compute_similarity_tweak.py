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
import treetaggerwrapper
from operator import itemgetter, attrgetter
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import jaccard_similarity_score
from scipy import spatial
#---------------------------------------------------------------------------------------------------

# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--


# Parameters:---------------------------------------------------------------------------------------
corpus 				= sys.argv[1]	   # Corpus	
lang_src 			= sys.argv[2]	   # Language : en/fr/...	
lang_tgt 			= sys.argv[3]	   # Language : en/fr/...	
assoc       		= sys.argv[4]	   # MI / ll / ODDS 
sim 				= sys.argv[5]	   # cos / jac 	
w 					= int(sys.argv[6]) # : window size 1/2/3... number of words before and after the center word
min_occ				= int(sys.argv[7]) # : filtering tokens with number of occurence less than min_occ

path_vocab_src     	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+lang_src + ".csv"
path_vocab_tgt     	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+lang_tgt + ".csv"


path_ctxvec_assoc_tgt = "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang_tgt + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc"

path_ctxvec_trad    = "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang_src + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc.trad"

path_dict 			= "../data/train/dictionaries/dicfrenelda-utf8.txt"

path_termlist_csv	= "../data/train/termlists/" + lang_src +'_' + lang_tgt + '_' + corpus +"_248.csv"
# ----------------------------------------------------------------------------------------------------


def assign_target_vocab_id(sorted_vocab,target_vec,id_start): # id_start after id target_vectors 
	word_to_id = {}
	id_to_word = {}
	_id = id_start
	_id_target_cand = 0

	vocab_size = 0
	# assign a unique id to the target vocabulary
	for x in sorted_vocab:
		word = x[0]
		freq = x[1]
		
		if target_vec.has_key(word):
	
			word_to_id[word] = _id_target_cand
			id_to_word[_id_target_cand ] = word
			_id_target_cand += 1			
	
	
		else:
			#print word + " "+ str(_id)
			word_to_id[word] = _id
			id_to_word[_id ] = word
			_id+=1
	
		vocab_size += 1
	#print str(_id) + " " + str(_id_target_cand) + " "+ str(id_start) 
	
	return word_to_id,id_to_word,vocab_size	


def assign_termlist_vocab_id(source_vec):

	src_word_to_id = {}
	src_id_to_word = {}
	_id_terms = 0	
	termlist_size = 0


	for x in source_vec: # use a different table to avoid source/target homonyms
		vect = source_vec[x].split(':')
		word = (vect[0].split('#'))[0]
		freq = (vect[0].split('#'))[1]
		#print word + " "+ str(freq) + " "+ str(_id)
		src_word_to_id[word] = _id_terms
		src_id_to_word[_id_terms ] = word
		_id_terms+=1
		termlist_size +=1

	return 	src_word_to_id,src_id_to_word,termlist_size

def get_termlist_matrix(source_vec, sorted_vocab ,src_word_to_id,W_termlist):

	for x in source_vec:
		#print x
		vect = source_vec[x].split(':')
		tmp = {}
		for j in range(1,len(vect)):
			#print vect[j]
			word = (vect[j].split('#'))[0]
			freq = (vect[j].split('#'))[1]
			#print word + " "+ str(freq)
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


		if flag == 0:
			print " aaaaaaaaaa" + x		
		values=line.split(' ')	

		coefs=np.asarray(values[1:],dtype='float32')	

		#print x + " "+ str(src_word_to_id[x])
		W_termlist[src_word_to_id[x], :] = coefs
		#print x + " "+ str(src_word_to_id[x]) + " "+ str(count)
		#print W_termlist[src_word_to_id[x], :] 
		#print coefs
		#print W

		#print len(W_termlist)
	#sys.exit()	
	return 	W_termlist


def get_target_matrix(target_vec, sorted_vocab, tgt_word_to_id,W_target):

	cpt_target_cand = 0
	for x in target_vec:

		#print "cpt_target_cand --> "+ " "+ str(cpt_target_cand)
		cpt_target_cand += 1
		#print x
		vect = target_vec[x].split(':')
		#print vect 
		#sys.exit()
		tmp = {}
		for j in range(1,len(vect)):
			word = (vect[j].split('#'))[0]
			freq = (vect[j].split('#'))[1]
			#print word + " "+ str(freq)
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


		#print "cpt_sorted --> "+ " "+ str(cpt_sorted)		

		if flag == 0:
			print " aaaaaaaaaa" + x		
		values=line.split(' ')	

		coefs=np.asarray(values[1:],dtype='float32')	

		#print x + " "+ str(tgt_word_to_id[x])
		if tgt_word_to_id.has_key(x):
			W_target[tgt_word_to_id[x], :] = coefs
		# if not that means that the wods has been filtered (hapax or freq less than min_)	
		#print coefs
		#print W

		#print len(W_target)
	
	return 	W_target






def from_vec_to_matrix(sorted_vocab,source_vec,target_vec,termlist,cpt_target_cand):


	tgt_word_to_id  = {}
	tgt_id_to_word  = {}
	src_word_to_id  = {}
	src_id_to_word  = {}
	vocab_size 		= 0
	termlist_size   = 0
	

	# assign a unique id to the target vocabulary
	(tgt_word_to_id,tgt_id_to_word,vocab_size) = assign_target_vocab_id(sorted_vocab,target_vec,cpt_target_cand)
	
	# assign a unique id to to evaluation list terms	
	(src_word_to_id,src_id_to_word,termlist_size) = assign_termlist_vocab_id(source_vec)	
	print "vocab_size " + str(vocab_size)
	print "termlist_size " + str(termlist_size)	 

	# Get termlist vectors matrix
	W_termlist = np.zeros((termlist_size, vocab_size))	
	W_termlist = get_termlist_matrix(source_vec, sorted_vocab,src_word_to_id,W_termlist)	

	#print W_termlist/W_termlist
	

	#for x in src_id_to_word:
#		print str(x) + " " + src_id_to_word[x]
#	print "_____________________"	
#	for x in tgt_id_to_word:
#		print str(x) + " " + tgt_id_to_word[x]	

	#sys.exit()	


	# Get target vectors matrix
	W_target = np.zeros((cpt_target_cand, vocab_size))
	W_target = get_target_matrix(target_vec, sorted_vocab,tgt_word_to_id,W_target)	


	path = "termlist_matrix.txt"
	write_vect_matrix(path,W_termlist)

	path = "target_matrix.txt"
	write_vect_matrix(path,W_target)

	#sys.exit()

	# Compute similarity

	#W_termlist_norm = np.zeros(W_termlist.shape)
	#d = (np.sum(W_termlist ** 2, 1) ** (0.5))
	#W_termlist_norm = (W_termlist.T / d).T


	#W_target_norm = np.zeros(W_target.shape)
	#d = (np.sum(W_target ** 2, 1) ** (0.5))
	#W_target_norm = (W_target.T / d).T


	#W_termlist = W_termlist_norm
	#W_target   = W_target_norm

	# Quotion
	#Q = np.dot(W_termlist, W_target.T) 
	# Ratio
	print "--------------------termlist-------------------"	
#	print W_termlist
	#R1=np.sqrt((W_termlist)**2)
	#R2=np.sqrt((W_target.T)**2)



	#print R1/R1
	
	import time
	start_time = time.time()


	#cos = Q/(np.dot(R1,R2))

	#cos = cosine_similarity(W_termlist, W_target)
	cos = 1 - pairwise_distances(W_termlist, W_target, metric='cosine')

	print("--- %s seconds ---" % (time.time() - start_time))
	#sys.exit()
	#cos =  1 -  pairwise_distances(W_termlist.astype('bool'), W_target.astype('bool'), metric='jaccard')

	#print cos
	#cos = {}
	#print jac

	if sim == "jac":
		cpt_i = -1

		for i in W_termlist:
			cpt_i +=1
			cpt_j = 0
			print cpt_i 
			for j in W_target:
				#cos[cpt_i][cpt_j] =  1-spatial.distance.cosine(i, j)
				#cos[cpt_i][cpt_j] = jaccard_similarity_score(i, j)
				#print i
				#print j

				min_= np.minimum(i,j)
				max_= np.maximum(i,j)
				a = np.sum(min_) 
				b = np.sum(max_)
				

				if  b != 0 :
					jacc= a / b 
				else: 
					jacc = -9999	
				#print str(a) + " "+ str(b) + " "+ str(jacc)	
				#print jacc
				cos[cpt_i][cpt_j] = jacc
				#cos[cpt_i][cpt_j] = jaccard_similarity_score(i, j, normalize=True)
				#print cos[cpt_i][cpt_j]
				cpt_j +=1
		#cos = Q/(R2)
		#print cos
				#sys.exit()
	print len(cos)
	# generate pairs similarity

	Rank={}

	for i in range(1,6000):
		Rank[i]=0
	map_=0
	count = 0
	for i in range (0,termlist_size):
		#print "Source ref termlist : "  + src_id_to_word[i]
		#print str(i) + " " +str(src_id_to_word[i])  + " " + termlist[src_id_to_word[i]] + " " +str(cos[i])

		tab_res=[]
 		for j in range(0,len(cos[i])):
 			#print str(i) + " "+ id_to_word[i]+ " "+ str( cos[ref_id_inv[word_to_id[input_term]]][i])	

 			tab_res.append((tgt_id_to_word[j],cos[i][j]))
 			#.print str(i) + " "+ str(j) + " "+  tgt_id_to_word[j] + " "+ str(cos[i][j])	
 			

 						
 		result=sorted(tab_res,key=itemgetter(1),reverse=True )


 		cpt=0

 		#print "****************************** " #+ src_id_to_word[i] + " " +termlist[src_id_to_word[i]]
		for cand in result:
			#print (cand[0])+ " "+ termlist[src_id_to_word[i]]
			
		#.	print str(i)  + " "+ str(src_id_to_word[i]) + " " + cand[0] + " "+ str(cand[1]) + " " + " Rang : " + str(cpt)
			cpt+=1
			if cand[0] == termlist[src_id_to_word[i]]:
				count+=1
				print str(i)  + " "+ str(src_id_to_word[i]) + " " + (termlist[src_id_to_word[i]]) + " "+ str(cand[1]) + " " + " Rang : " + str(cpt)
				map_+=(1/cpt)
				for k in range(cpt,101):
					Rank[k]+=1  
		#print "******************************"			

	#sys.exit()	

	# print final result :

	for k in range(1,101):
	
		if k==1:
			print str(k) +" "+str(Rank[k]) + " "+ str((Rank[k]/len(termlist))*100) + " "+ str(len(termlist))

		if k%5==0:
			print str(k) +" "+ str(Rank[k]) + " "+ str((Rank[k]/len(termlist))*100) + " "+ str(len(termlist))



	map_=(map_/(termlist_size))*100

	print "MAP = " + str(map_) + " "+ str(termlist_size)
	
	print str(count)




	sys.exit()


def load_occurrence_vectors(path_occvec,min_occ): 

	occ 	= {}
	tab_res = []

	f = open(path_occvec,'r')
	
	for line in f:
		
		line = ((line.decode('utf-8'))).strip()
		vect = line.split('\t')
		if len(vect)>1 and int(vect[1]) >= 2:# and int(vect[1]) >= min_occ: # avoid '' # to be solved beforehand and filter min_occ
			occ[vect[0]] = int(vect[1])
			tab_res.append((vect[0],int(vect[1])))	


	# Sort vocabulary	
	sorted_vocab = sorted(tab_res,key=itemgetter(1),reverse=True)		
	print ('%s \tVocabulary size' % len(occ))		
	return occ,sorted_vocab



def load_context_vectors(path_vec,val,twick):

	context_vectors = {}

	f = open(path_vec,'r')
	cpt = 0
	for line in f:
		cpt+=1
		line = ((line.decode('utf-8'))).strip()
		vect = line.split(':')

		vecttwick=""
		for k in range(1,len(vect)):
			ch  = vect[k].split('#')
			#print ch[1]
			if float(ch[1]) < -val:# or float(ch[1]) > val :
				if float(ch[1]) < -val:
					twick = -twick
				
				vecttwick = vecttwick + ':' + ch[0]+'#'+ str(twick)
			else:
				vecttwick = vecttwick + ':' + vect[k]

		
		vecttwick = vect[0]  + vecttwick					
		#print vecttwick
		#sys.exit()
		context_vectors[(vect[0].split('#'))[0]] = vecttwick
	
	print ('%s \tContext vectors size' % len(context_vectors))		
	return context_vectors,cpt

def load_termlist(path_termlist):

	termlist = {}
	f = open(path_termlist,'r')
	for line in f:
		#line = ((line.decode('utf-8'))).strip()
		line = ((line)).strip()
		vect = line.split('\t')
		termlist[vect[0]] = vect[1].decode('utf-8')
	#	print 	line
	print ('%s \tTermlist size' % len(termlist))		
	#sys.exit()
	return termlist


def write_vect_matrix(path,matrix):
	with  open(path,'w') as f1 :		

		for x in matrix:
			
			
			f1.write (np.array2string(x)  +  '\n')


#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':

	source_vec      = {}
	target_vec      = {}
	termlist        = {}
	occ 	        = {}	  
	sorted_vocab    = []
	cpt_src_terms   = 0
	cpt_target_cand = 0

	val   = 1.
	twick = 0
    
	occ,sorted_vocab   			   = load_occurrence_vectors(path_vocab_tgt,min_occ)
	source_vec,cpt_src_terms	   = load_context_vectors(path_ctxvec_trad,val,twick)
	target_vec,cpt_target_cand	   = load_context_vectors(path_ctxvec_assoc_tgt,val,twick)
	termlist 		   			   = load_termlist(path_termlist_csv)	

	from_vec_to_matrix(sorted_vocab,source_vec,target_vec,termlist,cpt_target_cand)















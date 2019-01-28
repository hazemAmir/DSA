# -*- coding: utf-8 -*-
#________________________________________ *** DSA *** _____________________________________________-
#																								   -
# 							  Distributional Standard Approach									   -
# 																							       - 
# * Standard Approach: Step 4 --> Translate Association Context Vectors ----------------------------
# 									                                                               -
# Author  : Amir HAZEM  			                                                               -
# Created : 19/11/2018				                       		                                   -
# Updated :	19/11/2018						                                                       -
# source  : 									 												   -
# 		  python run_translate_context_vectors.py breast_cancer en fr mi 3 5				       - 
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
import codecs
import DSA
import argparse
#---------------------------------------------------------------------------------------------------

# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--

# save translated source context vector
def write_context_vectors(path_ctxvec_trad, vect):
	with  codecs.open(path_ctxvec_trad,'w',encoding = 'utf-8') as f:	
		for x in vect:
			f.write (vect[x] + '\n')
# ----------------------------------------------------------------------------------------------------
# Load Dictionary 
def load_fren_elra_dictionary(path_dict, occ_src,occ_tgt,termlist):
	dico = {}
	dico_cooc = {} # keep the total cooccurrence of the head word
	with  codecs.open(path_dict,'r',encoding = 'utf-8') as f:
		count_all = 0
		count_distinct = 0
		for line in f:
			line = (line).strip()
			line = line.split(';')
			word_tgt = line[0]
			word_src = line[3]
																	   # uncomment this termlist filtering for chain comparison	
			if occ_src.has_key(word_src) and occ_tgt.has_key(word_tgt) and not termlist.has_key(word_src):
				
				if dico.has_key(word_src):
					dico[word_src] = dico[word_src] + '::' + word_tgt
					dico_cooc[word_src] += occ_tgt[word_tgt]
				else:	
					dico[word_src] = word_tgt
					dico_cooc[word_src] = occ_tgt[word_tgt]
					count_distinct +=1
				count_all +=1	
	#print "Projected dico size" + " ALL --> "+ str(count_all) + " Distinct --> "+ str(count_distinct)		
	return dico,dico_cooc
# ----------------------------------------------------------------------------------------------------
# Translate the source contexst vector
def trad_context_vectors(termlist, context_vectors_assoc, dico, dico_cooc, occ_src, occ_tgt):
	
	context_vectors_trad = {}
	for src in termlist:
		
		if context_vectors_assoc.has_key(src):
			vect = context_vectors_assoc[src].split(':')
			head 	 = (vect[0].split('#'))[0]
			head_occ = (vect[0].split('#'))[1]
			
			vect_trad = ""
			for j in range (1,len(vect)):

				word 	   = (vect[j].split('#'))[0]
				word_assoc = float((vect[j].split('#'))[1])

				if dico.has_key(word) and word != head:
					#print "---------------------------"
					trad = dico[word].split("::")
						
					for k in trad:

						new_assoc = (word_assoc * occ_tgt[k]) /dico_cooc[word]
						vect_trad = vect_trad + ':' + k + '#' + str(new_assoc)
				#.else:
				#.	if word == head:
				#.		if dico.has_key(head):
				#.			print word + " "+ head + " "+ dico[head]		
				#.		else:
				#.			print word + " "+ head + " no translation available"
			context_vectors_trad[head] = vect[0] + vect_trad
		else:
			print src + " does not exist"

	return context_vectors_trad
# ----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
# Main 
#-----------------------------------------------------------------------------------------------------

if __name__=='__main__':

	occ_src 			  = {}
	occ_tgt 			  = {}
	context_vectors_assoc = {}
	context_vectors_trad  = {}
	dico 				  = {}
	dico_cooc			  = {}
	termlist  			  = {}	
	termlist_inv		  = {}	

	# Load arguments ---------------------------------------------------------------------------------
	args = DSA.load_args()
	# ----------------------------------------------------------------------------------------------------

	# Parameters:---------------------------------------------------------------------------------------
	corpus 				= args.corpus 	    	# sys.argv[1]	   # Corpus	
	source_lang			= args.source_lang		# sys.argv[2]	   # 
	target_lang			= args.target_lang		# sys.argv[3]	   # 
	assoc       		= args.assoc			# sys.argv[4]	   # MI / JAC / ODDS 
	w 					= int(args.w)			# int(sys.argv[5]) # : window size 1/2/3... number of words before and after the center word
	min_occ				= int(args.min_occ)		# int(sys.argv[6]) # : filtering tokens with number of occurence less than min_occ
	termlist_name		= args.termlist_name	# sys.argv[7]	   # Term list name (evaluation list name)					   
	dictionary_name		= args.dictionary_name  # sys.argv[8] 	   # Bilinguam dictionary


	path_vocab_src     	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+source_lang + ".csv"
	path_vocab_tgt     	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+target_lang + ".csv"
	path_ctxvec			= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+source_lang + '_w' + str(w) + ".vect"
	path_ctxvec_assoc	= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+source_lang + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc"
	path_ctxvec_trad    = "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+source_lang + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc.trad"
	path_dict 			= "../data/train/dictionaries/" + dictionary_name #dicfrenelda-utf8.txt"
	path_termlist_csv	= "../data/train/termlists/"+ termlist_name
	# ----------------------------------------------------------------------------------------------------


	try: 
		print "Translate " +  DSA.Language_MAP(source_lang) + " source context vectors..."
		# Load source vocabulary
		occ_src 			  = DSA.load_occurrence_vectors(path_vocab_src)
		# Load target vocabulary
		occ_tgt 			  = DSA.load_occurrence_vectors(path_vocab_tgt)
		# Load source context vectors
		context_vectors_assoc = DSA.load_context_vectors(path_ctxvec_assoc)
		# Load term list (evaluation list)
		termlist,termlist_inv =	DSA.load_termlist(path_termlist_csv)	
		# Load bilingual dictionary
		dico,dico_cooc		  = load_fren_elra_dictionary(path_dict,occ_src,occ_tgt,termlist)
		# Translate source context vectors
		context_vectors_trad  = trad_context_vectors(termlist, context_vectors_assoc, dico, dico_cooc, occ_src, occ_tgt)
		# Save translated context vectors
		write_context_vectors(path_ctxvec_trad, context_vectors_trad)

		print "Done."	
	except:
		print "Unexpected error ", sys.exc_info()[0]	










# -*- coding: utf-8 -*-
#________________________________________ *** DSA *** _____________________________________________-
#																								   -
# 							  Distributional Standard Approach									   -
# 																							       - 
# * Standard Approach: Step 2 --------> Context Vectors --------------------------------------------
# 									                                                               -
# Author  : Amir HAZEM  			                                                               -
# Created : 19/11/2018				                       		                                   -
# Updated :	19/11/2018						                                                       -
# source  : 									 												   -
# 		  python run_context_vectors.py breast_cancer en postag 1 3							       - 
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
import treetaggerwrapper
from operator import itemgetter, attrgetter
#---------------------------------------------------------------------------------------------------

# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--


# Parameters:---------------------------------------------------------------------------------------
corpus 		= sys.argv[1]
lang 		= sys.argv[2]	# Language : en/fr/...	
corpus_type = sys.argv[3]	# Flag     : tok/lem/postag
flag_filter = True if  int(sys.argv[4]) == 1 else False	# Filter stopwords 1/0
w 			= int(sys.argv[5]) # : window size 1/2/3... number of words before and after the center word
#min_occ		= int(sys.argv[6]) # : default 1 (no filtering) 
							   #   filtering tokens with number of occurence less than min_occ 
							   #   better not to filter now (do it after association measures computation)

corpus_dir  	= "../data/train/corpora/" + corpus  + '/' + corpus_type + '/' + lang
stopwords_path 	= "../data/train/stopwords/" + "stopwords_" + lang + ".txt" 
path_vocab      = "../data/train/corpora/" + corpus + '/tmp/vocab_'+lang + ".csv"
path_ctxvec		= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang + '_w' + str(w) + ".vect"


def postag_norm(tag,lang):
	
	if lang =="fr": #------------------------------------------------------------------------------- 
		# Verb TAGS: VER:cond conditional, VER:futu futur, VER:impe imperative, VER:impf imperfect, 
		# VER:infi infinitive, VER:pper past participle, VER:ppre present participle, VER:pres present, 
		# VER:simp simple past, VER:subi subjunctive imperfect, VER:subp subjunctive present
		# Noun Tags: N = NAM proper name, NOM	noun
		# Adjective Tags: ADJ
		if tag in ['VER:cond', 'VER:futu', 'VER:impe', 'VER:impf','VER:infi','VER:pper', 'VER:ppre','VER:pres','VER:simp', 'VER:subi','VER:subp']: tag = 'v'
		if tag in ['NOM','NAM']: tag = 'n'
		if tag in ['ADJ']:		 tag = 'a'

	if lang =="en": #------------------------------------------------------------------------------- 	
		# Verb TAGS: VB  base form, VBD past tense, VBG gerund or present participle, VBN past participle
		# VBP non-3rd person singular present, VBZ 3rd person singular present 
		# Noun Tags: NN singular or mass, NNS plural, NNP Proper noun singular,	NNPS Proper noun plural 
		# Adjective Tags: JJ Adjective, JJR	Adjective comparative, JJS 	Adjective superlative 
		if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: tag = 'v'
		if tag in ['NN', 'NNS', 'NNP', 'NNPS']: 			 tag = 'n' 
		if tag in ['JJ', 'JJR', 'JJS']: 					 tag = 'a' 

	return tag	


def write_vocab(path_vocab,occ):

	tab_res = []

	for tok in occ:
		#print tok + " "+ str(occ[tok])
		tab_res.append((tok,occ[tok]))	

	# Sort vocabulary	
	result=sorted(tab_res,key=itemgetter(1),reverse=True)	
	with  open(path_vocab,'w') as f1 :
		for tok in result:
			f1.write (tok[0].encode('utf-8') + '\t' + str(tok[1])+ '\n')


def write_context_vectors(path_ctxvec,coocc, occ):

	context_vectors={}
	
	for pair in coocc:
		
		xy = pair.split(' ')
		head = xy[0]
		
		
		tail = xy[1]+"#"+ str(coocc[pair])

		if context_vectors.has_key(head):
			context_vectors[head] = context_vectors[head] + ":" + tail
		else:
			context_vectors[head] = tail	

	with  open(path_ctxvec,'w') as f1 :		

		for x in context_vectors:
			count = occ[x] # len(context_vectors[x].split(':'))
			line = x + '#'+ str(count) +':'+ context_vectors[x]				
			f1.write (line.encode('utf-8') + '\n')			







	


def load_stopwords(path):
	stopwords = {}
	f = open(path,'r')
	for line in f:
		tab = (line.decode('utf-8')).split()
		stopwords[tab[0]] = ((tab[0].strip()).lower())
		
	f.close()
	print ('%s \tStopwords' % len(stopwords))

	return stopwords

def build_context_vectors(corpus_dir,lang,corpus_type,stopwords,w):
	tab_occ = {}
	tab_coocc = {}


	onlyfiles = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]
	
	for filename in onlyfiles:
		print filename


		f = open(corpus_dir+"/"+filename,'r')
		for line in f:
			line = ((line.decode('utf-8'))).strip()

			
			#print "Ori : ---> " + line+ "\n" 
			sent = ""
			if corpus_type == 'tok' or corpus_type == 'lem':
				if flag_filter:
					# Filter stopwords
					sent = ' '.join(x for x in line.split(' ') if not stopwords.has_key(x) and x.isalpha())
				else:
					sent = ' '.join(x for x in line.split(' ') if x.isalpha())	

			if corpus_type == "postag" :
				if flag_filter:
					# Filter stopwords
					sent = ' '.join(x for x in line.split(' ') if not stopwords.has_key((x.split('_pos:')[0]).lower()) and (x.split('_pos:')[0]).isalpha() and postag_norm(x.split('_pos:')[1],lang) in ['n','v','a'])
				else:
					sent = ' '.join(x for x in line.split(' ') if (x.split('_pos:')[0]).isalpha() and postag_norm(x.split('_pos:')[1],lang) in ['n','v','a'])		



			# count cooccurences :		
			#print "FILT : ---> " + sent		
			cpt = 0
			seg = sent.split(' ')
			for i in seg:

				#print i + " "+ str(cpt)

				# Count vocabulary occurence -----
				token = (i.split('_pos:')[0]).lower()
				if tab_occ.has_key(token):
					tab_occ[token] += 1
				else:
					tab_occ[token] = 1
				# --------------------------------	

				# Count cooccurence pairs -------------------------------------------
				for j in range(cpt-w,cpt+w+1):
					#print j
					if j>=0 and j<len(seg):
						if i != seg[j]:
							pair = token + " "+ (seg[j].split('_pos:')[0]).lower()
							if 	tab_coocc.has_key(pair): 
								tab_coocc[pair]+=1 
							else: 
								tab_coocc[pair]= 1
							#print pair
				cpt+=1
				# --------------------------------------------------------------------	
	return tab_occ,tab_coocc


#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':

	stopwords = {}
	occ   	  = {}
	coocc 	  = {}
	tab_res   = []

	stopwords = load_stopwords(stopwords_path)
	
	occ,coocc=build_context_vectors(corpus_dir,lang,corpus_type, stopwords,w)	

	write_vocab(path_vocab,occ)	 
	write_context_vectors(path_ctxvec,coocc,occ)
	

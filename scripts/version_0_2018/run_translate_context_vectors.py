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
import treetaggerwrapper
from operator import itemgetter, attrgetter
import xml.etree.ElementTree as ET
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
assoc       		= sys.argv[4]	   # MI / JAC / ODDS 
w 					= int(sys.argv[5]) # : window size 1/2/3... number of words before and after the center word
min_occ				= int(sys.argv[6]) # : filtering tokens with number of occurence less than min_occ

path_vocab_src     	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+lang_src + ".csv"
path_vocab_tgt     	= "../data/train/corpora/" + corpus + '/tmp/vocab_'+lang_tgt + ".csv"
path_ctxvec			= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang_src + '_w' + str(w) + ".vect"

path_ctxvec_assoc	= "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang_src + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc"
path_ctxvec_trad    = "../data/train/corpora/" + corpus + '/context_vectors/'+corpus+'_'+lang_src + '_w' + str(w) + "_min" + str(min_occ)+ "_"+ assoc + ".assoc.trad"
path_dict 			= "../data/train/dictionaries/dicfrenelda-utf8.txt"
path_termlist		= "../data/train/termlists/en_fr_breast_248.xml"
path_termlist_out	= "../data/train/termlists/" + lang_src +'_' + lang_tgt + '_' + corpus +"_248.csv"
path_termlist_csv	= "../data/train/termlists/" + lang_src +'_' + lang_tgt + '_' + corpus +"_248.csv"
# ----------------------------------------------------------------------------------------------------


def load_occurrence_vectors(path_occvec):

	occ = {}
	f = open(path_occvec,'r')
	for line in f:
		
		line = ((line.decode('utf-8'))).strip()
		vect = line.split('\t')
		if len(vect)>1: # avoid '' # to be solved beforehand
			occ [vect[0]]= int(vect[1])

	print ('%s \tVocabulary size' % len(occ))		
	return occ


def write_context_vectors(path_ctxvec_trad, context_vectors_trad):

		
	with  open(path_ctxvec_trad,'w') as f1 :		

		for x in context_vectors_trad:
			
			f1.write (context_vectors_trad[x].encode('utf-8') + '\n')



def from_xml_to_csv_termlist(path_termlist,path_termlist_out):
	
	tree = ET.parse(path_termlist)
	root=tree.getroot()

	termlist={}

	for cand in root.findall('TRAD'):
		#print cand
		valid=cand.get('valid')
		if valid=="yes":
			for lang in cand.findall('LANG'):
				ln= lang.get('type')
				for term in lang.findall('TERM'):
					if ln =="en":
						en= (term.text)#.decode('utf-8')  
					else:
						fr= (term.text)#.decode('utf-8') 

	#print "fr :" + fr			
	#print "en :"+ en

		termlist[en]=fr#.encode('utf-8')

	with  open(path_termlist_out,'w') as f1 :	
		count=0
		for en in termlist:
			#print en + " "+ termlist[en]
			line = en+'\t'+ termlist[en]
			f1.write(line.encode('utf-8') + '\n')
			#f1.write(line + '\n')
			count+=1

	print str(count)


def load_context_vectors(path_ctxvec_assoc):

	context_vectors = {}
	f = open(path_ctxvec_assoc,'r')
	for line in f:
		line = ((line.decode('utf-8'))).strip()
		vect = line.split(':')
		context_vectors[(vect[0].split('#'))[0]]=line
	
	print ('%s \tContext vectors size' % len(context_vectors))		
	return context_vectors


def load_termlist(path_termlist):

	termlist = {}
	f = open(path_termlist,'r')
	for line in f:
		line = ((line.decode('utf-8'))).strip()
		vect = line.split('\t')
		termlist[vect[0]] = vect[1]
	
	print ('%s \tTermlist size' % len(termlist))		
	return termlist



def load_fren_elra_dictionary(path_dict, occ_src,occ_tgt,termlist):

	dico = {}
	dico_cooc = {} # keep the total cooccurrence of the head word
	f = open(path_dict,'r')
	count_all = 0
	count_distinct = 0
	for line in f:
		line = ((line.decode('utf-8'))).strip()
		line = line.split(';')
		word_tgt = line[0]
		word_src = line[3]
		
																   # uncomment this termlist filtering for chain comparison	
		if occ_src.has_key(word_src) and occ_tgt.has_key(word_tgt) and not termlist.has_key(word_src):
			#print word_src + " "+ word_tgt
			if dico.has_key(word_src):
				dico[word_src] = dico[word_src] + '::' + word_tgt
				dico_cooc[word_src] += occ_tgt[word_tgt]

			else:	
				dico[word_src] = word_tgt
				dico_cooc[word_src] = occ_tgt[word_tgt]
				count_distinct +=1

			count_all +=1	
	print "filetered dico size" + " ALL --> "+ str(count_all) + " Distinct --> "+ str(count_distinct)		
	return dico,dico_cooc


def trad_context_vectors(termlist, context_vectors_assoc, dico, dico_cooc, occ_src, occ_tgt):
	
	context_vectors_trad = {}

	for src in termlist:
		#print context_vectors_assoc[src]

		vect = context_vectors_assoc[src].split(':')

		head 	 = (vect[0].split('#'))[0]
		head_occ = (vect[0].split('#'))[1]
		
		#print head + " " + str(head_occ) 

		vect_trad = ""
		for j in range (1,len(vect)):

			word 	   = (vect[j].split('#'))[0]
			word_assoc = float((vect[j].split('#'))[1])

			#print vect[j] 

			if dico.has_key(word) and word != head:
				#print "---------------------------"
				#print dico[word] 
				trad = dico[word].split("::")
				#print "source word :" + word

				
				for k in trad:

					new_assoc = (word_assoc * occ_tgt[k]) /dico_cooc[word]
					#print k + " "+ str(new_assoc)
					vect_trad = vect_trad + ':' + k + '#' + str(new_assoc)

			else:
				if word == head:
					if dico.has_key(head):
						print word + " "+ head + " "+ dico[head]		
					else:
						print word + " "+ head + " no translation available"
		#vect_trad = head + '#' + head_occ + vect_trad 			
		context_vectors_trad[head] = vect[0] + vect_trad
		
	return context_vectors_trad



#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':

	occ_src 			  = {}
	occ_tgt 			  = {}
	context_vectors_assoc = {}
	context_vectors_trad  = {}
	dico 				  = {}
	dico_cooc			  = {}
	termlist  			  = {}	

	#from_xml_to_csv_termlist(path_termlist,path_termlist_out)
	#sys.exit()

	occ_src 			  = load_occurrence_vectors(path_vocab_src)

	occ_tgt 			  = load_occurrence_vectors(path_vocab_tgt)

	context_vectors_assoc = load_context_vectors(path_ctxvec_assoc)

	termlist 			  =	load_termlist(path_termlist_csv)	

	dico,dico_cooc		  = load_fren_elra_dictionary(path_dict,occ_src,occ_tgt,termlist)

	
	

	context_vectors_trad = trad_context_vectors(termlist, context_vectors_assoc, dico, dico_cooc, occ_src, occ_tgt)

	write_context_vectors(path_ctxvec_trad, context_vectors_trad)











# -*- coding: utf-8 -*-
#________________________________________ *** DSA *** _____________________________________________-
# 																							       - 
# * Standard Approach: Step 1 ---------> Preprocessings --------------------------------------------
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
import treetaggerwrapper
#---------------------------------------------------------------------------------------------------

# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'		  #--
np.random.seed(42)					  	  #--
rn.seed(12345)						  	  #--	
# ----------------------------------------#--


# Parameters:---------------------------------------------------------------------------------------
corpus 		= sys.argv[1]
lang 		= sys.argv[2]	# Language : en/fr/...	

lem_flag	= True if  int(sys.argv[3]) == 1 else False	# 1/0
pos_flag	= True if  int(sys.argv[4]) == 1 else False	# 1/0

TAGDIR		= "./tree-tagger-linux/"
corpus_dir  = "../data/train/corpora/" + corpus + "/raw/" + lang
out_tok		= "../data/train/corpora/" + corpus + "/tok/" + lang
out_lem		= "../data/train/corpora/" + corpus + "/lem/" + lang	
out_pos		= "../data/train/corpora/" + corpus + "/postag/" + lang

#---------------------------------------------------------------------------------------------------

# Preprocesses the corpus tokenization lemmatization and postagging regarding the parameters
def preprocessings(corpus_dir,LANG,TAGDIR,lem_flag,pos_flag):

	onlyfiles = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]

	tagger = treetaggerwrapper.TreeTagger(TAGLANG=LANG,TAGINENC='utf-8',TAGOUTENC='utf-8',TAGDIR=TAGDIR)

	if lem_flag: print "Lemmatization... ok"
	if pos_flag: print "PosTagging... ok"
	
	for filename in onlyfiles:
		print filename
	
		with  open(out_tok+'/'+filename,'w') as f1 ,open(out_lem+'/'+filename,'w') as f2,open(out_pos+'/'+filename,'w') as f3:

			f = open(corpus_dir+"/"+filename,'r')
			for line in f:
				line = (line.decode('utf-8')).strip()
				
				# Tokenization ---------------------------------------
				sent_tok  = (' '.join(nltk.word_tokenize(line.lower())))
				#print sent_tok
				f1.write(sent_tok.encode('utf-8')+'\n')

				if lem_flag:
					tags = tagger.tag_text(unicode(sent_tok))
					# Lemmatization ------------------------------------------------------------
					sent_lem = ' '.join(x.split('\t')[2] for x in tags if len(x.split('\t'))==3)
					f2.write(sent_lem.encode('utf-8')+'\n')
					# PosTagging ---------------------------------------------------------------
					if pos_flag:
						sent_pos = ' '.join( [x.split('\t')[2]+"_pos:"+x.split('\t')[1] for x in tags if len(x.split('\t'))==3]) 
						f3.write(sent_pos.encode('utf-8')+'\n')
						#print "PosTagging... ok"
#--------------------------------------------------------------------------------------------------
			

#--------------------------------------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
	
	preprocessings(corpus_dir,lang,TAGDIR,lem_flag,pos_flag)			

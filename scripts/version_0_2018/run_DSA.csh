#!/bin/csh
# --------- Run the Distrubutional Bag of Words Approach (DSA) --------------------------------
# 									                                                          -
# Author  : Amir HAZEM  			                                                          -
# Created : 18/01/2019				                                                          -
# Updated :	23/01/2019				                                                          -
# source  : 												  								  -
# command : ./run_DSA.csh																	  -	
#									                                                          -
#----------------------------------------------------------------------------------------------

# Parametrs -----------------------------------------------------------------------------------
#
set task 		= $argv[1]      # sub add sup	       										 --
set limit_ngram = $argv[2]  	# limitngram to extract										 --
set flag_ngram  = $argv[3] 		# 1 2 3 4               								     --
set flag_all    = $argv[4]	    # all : us     										         --
set asso 		= $argv[5]		# fasttext w2v												 --	
set min_occ		= $argv[6]		# 300 100                                                    --
set p_k			= $argv[7]		# precision at top p_k                                       --
#																							 --
# ---------------------------------------------------------------------------------------------	

# Main ----------------------------------------------------------------------------------------
# 
#echo $model
if (  $limit_ngram > 0  ) then 
	
	#1) preprocessing
		
	python run_build_ngrams_context_vectors.py $limit_ngram $flag_ngram $flag_all 

	
	
else
	echo "Error please check the parameters"
endif 


# ----------------------------------------------------------------------------------------------
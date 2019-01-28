#!/bin/csh
# --------- Run the Distrubutional Bag of Words Approach (DSA) -------------------------------------------------
# 									                                                            			   -
# Author  : Amir HAZEM  			                                                          				   -
# Created : 18/01/2019				                                                              		       -
# Updated :	28/01/2019				                                                				           -
# source  : https://github.com/hazemAmir/DSA 												  				   -
# command : ./run_DSA.csh breast_cancer en fr mi cos 3 5 True en_fr_breast_cancer_248.csv dicfrenelda-utf8.txt -	
#									                                                                           -
#---------------------------------------------------------------------------------------------------------------

# Get the number of parameters
set len_param = $#argv

if (  $len_param == 10  ) then

	# Parametrs -----------------------------------------------------------------------------------
	#
	set corpus 		= $argv[1]		# Bilingual corpus name (breast_cancer / wind_energy / ...)
	set source_lang = $argv[2]		# Source language       (En / Fr / Es)
	set target_lang = $argv[3]		# Target language		(En / Fr / Es)
	set asso 		= $argv[4]		# Association measure   (MI / ODDS / LL)
	set simil 		= $argv[5]		# Similarity measure    (Cos / Jac)
	set w 			= $argv[6]		# Context window size   (1 / 2 / 3 / 4...) 
	set min_occ		= $argv[7]      # Hapax filtering       (default 5) 
	set filter		= $argv[8]      # Stopwords filtering   (Bool: True or False)
	set termlist    = $argv[9]		# Evaluation list name  (en_fr_breast_cancer_248.csv)
	set dico_name	= $argv[10]		# Dictionary name 		(dicfrenelda-utf8.txt)
	# 
	#																							 --
	# ---------------------------------------------------------------------------------------------	



	# Main ----------------------------------------------------------------------------------------
	# 
		

	echo "------------------- Distributional Standard Approach (DSA) -------------------"

	# 1) Context vectors: 
	# -------------------
	# Build source and target context vectors --------------------	 	
	
	python run_context_vectors.py -c $corpus -l $source_lang -ct postag -f  $filter -w $w 

	python run_context_vectors.py -c $corpus -l $target_lang -ct postag -f  $filter -w $w

	# 2) Association measure:
	# -----------------------
	# Compute source and target words association measure
	
	python run_association_measures.py -c $corpus -l $source_lang -s $source_lang -t $target_lang -a $asso -w $w -min $min_occ -eval $termlist

	python run_association_measures.py -c $corpus -l $target_lang -s $source_lang -t $target_lang -a $asso -w $w -min $min_occ -eval $termlist

	# 3) Translate vectors:
	# --------------------
	# Translation of evaluation source vectors to the target languge using a bilingual dictionary

	python run_translate_context_vectors.py -c $corpus -s $source_lang -t $target_lang -a $asso -w $w -min $min_occ -eval $termlist -dico $dico_name

	# 4) Similarity measure:
	# ----------------------
	# Compute similarity between translated source and target vectors

	python run_compute_similarity.py -c $corpus -s $source_lang -t $target_lang -a $asso -sim $simil -w $w -min $min_occ -eval $termlist -dico $dico_name
	
else
	echo "Error please check the parameters"
endif 


# ----------------------------------------------------------------------------------------------
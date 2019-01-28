# -*- coding: utf-8 -*-
# --------------------------------------------------------------------
# DSA: Unzip wikipedia files 							 			 -
#                                                                    -
# Author  : Amir HAZEM                                               -     
# Created : 15/01/2018                                               -  
# Updated : 15/01/2018                                               -
#                                                                    -  
# --------------------------------------------------------------------



from __future__ import division
import argparse
import numpy as np
import sys
import unicodecsv
import math
import os
import io
import re
import encodings
import nltk
from nltk.corpus import wordnet as wn
from operator import add



import os.path 
  
path_in="/home/hazem-a/Bureau/LS2N_2018/data/wikipedia_test/raw/"

path_out="/home/hazem-a/Bureau/LS2N_2018/data/wikipedia_test/unzip/"

def unzip(path_in,path_out):  
    fichier=[]  
    sep=" "	
    id_=0
    for root, dirs, files in os.walk(path_in):  
        for i in files:  
            #fichier.append(os.path.join(root, i))  
            id_+=1
            print root
            print i
            
            file_in=root+"/"+i

            ch=root.split("/")

            file_out=path_out+i+"."+ch[len(ch)-1]+".txt"
            print file_in
            print file_out
            
            
            cmd="find "+ root + " -name " + "'*bz2' " + "-exec bunzip2 -c {} \; > " +  file_out

            print cmd
            os.system(cmd)

            #sys.exit()
    






#----------------------------------------- 
# Main 
#-----------------------------------------

if __name__=='__main__':


	unzip(path_in,path_out)
    
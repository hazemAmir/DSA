# DSA
Distributional Standard Approach (DSA) for bilingual lexicon extraction from specialized comparable corpora.
The distributional standard approach is the earliest state of the art approaches used in the task of  bilingual terminology extraction from specialized comparable corpora. It is based on word context vectors representation and allows the alignment of pairs of words thanks to vectors similarity.  

DSA consists in the following steps:

0) Data preprocessing (Tokenization, Part of Speech Tagging and Lemmatization)
1) Context vectors construction 
2) Association measure computation 
3) Source context vectors translation
4) Context vectors similarity computation  
5) Evaluation of bilingual terminology extraction (MAP score)

## Requirements

- Python 2.7 
- NumPy
- SciPy
- treetaggerwrapper  https://treetaggerwrapper.readthedocs.io/en/latest/ (for text pre-processing)

## Quick start (Reproducing Results)

```
1. ./run_DSA.csh corpus source_lang target_lang asso simil w min_occ filter termlist dico_name

example:
./run_DSA.csh breast_cancer en fr mi cos 3 5 True en_fr_breast_cancer_248.csv dicfrenelda-utf8.txt

The default parameters are:

- corpus      : Bilingual corpus name (breast_cancer / wind_energy / ...)
- source_lang : Source language       (En / Fr / Es)
- target_lang : Target language		    (En / Fr / Es)
- asso        : Association measure   (MI / ODDS / LL)
- simil       : Similarity measure    (Cos / Jac)
- w           : Context window size   (1 / 2 / 3 / 4...) 
- min_occ     : Hapax filtering       (default 5) 
- filter      : Stopwords filtering   (Bool: True or False)
- termlist    : Evaluation list name  (en_fr_breast_cancer_248.csv)
- dico_name   : Dictionary name 		(dicfrenelda-utf8.txt)
```

## Bilingual corpora
- The bilingual comparable corpus consists in a set of raw text files. 
- It should be under the directory  **DSA/data/train/corpora/corpus_name/raw/source_language** and **DSA/data/train/corpora/corpus_name/raw/target_language**
- After pre-processing, tok, lem and postag directories are created. These directories correspond to tokenized, lemmatized and potagged files.
- By default, the corpora are placed under the root (DSA) directory as follows: 
  **DSA/data/train/corpora/corpus_name/raw/source_language** and 
  **DSA/data/train/corpora/corpus_name/raw/target_language**
- If you want to specify another path for your corpus, it should respect the following path format: 
  **your_corpus_path/corpus_name/raw/source_language** and  **your_corpus_path/corpus_name/raw/target_language**
  
## Bilingual Dictionary
- The bilingual dictionary is a csv text file with ';' separator 
- It is located in **DSA/data/train/dictionaries/dico_name**
- It follows the **ELRA** format **source_word**;**postag**;**TR**-**source_language-target_language**;**target_word**;**postag**;
- Example:  blessure;S;TR-FR-EN;injury;S

## Evaluation termlist
- The evaluation termlist is a csv file with a 'TAB' separator
- It consists in pairs of source termes and their corresponding translations
- It is located in **DSA/data/train/termlists/**
- Example: DSA/data/train/termlists/en_fr_breast_cancer_248.csv  

## Usage (DSA step by step)
Using DSA involves the following steps: 

## 0) Pre-processing
python run_preprocessing.py -c corpus -l lang --lem --postag --default_inout 

```
- corpus	: Corpus name (-c corpus)
- lang		: Corpus language (-l language) 
- lem		: Corpus lemmatization (--lem)
- postag	: Corpus postagging (--postag)
- default_inout	: Default corpus directory (--default_inout), if specified default directories are used to load the corpus.
		  the corpus should be in the directory DSA/data/train/corpus_name/raw/source_language and
		  DSA/data/train/corpus_name/raw/target_language	
- input		: Raw Corpus input directory, if default_inout is not used  (--input corpus path )
- output	: Pre-processed corpus output directory, if default_inout is not used  (--output corpus path)

--> Breast cancer pre-processing example:

python run_preprocessing.py -c breast_cancer -l en --lem --postag --default_inout

python run_preprocessing.py -c breast_cancer -l fr --lem --postag --default_inout

```


## 1) Context vectors

### Build source and target context vectors 
```
	python run_context_vectors.py -c $corpus -l $source_lang -ct postag -f  $filter -w $w 

	python run_context_vectors.py -c $corpus -l $target_lang -ct postag -f  $filter -w $w
```
## 2) Association measure
### Compute source and target words association measure
```	
python run_association_measures.py -c $corpus -l $source_lang -s $source_lang -t $target_lang -a $asso -w $w -min $min_occ -eval $termlist

python run_association_measures.py -c $corpus -l $target_lang -s $source_lang -t $target_lang -a $asso -w $w -min $min_occ -eval $termlist
```
## 3) Translate vectors
   --------------------
### Translation of evaluation source vectors to the target languge using a bilingual dictionary
```
python run_translate_context_vectors.py -c $corpus -s $source_lang -t $target_lang -a $asso -w $w -min $min_occ -eval $termlist -dico $dico_name
```
## 4) Similarity measure

### Compute similarity between translated source and target vectors
```
python run_compute_similarity.py -c $corpus -s $source_lang -t $target_lang -a $asso -sim $simil -w $w -min $min_occ -eval $termlist -dico $dico_name
```

## To do
- Include other languages  (spanish coming soon)
- Include new dictionaries 
- Include new corpora
- Include corpus comparability measures


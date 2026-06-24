# Thesis
Code for thesis project for the master Language &amp; AI

This repository contains all the code for the Master thesis titled "Syntax versus Semantics: Adapting the Microportrait Pipeline with Semantic Role Labeling". The following files are included:

**microportraits.py** - 
This script is taken from the following github: https://github.com/cltl/micro-portraits/tree/master/microportraits 
The script has been updated to remove old functions that are no longer in use and to adapt the logic to be able to retrieve SRL information from the NAF files, if present. This has been done by Antske Fokkens.

**MP_glob.py** - 
Short script that allows you to apply the microportraits script over a folder.

**naf2conll.py** - 
This script turns the NAF files that are needed for the microportraits pipeline into .txt files of the CoNLL format. 

**XLM_R.py** - 
This is a util script that contains all functions to make SRL predictions using the finetuned XLM_RoBERTa model.

**SRL_applied.ipynb** - 
This notebook makes use of the XLM_R.py util script to make predictions. The notebook contains a link to the fintuned XLM-R model.

**conll2naf.py** - 
This script turns the conll files with SRL predictions into a new layer in the NAF files. This way, the microportraits.py script can retrieve the SRL information directly from a layer in the NAF file.

**srl_data.py** - 
This is a helper script for conll2naf.py to create the SRL classes and layer in the NAF file. This file is directly taken from the following github page: https://github.com/cltl/KafNafParserPy/tree/master/KafNafParserPy 
The script is included for completeness.

**PMI.py** -
This script calculates the PMI scores for the qualitative evaluation that has been done in the thesis project. This script includes all descriptions within a nanoportrait.

**PMI_labelonly.py** - 
This script also calculates the PMI scores, but only for the label that has been defined as a target term. Any other labels that are included within that specific portrait are not included. See Chapter 4 results of the thesis for a discussion on why.

**requirements.txt** - 
All modules needed to run the code above.


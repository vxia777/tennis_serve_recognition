# tennis_serve_recognition

This repository contains code for doing tennis serve classification based on video input data and using CNN/RNN architectures. 
Specific files of interest are:

data -- folder containing raw video data as well as generated sequence features for model training/analyses, only a subset uploaded on here  
model -- folder detailing relevant RNN LSTM model used as well as parameters used  
dataprep_split.py -- script to generate balanced randomized train/dev/test sets.  
data_utils.py -- utility functions for manipulating raw video data as well as abstraction of dataset class used in model training/evaluations  
inceptionv3_featureextract.py -- generation of sequenced frame features using InceptionV3  
train_eval_script.ipynb -- script for LSTM model training and evaluation of output for the sequenced CNN-generated features  

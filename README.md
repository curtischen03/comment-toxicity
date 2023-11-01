This is a machine learning project that uses neural networks to determine the toxicity of a comment, which is put into these categories: toxic, severly toxic, obscene, threat, insult, and identity hate. A 1 indicates that it is part of that category.

Datasets used: 
Kaggle - Toxic Comment Classification Challenge\
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Packages used:\
tensorflow\
numpy\
pandas

Steps to replicate: 
1. Download the Kaggle dataset in the link and extract the csv file.
2. Run the runModel.py script. This file creates the neural network, trains the neural network, and saves the model as 'model.h5.'

Testing your own data:
1. Edit the predict.py script and add function calls for score_comment.
```
score_comment('I hate you')
```
3. Run the predict.py script.

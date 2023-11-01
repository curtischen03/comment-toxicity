import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import TextVectorization
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Bidirectional,Dense,Embedding
from keras.models import load_model
import gradio as gr

df = pd.read_csv("train.csv")
#print(df.head())
#print(df.iloc[0]['comment_text'])
#print(df[df.columns[2:]].iloc[0])

#tokenization
X = df['comment_text']
Y = df[df.columns[2:]].values
MAX_WORDS = 200000 
vectorizer = TextVectorization(max_tokens=MAX_WORDS, #number of words in vocab
                                               output_sequence_length=1800, #sentence length
                                               output_mode='int') #map words to ints

vectorizer.adapt(X.values)
#print(vectorizer("Hello my name is here")[:5])


currModel = load_model("model.h5")


#Evaluate
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = list(currModel.predict(vectorized_comment).flatten())
    classes = list(df.columns[2:])
    text = ''
    #print(classes)
    #print(results)
    for i in range(len(results)):
        text = text + " " + classes[i] + ": " + str(results[i] > 0.5) + '\n'
    return text

#test whatever comment you want here:
print(score_comment('I hate you'))
interface = gr.Interface(fn=score_comment, inputs="text",outputs='text')
interface.launch(share=True)
    



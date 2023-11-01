import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import TextVectorization
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Bidirectional,Dense,Embedding
from keras.models import load_model
print(tf.config.list_physical_devices('GPU'))
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
vectorized_text = vectorizer(X.values)
#pipeline
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text,Y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) #helps bottlenecks

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))
model = Sequential()
model.add(Embedding(MAX_WORDS+1,32))
model.add(Bidirectional(LSTM(32,activation='tanh')))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='sigmoid'))
model.compile(loss='BinaryCrossentropy',optimizer='Adam',metrics=["accuracy"])
model.summary()
history=model.fit(train,epochs=1,validation_data=val)
results =model.evaluate(test)
model.save("model.h5")

model.compile(loss='BinaryCrossentropy',optimizer='Adam',metrics=["accuracy"])
#currModel.evaluate(test)

input_text = vectorizer('You freaking suck!')
res = model.predict(np.array([input_text]))
#print(df.columns[2:])
#print(res)

batch = test.as_numpy_iterator().next()
batch_X,batch_Y=test.as_numpy_iterator().next()
#print((currModel.predict(batch_X)>0.5).astype(int))



from keras.metrics import Precision,Recall,CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()
for batch in test.as_numpy_iterator():
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

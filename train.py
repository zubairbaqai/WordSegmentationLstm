from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Reshape
from keras.datasets import imdb

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from typing import Tuple, List, Dict

import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.python.tools import inspect_checkpoint as chkp

import collections
import math
import random
import re
import json
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Concatenate
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense
from keras.optimizers import SGD
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
import pickle



label_path = "../icwb2-data/training/as_Label.utf8"
Input_path = "../icwb2-data/training/as_Input.utf8"
Output_path = "../icwb2-data/training/Output.utf8"

Inputs=open(Input_path, encoding='utf8')
Labels=open(label_path, encoding='utf8')

InputText=Inputs.read()
LabelsText=Labels.read()



VOCABULARY_SIZE = 15000 #100_000


#This function thats in the complete text , result its dictionary , converted text in form of index of dictionary
def build_dataset(words_in_contex, vocab_size, dictionary=None, inverted_dictionary=None, skip_unk=False):

    if not dictionary:
        
        counter = collections.Counter()
        for contex in words_in_contex:
            for word in contex:
                counter[word] += 1
        counter_len = len(counter)
        
        if(len(counter)<vocab_size):
            vocab_size=len(counter)+1
            VOCABULARY_SIZE=vocab_size
        # I build the dictionary taking the vocab_size - 1 most common elements, and adding an UNK element for the others
        
        dictionary = {key: index for index, (key, _) in enumerate(counter.most_common(vocab_size - 1))}
        assert "UNK" not in dictionary
        dictionary["UNK"] = vocab_size - 1
        dictionary.update({k:v+1 for k, v in dictionary.items()})
        dictionary["<PAD>"] = 0
        vocab_size=len(dictionary)+1
        VOCABULARY_SIZE=vocab_size

        inverted_dictionary = {value: key for key, value in dictionary.items()}
       
    data = []
    for contex in words_in_contex:
        paragraph = []
        for i in contex:
            id_ = dictionary[i] if i in dictionary else dictionary["UNK"]
            if skip_unk and id_ == dictionary["UNK"]:
                continue
            paragraph.append(id_)
        data.append(paragraph)
    
    return data, dictionary, inverted_dictionary




#We are retrieving our data.pkl , in case we want to retrain , its not required for further predictions or scoring 
DataExist=False
try:
    model = open('data.pkl', 'r')
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    with open('reverse_dictionary.pkl', 'rb') as f:
        reverse_dictionary = pickle.load(f)

    
    
    DataExist=True

except FileNotFoundError:
    print("Pickles not found")
    pass


#We are storing our data.pkl,dictionary and reverse dict as pickle files  , in case we want to retrain , its not required for further predictions or scoring 

if(not DataExist):
    data, dictionary, reverse_dictionary = build_dataset(InputText, VOCABULARY_SIZE)
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    with open('reverse_dictionary.pkl', 'wb') as f:
        pickle.dump(reverse_dictionary, f, pickle.HIGHEST_PROTOCOL)



#Converting our Labels in form of 0,1,2,3 
LabelData = []
for contex in LabelsText:
    paragraph = []
    for i in contex:
        if(i =="B"):
            paragraph.append(0)
        if(i =="E"):
            paragraph.append(1)
        if(i =="I"):
            paragraph.append(2)
        if(i =="S"):
            paragraph.append(3)
    LabelData.append(paragraph)




#Now from our data which is in the integer form , we make the Bigrams 
bigrams=[]

for i in range(len(data)-1):
    add=[]
    add.append(data[i][0])
    add.append(data[i+1][0])
    add=np.asarray(add)

    bigrams.append(add)

bigrams.append(np.array([0,0]))
bigrams=np.asarray(bigrams)



#Now from our data which is in the integer form , we make the Unigrams 

for i in range(len(data)):
    add=[]
    add.append(data[i][0])
    add.append(0)
    add=np.asarray(add)

    data[i]=add

data=np.asarray(data)



#We are taking 100 Characters are a time , making the inputs as 100
ArraySize=100


#Converting our Labels into form of OneHotEncoding
onehotLabel = to_categorical(LabelData)
npData=np.array(data)


#We want to make Sure , that our inputs are in array size of multiple of 100 , because thats what we set as input shape for model
npData=npData[:(len(npData)-len(npData)%ArraySize)]
bigrams=bigrams[:(len(bigrams)-len(bigrams)%ArraySize)]
onehotLabel=onehotLabel[:(len(onehotLabel)-len(onehotLabel)%ArraySize)]


#now we know we may have many many lines , so we break those many lines into multiple rows , where each row will have 100 characters(bigrams, unigrams)
npData.shape=(-1,100,2)
bigrams.shape=(-1,100,2)
onehotLabel.shape=(-1,100,4)


#We are now splitting the training and test set .
unitrainX, unitestX, y_train, y_test = train_test_split(npData, onehotLabel, test_size=0.02,shuffle=False)
bitrainX, bitestX, y_train, y_test = train_test_split(bigrams, onehotLabel, test_size=0.02,shuffle=False)


#if model is saved before no need to remake it
modelExist=False
try:
    model = open('my_model.h5', 'r')
    model = load_model('my_model.h5')
    modelExist=True

except FileNotFoundError:
    print("came here")
    pass

#if model is not stored we generate the model
if(not modelExist):

    #As discussed earlier we will have 2 inputs , Bigrams and unigrams of input shape of 100,2 (means it must take 100 character) , (dont worry i have fixed the case if we input less than 100 characters)
    InputsUnigram = Input(shape=(100,2), name='uniInput')
    InputsBigram=  Input(shape=(100,2), name='biInput')

    #We make two embedding layers , each for unigram and bigram
    UnigramEmbed = Embedding(VOCABULARY_SIZE, 128)(InputsUnigram)
    BigramEmbed = Embedding(VOCABULARY_SIZE, 128)(InputsBigram)

    #we concatenate the 2 embeddings , we didnt mix because unigram has 0 as second input so we dont want to make the bigram embedding dependant on it
    CombinedEmbd = Concatenate(axis=2)([UnigramEmbed,BigramEmbed])

    #Now after many troubling hours , i Came to know LSTM can only take inputs of 2 dimensions , so i mix the 2nd and 3rd dimesion , which after testing dosnt impact the result , so we are good here
    CombinedEmbd= Reshape((100, 512))(CombinedEmbd)
    print(CombinedEmbd)
    
    #We used Dropout between inputs and LSTM layer, as well as dropouts in between the LSTM cells . 
    blstm = Bidirectional(LSTM(64, return_sequences=True,dropout=0.2, recurrent_dropout=0.2))(CombinedEmbd)
    output = (Dense(4, activation='softmax'))(blstm)
    model = Model(inputs=[InputsUnigram,InputsBigram], output=output)
    
    sgd = SGD(lr=0.04, decay=1e-5, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
    model.summary()
    batch_size=25
    model.fit({'uniInput': unitrainX, 'biInput': bitrainX}, y_train,
              batch_size=batch_size,
              epochs=6,
              validation_data=[{'uniInput': unitestX, 'biInput': bitestX}, y_test])


    model.save('my_model.h5')




model.summary()

    
	







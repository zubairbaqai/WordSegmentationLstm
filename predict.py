from argparse import ArgumentParser
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from typing import Tuple, List, Dict

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Uncomment this to run on Mac OS
import matplotlib.pyplot as plt

from collections import Counter
from tensorflow.python.tools import inspect_checkpoint as chkp

import collections
import math
import random
import re
import json
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense
from keras.optimizers import SGD
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
import pickle


#Global Variables 
InputLength=100
data=[]
bigrams=[]
originalLength=0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")
    return parser.parse_args()


def predict(input_path, output_path, resources_path):

    global data
    global bigrams

    #here we Open our Input and Output Files , Input file will have Chinese Unsegmented Text, and model will send BEIS labels corrosponding to each input into output file
    try:
        Inputs=open(input_path, encoding='utf8')
        file1 = open(output_path, "w")
    except FileNotFoundError:
        print("Files not found")
    

    #We load Dictionary , We Generated this Dictionary in Train.py
    try:
        dictionarycheck = open(resources_path+'/dictionary.pkl', 'r')
        with open(resources_path+'/dictionary.pkl', 'rb') as f:
            dictionary = pickle.load(f)
        
        DataExist=True
        print("Success")

    except FileNotFoundError:
        print("Dictionary not found")

    #We are loading our model , This was also Generated in Train.py
    try:
        model = open(resources_path+'/my_model.h5', 'r')
        model = load_model(resources_path+'/my_model.h5')
        modelExist=True

    except FileNotFoundError:
        print("Model Not Found")
    FileText=Inputs.read()

    #in Case if you want to text with writing an input directly 
    #FileText="主要分佈在熱帶及亞熱帶水域，"

    #This is a method which will make our inputs into Integers, generate the Unigram and bigrams ,
    preprocess(FileText,dictionary)
  

    
    data=np.array(data)
    bigrams=np.array(bigrams)

    #Converting the Shape of npArray in form of (X,100,2) for both bigrams and unigrams . 
    #NOTE- the length of unigram is 2 because to concatenate two embedding we needed to make sure the inputs are of same shape . so i added a dummy value (0) with each unigram , which wont effect working
    data.shape=(-1,100,2)
    bigrams.shape=(-1,100,2)

    #We feed unigrams and bigrams into the mode 
    result=model.predict([data,bigrams])



    #The output will be of shape (X,100,4)

    StringGen=""
    for output in result:
        output=(output.argmax(axis=-1))
        for char in range(100):

             #Inputs.append(reverse_dictionary[unitrainX[0][char][0]])
             
             
             if(output[char] == 0):
                Answer="B"
             if(output[char]  == 1):
                Answer="E"
             if(output[char]  == 2):
                Answer="I"
             if(output[char]  == 3):
                Answer="S"
             StringGen=StringGen+Answer




    #If we recall that we added dummy values in our original string to make a multiple of 100 , we also stored the actual length of who string . so we removed the unnecessary outputs due to padding
    #We get store these outputs BIES labels in the File which we were passed in the arguments
    StringGen=StringGen[0:originalLength]
    finalOutput=""
    counter=0
    for i in range(len(FileText)):
        if(FileText[i]=="\n"):
            finalOutput=finalOutput+"\n"
        else:
            
            finalOutput=finalOutput+StringGen[counter]
            counter=counter+1
    print(finalOutput)
    file1.write(finalOutput)

    file1.close()
    


def preprocess(FileText,dictionary):

##////////////////Converting Text to Dictionary /////////////////////##

    global data
    global InputLength
    global bigrams
    global originalLength
    FileText=FileText.replace("\n","")
    
    originalLength=len(FileText)
    
    
    print(data)    

    
    for contex in FileText:
        paragraph = []
        for i in contex:
            id_ = dictionary[i] if i in dictionary else dictionary["UNK"]
                # if skip_unk and id_ == dictionary["UNK"]:
                #     continue
            paragraph.append(id_)
        data.append(paragraph)


    #Since our model takes input of 100 length , we need to make sure we add padding at end , 
    while(len(data)%InputLength is not 0):
         data.append([0])


 #########We have created words to dict dictionary so far ####

 ### now we create the bigrams and unigrams , "data" are out unigrams , and bigrams have bigrams of that character

    


    for i in range(len(data)-1):
        add=[]
        add.append(data[i][0])
        add.append(data[i+1][0])
        add=np.asarray(add)

        bigrams.append(add)

    bigrams.append(np.array([0,0]))
    bigrams=np.asarray(bigrams)




    for i in range(len(data)):
        add=[]
        add.append(data[i][0])
        add.append(0)
        add=np.asarray(add)

        data[i]=add

    data=np.asarray(data)








if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)

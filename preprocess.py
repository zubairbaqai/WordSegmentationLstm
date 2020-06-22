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


train_path = "../icwb2-data/training/as_trainconverted.utf8"
label_path = "../icwb2-data/training/as_Label.utf8"
Input_path = "../icwb2-data/training/as_Input.utf8"

fileText=open(train_path, encoding='utf8')
TAGS=[]

counter=0
Training=False

for  line in fileText:
    previousChar=None
    LineTags=[]
    line=line+" "
    for indexOfChar in range(len(line)-2):
        #Case 1 if its a single character
        #print(line[indexOfChar])
        
        if(line[indexOfChar]==" " or line[indexOfChar] == "\u3000" or line[indexOfChar] == "\n"):
            
            previousChar=None
            continue
        
        elif(previousChar==None and (line[indexOfChar+1] == " " or line[indexOfChar+1] == "\u3000" or line[indexOfChar+1] == "\n")):
            LineTags.append("S")
            continue
        #Case 2 if its a Space
        elif(line[indexOfChar]==" " or line[indexOfChar] == "\u3000" or line[indexOfChar] == "\n"):
            print(line[indexOfChar]==" ")
            previousChar=None
            continue
        #Case 3 if Beginning Character
        elif((line[indexOfChar+1]!=" " or line[indexOfChar+1] != "\u3000" or line[indexOfChar+1] != "\n") and previousChar==None):
            LineTags.append("B")
        elif(previousChar!=None and (line[indexOfChar+1]!=" " and line[indexOfChar+1] != "\u3000" and line[indexOfChar+1] != "\n")):
            LineTags.append("I")
        elif(previousChar!=None and (line[indexOfChar+1]==" " or line[indexOfChar+1] == "\u3000" or line[indexOfChar+1] == "\n")):
            LineTags.append("E")
        else:
            print("CAM HERE")
            print(line[indexOfChar])
            LineTags.append(line[indexOfChar])



        previousChar=line[indexOfChar]


    TAGS.append(LineTags)






    
    
    
    
    
    
    counter=counter+1
    
    
fileText.close()


####### STEP1 #####
file = open(label_path,"w")
outputz=""
counterz=0
for item in TAGS:
    for label in item:
        counterz=counterz+1
        outputz=outputz+label


    #comment this when training
    if(not Training):
    	outputz=outputz+"\n"




#file.write(outputz[:4000])
file.write(outputz)


file.close()



###### STEP 2 ######
print("HERE")
file = open(train_path, encoding='utf8')
text=file.read()
text=text.replace(" ", "")
text=text.replace("\u3000","")

#Use this line when training , for predicting comments this
if(Training):
	text=text.replace("\n","")


file1 = open(Input_path, "w")
#file1.write(text[:4000])
file1.write(text)

file1.close()


print("Total Number of Characters in file After removing whitespaces and newline delimiter" , len(text))
print("Total Number of Labels we have created" , counterz)



file0 = open(Input_path, encoding='utf8')
print(len(file0.read()))
file1 = open(label_path, encoding='utf8')
print(len(file1.read()))



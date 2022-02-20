from os import supports_effective_ids
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_openml
from PIL import Image
import numpy as np
import cv2
import pandas
from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle
import scipy.io

def initialiseIris():
    iris=pandas.read_csv("iris.csv")                                                                                                  
    x=iris.loc[:,"petal_length"]                                                                                                                                    
    y=iris.loc[:,"petal_width"]                                                                                                                                      
    data=list(zip(x,y))
    target=iris.loc[:,"species"] 
    kopt=0
    accuracy=0
    evals=[]
    for n_neighbors in range(2,11):
       clf = KNeighborsClassifier(n_neighbors=n_neighbors)
       clf.fit(data,target) 
       score = accuracy_score(clf.predict(data), target)
       evals.append({'k': n_neighbors, 'accuracy': score})
       if score>accuracy:
         accuracy=score
         kopt=n_neighbors
         model=clf
    return model,x,y,target,kopt,accuracy,evals

def initialiseDigit():
    model=pickle.load(open('knnpickle_fileFinal2', 'rb'))
    data = pd.read_csv("optimalFinal2.csv")
    data2 = pd.read_csv("evalsFinal2.csv")
    # converting column data to list
    kopt = data['kopt'].tolist()[0]
    accuracyopt = data['accuracy'].tolist()[0]
    k=data2['k'].tolist()
    accuracy=data2['accuracy'].tolist()
    evals=[]
    for i in range(0,len(k)):
        evals.append({'k': k[i], 'accuracy': accuracy[i]})
    return model,kopt,accuracyopt,evals

def convertImageToPixels(imgC):
  img = Image.open(imgC)
  img_array = np.asarray(img)
  resized = cv2.resize(img_array, (28, 28 ))
  gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #(28, 28)
  image = cv2.bitwise_not(gray_scale)
  plt.imshow(image, cmap=plt.get_cmap('gray'))
  image = image.reshape(1, 784)
  return image


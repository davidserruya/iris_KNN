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



def initialiseIris(choix):
    iris=pandas.read_csv("iris.csv")                                                                                                  
    x=iris.loc[:,"petal_length"]                                                                                                                                    
    y=iris.loc[:,"petal_width"]                                                                                                                                      
    data=list(zip(x,y))
    target=iris.loc[:,"species"]
    if choix=="iris interface":
      return data,target,x,y
    return data,target

def initialiseDigit():
    model=pickle.load(open('knnpickle_file', 'rb'))
    data = pd.read_csv("test.csv")
    # converting column data to list
    k = data['k'].tolist()
    errors = data['errors'].tolist()
    kopt = data['kopt'].tolist()
    return model,kopt,k,errors

def findErrorsK(xtrain,ytrain,xtest,ytest):
   data = {}
   for k in range(2,15):
     knn = KNeighborsClassifier(k)
     data[k]=(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
   k = list(data.keys())
   errors = list(data.values())
   return k,errors

def figurek(k,errors):
  fig, ax = plt.subplots(figsize=(15, 5))
  ax.plot(k, errors, 'o-')
  ax.set_xlim(1, 15)
  ax.set_xlabel("K")
  ax.set_ylabel("Nombred'erreurs")
  plt.title("Taux d'erreurs pour les différents classifieurs K")
  return fig


def splitDataset(data,target,size):
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=size)
    return xtrain,ytrain,xtest,ytest

# Fonction qui permet de trouver le K optimal
def findkOpt (data,target):
  kopt=0
  accuracy=0
  for n_neighbors in range(1,12):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(data,target) 
    score = accuracy_score(clf.predict(X), y)
    if score>accuracy:
        kopt=n_neighbors
  return kopt


# Fonction qui permet de définir d'après l'algorithme des k plus proches voisins 
# l'espèce de l'iris demandé selon sa longueur et sa largeur de taille de pétale
def findPrediction(data,target,size):
  xtrain,ytrain,xtest,ytest=splitDataset(data,target,size)
  minerror,kopt=findkOpt(xtrain,ytrain,xtest,ytest)
  model = KNeighborsClassifier(n_neighbors=kopt)
  return model.fit(data,target), kopt

def convertImageToPixels(imgC):
  img = Image.open(imgC)
  img_array = np.asarray(img)
  resized = cv2.resize(img_array, (28, 28 ))
  gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #(28, 28)
  image = cv2.bitwise_not(gray_scale)
  plt.imshow(image, cmap=plt.get_cmap('gray'))
  image = image.reshape(1, 784)
  return image


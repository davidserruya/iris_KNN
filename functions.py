from os import supports_effective_ids
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_openml
from PIL import Image
import numpy as np
import cv2
import pandas

def initialiseIris():
    iris=pandas.read_csv("iris.csv")                                                                                                  
    x=iris.loc[:,"petal_length"]                                                                                                                                    
    y=iris.loc[:,"petal_width"]                                                                                                                                      
    data=list(zip(x,y))
    target=iris.loc[:,"species"]
    return data,target

def initialiseDigit():
    mnist = fetch_openml('mnist_784', version=1)
    sample = np.random.randint(70000, size=5000)
    data = mnist.data.values[sample]
    target = mnist.target.values[sample]
    return data,target

def splitDataset(data,target,size):
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=size)
    return xtrain,xtest,ytrain,ytest

# Fonction qui permet de trouver le K optimal
def findk (values,species):
  klist=[]
  # tester K alant de 2 à 15
  for i in range (2,15):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(values,species)
    error = 1 - model.score(values,species)
    klist.append(error)
  min1=min(klist)
  k=klist.index(min1)+2
  return k

# Fonction qui permet de définir d'après l'algorithme des k plus proches voisins 
# l'espèce de l'iris demandé selon sa longueur et sa largeur de taille de pétale
def findPrediction(data,target):
  model = KNeighborsClassifier(n_neighbors=findk(data,target))
  return model.fit(data,target)

def convertImageToPixels(imgC):
  img = Image.open(imgC)
  img_array = np.asarray(img)
  resized = cv2.resize(img_array, (28, 28 ))
  gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #(28, 28)
  image = cv2.bitwise_not(gray_scale)
  plt.imshow(image, cmap=plt.get_cmap('gray'))
  image = image.reshape(1, 784)
  return image
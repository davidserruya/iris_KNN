from os import supports_effective_ids
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

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
def findPrediction(x,y,species,longueur,largeur):
  values=list(zip(x,y))
  model = KNeighborsClassifier(n_neighbors=findk(values,species))
  model.fit(values,species)
  return model.predict([[longueur,largeur]])

def convertImageToPixels(imgC):
  img = Image.open(imgC)
  img_array = np.asarray(img)
  resized = cv2.resize(img_array, (28, 28 ))
  gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #(28, 28)
  image = cv2.bitwise_not(gray_scale)
  plt.imshow(image, cmap=plt.get_cmap('gray'))
  image = image.reshape(1, 784)
  return image
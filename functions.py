from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def findk (d,lab):
  klist=[]
  for i in range (1,15):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(d,lab)
    error = 1 - model.score(d, lab)
    klist.append(error)
  min1=min(klist)
  k=klist.index(min1)+1
  return k

def findPrediction(x,y,lab,longueur,largeur):
  d=list(zip(x,y))
  model = KNeighborsClassifier(n_neighbors=findk(d,lab))
  model.fit(d,lab)
  return model.predict([[longueur,largeur]])
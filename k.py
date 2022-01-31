from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import ImageFilter
from functions import splitDataset, initialiseIris, initialiseDigit

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>K MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Quelle ')
genre = st.sidebar.radio(
     "Choisissez le cas",
     ('Iris_KNN','Digit_KNN'))
# Fin affichage barre latérale

st.markdown("<h1 style='text-align: center; color: red;'>K INTERFACE</h1>", unsafe_allow_html=True)

st.markdown(
"""
Pour trouver le k optimal, on va simplement tester le modèle pour tous les k de 2 à 15, mesurer l’erreur test et afficher la performance en fonction de k :
    """)

if genre == 'Iris_KNN':
    functionInitialise='initialiseIris()'
    size=0.66
    
else:   
    functionInitialise='initialiseDigit()'
    size=0.8
    
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def initialise2():
  data2,target=eval(functionInitialise)
  xtrain,xtest,ytrain,ytest=splitDataset(data2,target,size)
  return xtrain,xtest,ytrain,ytest
xtrain,xtest,ytrain,ytest=initialise2()
data = {}
for k in range(2,15):
  knn = KNeighborsClassifier(k)
  data[k]=(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
Courses = list(data.keys())
values = list(data.values())
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(Courses, values, 'o-')
ax.set_xlim(1, 15)
ax.set_xlabel("x")
ax.set_ylabel("P(x)")
st.pyplot(fig)
st.markdown("""
      Comme on peut le voir, le k-NN le plus performant est celui pour lequel k = 4.  
      On connaît donc notre classifieur final optimal : 4-nn. Ce qui veut dire que c'est celui qui classifie le mieux les données, et qui donc dans ce cas précis reconnaît au mieux les nombres écrits à la main.
    """)


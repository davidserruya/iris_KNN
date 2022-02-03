from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import ImageFilter
from functions import splitDataset, initialiseIris, initialiseDigit, findErrorsK, figurek, findkOpt

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>K MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Déterminez la répartition entre le training et le testing set')
size = st.sidebar.number_input('Entrez le pourcentage du training set',min_value=0.0,max_value=1.0, value=0.8)
st.sidebar.title('Quelle est le K optimal du modèle ?')
genre = st.sidebar.radio(
     "Choisissez un des deux problèmes : ",
     ('Iris_KNN','Digit_KNN'))
# Fin affichage barre latérale

st.markdown("<h1 style='text-align: center; color: red;'>K INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
Pour trouver le k optimal :
- On va simplement tester le modèle pour tous les k de 2 à 15
- Pour chaque k, on mesure le taux d'erreurs
- On choisit le k pour lequel le taux d'erreurs est le plus faible
    """)

if genre == 'Iris_KNN':
    functionInitialise='initialiseIris("k interface")'
    texte="la classe des iris."
    
else:   
    functionInitialise='initialiseDigit()'
    texte="les chiffres écrits à la main."
    

@st.cache(allow_output_mutation=True)
def initialise():
  data2,target=eval(functionInitialise)
  xtrain,xtest,ytrain,ytest=splitDataset(data2,target,size)
  return xtrain,xtest,ytrain,ytest
xtrain,ytrain,xtest,ytest=initialise()
k,errors=findErrorsK(xtrain,ytrain,xtest,ytest)


st.pyplot(figurek(k,errors))
minerror,kopt=findkOpt (xtrain,ytrain,xtest,ytest)
st.write("Comme on peut le voir, le k-NN le plus performant est celui pour lequel k = ",kopt," avec ",round(minerror,2)," erreurs. On connaît donc notre classifieur final optimal : ",kopt,"-nn. Ce qui veut dire que c'est celui qui classifie le mieux les données, et qui donc dans ce cas précis reconnaît au mieux ",texte) 





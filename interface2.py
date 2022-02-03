from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import ImageFilter
from functions import  *
import cv2

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>DIGIT MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Déterminez la répartition entre le training et le testing set')
size = st.sidebar.number_input('Entrez le pourcentage du training set',min_value=0.0,max_value=1.0, value=0.8)
st.sidebar.title('Déterminez le chiffre présent sur votre image : ')
uploaded_files = st.sidebar.file_uploader("Déposez une image au format PNG/JPG/JPEG",type=["png","jpg","jpeg"])
# Fin affichage barre latérale

@st.cache(suppress_st_warning=True)
def initialise(): 
 data,target=initialiseDigit()
 model,kopt=findPrediction(data,target,size)
 return model, kopt

model,kopt=initialise()


# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>DIGIT INTERFACE</h1>", unsafe_allow_html=True)
image = Image.open('frise.png')
col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
col2.image([image])

st.markdown("""
Le jeu de données utilisé sur cette interface est un dataset très célèbre, appelé MNIST.   
Il est constitué d'un ensemble de 70000 images 28x28 pixels en noir et blanc annotées du chiffre correspondant (entre 0 et 9). """) 
st.markdown("""
L'objectif de ce jeu de données était de permettre à un ordinateur d'apprendre à reconnaître des nombres manuscrits automatiquement (pour lire des chèques par exemple).  
Votre image sera donc convertie en 28x28 pixels avant d'être testée par l'algorithme.""")

if uploaded_files is not None:
 image = convertImageToPixels(uploaded_files)
 predicted= model.predict(image)
 resultat=predicted[0]
 st.write("L'image que vous avez choisi est la suivante : ")
 if uploaded_files is not None:
    col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
    col2.image(uploaded_files, use_column_width=True)
 st.write("D'après l'algorithme des k plus proches voisins, où ",kopt," est le K optimal, votre image représente un : ")

 if resultat=='0':
    imageResult = Image.open('digit/0.png')
 elif resultat=='1':
   imageResult = Image.open('digit/1.png')
 elif resultat=='2':
   imageResult = Image.open('digit/2.png')
 elif resultat=='3':
   imageResult = Image.open('digit/3.png')
 elif resultat=='4':
   imageResult = Image.open('digit/4.png')
 elif resultat=='5':
   imageResult = Image.open('digit/5.png')
 elif resultat=='6':
   imageResult = Image.open('digit/6.png')
 elif resultat=='7':
   imageResult = Image.open('digit/7.png')
 elif resultat=='8':
   imageResult = Image.open('digit/8.png')
 else:
   imageResult = Image.open('digit/9.png')

 col1, col2, col3 = st.columns([0.2, 0.1, 0.4])
 col3.image([imageResult])







 





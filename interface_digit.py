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

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>DIGIT MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Déterminez le chiffre présent sur votre image : ')
uploaded_files = st.sidebar.file_uploader("Déposez une image au format PNG/JPG/JPEG",type=["png","jpg","jpeg"])
img_file_buffer = st.sidebar.camera_input("Ou prenez une photo")
# Fin affichage barre latérale

@st.cache(suppress_st_warning=True)
def initialise(): 
 return initialiseDigit()

model,kopt,k,error=initialise()


# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>DIGIT INTERFACE</h1>", unsafe_allow_html=True)
image = Image.open('digit_images/frise.png')
col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
col2.image([image])

st.markdown("""
Le jeu de données utilisé sur cette interface est un dataset très célèbre, appelé MNIST.   
Il est constitué d'un ensemble de 70000 images 28x28 pixels en noir et blanc annotées du chiffre correspondant (entre 0 et 9). """) 
st.markdown("""
L'objectif de ce jeu de données était de permettre à un ordinateur d'apprendre à reconnaître des nombres manuscrits automatiquement (pour lire des chèques par exemple).  
Votre image sera donc convertie en 28x28 pixels avant d'être testée par l'algorithme.""")

if uploaded_files is not None or img_file_buffer is not None:
 if uploaded_files is not None:
   image = convertImageToPixels(uploaded_files)
   imageCol=uploaded_files
 else:
   image = convertImageToPixels(img_file_buffer)  
   imageCol=img_file_buffer
 predicted= model.predict(image)
 resultat=predicted[0]
 st.write("L'image que vous avez choisi est la suivante : ")
 if uploaded_files is not None or img_file_buffer is not None:
    col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
    col2.image(imageCol, use_column_width=True)
 st.write("D'après l'algorithme des k plus proches voisins, où ",kopt," est le K optimal, votre image représente un : ")

 if resultat=='0':
    imageResult = Image.open('digit_images/0.png')
 elif resultat=='1':
   imageResult = Image.open('digit_images/1.png')
 elif resultat=='2':
   imageResult = Image.open('digit_images/2.png')
 elif resultat=='3':
   imageResult = Image.open('digit_images/3.png')
 elif resultat=='4':
   imageResult = Image.open('digit_images/4.png')
 elif resultat=='5':
   imageResult = Image.open('digit_images/5.png')
 elif resultat=='6':
   imageResult = Image.open('digit_images/6.png')
 elif resultat=='7':
   imageResult = Image.open('digit_images/7.png')
 elif resultat=='8':
   imageResult = Image.open('digit_images/8.png')
 else:
   imageResult = Image.open('digit_images/9.png')

 col1, col2, col3 = st.columns([0.2, 0.1, 0.4])
 col3.image([imageResult])







 





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
from functions import findk,findPrediction, convertImageToPixels
import cv2

mnist = fetch_openml('mnist_784', version=1)
sample = np.random.randint(70000, size=5000)
data = mnist.data.values[sample]
target = mnist.target.values[sample]
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(data, target)

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>DIGIT MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Déterminer le chiffre présent sur votre image : ')
uploaded_files = st.sidebar.file_uploader("Déposez une image au format png",type=["png","jpg","jpeg"])
# Fin affichage barre latérale

# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>DIGIT INTERFACE</h1>", unsafe_allow_html=True)

if uploaded_files is not None:
 image = convertImageToPixels(uploaded_files)
 predicted= knn.predict(image)
 resultat=predicted[0]
 st.write("L'image que vous avez choisi est la suivante : ")
 if uploaded_files is not None:
    col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
    col2.image(uploaded_files, use_column_width=True)
 st.write("D'après l'algorithme des k plus proches voisins, votre image représente un : ", resultat)







 





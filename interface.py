from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from functions import findk,findPrediction


# Traitement CSV
iris=pandas.read_csv("/Users/david/Desktop/iris_KNN/iris.csv")                                                                                                  
x=iris.loc[:,"petal_length"]                                                                                                                                    
y=iris.loc[:,"petal_width"]                                                                                                                                     
species=iris.loc[:,"species"]                                                                                                                                       
# Fin traitement CSV

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>IRIS MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Déterminer l\'espèce de votre iris : ')
longueur = st.sidebar.slider('Entrez la longueur en cm', 0.0, 10.0, 0.1 )
largeur= st.sidebar.slider('Entrez la largueur en cm', 0.0, 10.0, 0.1  )
# Fin affichage barre latérale

# Appel fonction de l'algorithme des K plus proches voisins
prediction= findPrediction(x,y,species,longueur,largeur)
# Fin appel

# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>IRIS INTERFACE</h1>", unsafe_allow_html=True)
"""
En 1936, Edgar Anderson a collecté des données sur 3 espèces d\'iris : l\'iris setosa, l\'iris virginica et l\'iris versicolor.
"""
image = Image.open('/Users/david/Desktop/iris_KNN/iris_setosa.jpeg')
image2 = Image.open('/Users/david/Desktop/iris_KNN/iris_versicolor.jpeg')
image3 = Image.open('/Users/david/Desktop/iris_KNN/iris_virginica.jpeg')
st.image([image,image2,image3],caption=["SETOSA","VERSICOLOR","VIRGINICA"])

"""
Pour chaque iris étudié, Anderson a mesuré (en cm) :
* la largeur des sépales
* la longueur des sépales
* la largeur des pétales
* la longueur des pétales
"""
st.write('Par souci de simplification, nous nous intéresserons uniquement à la largeur et à la longueur des pétales.')


# Affichage graphique des K plus proches voisins
plt.axis('equal')
plt.scatter(x[species == 0], y[species == 0], color='g', label='setosa')
plt.scatter(x[species == 1], y[species == 1], color='r', label='versicolor')
plt.scatter(x[species == 2], y[species == 2], color='b', label='virginica')
plt.scatter(longueur, largeur, color='k')
plt.legend()
plt.savefig('saved_figure.png')
#fin affichage graphique des K plus proches voisins


# Affichage résultats
st.write("Votre iris a des pétales de longueur de ", longueur, ' cm et de largeur de ',largeur,' cm.')
image = Image.open('saved_figure.png')
col1, col2, col3 = st.columns([0.2, 1, 0.2])
col2.image(image, use_column_width=True)
st.write("D'après l'algorithme des k plus proches voisins, où ", findk(list(zip(x,y)),species)," est le K optimal, votre iris est de l'espèce : ")

if prediction[0]==0:
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_setosa.jpeg')
  col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
  col2.image(image, use_column_width=True, caption='SETOSA')
elif prediction[0]==1:
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_versicolor.jpeg')
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_versicolor.jpeg')
  col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
  col2.image(image, use_column_width=True, caption='VERSICOLOR')
else:
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_virginica.jpeg')
  col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
  col2.image(image, use_column_width=True, caption='VIRGINICA')
# Fin affichage résultats

# Fin affichage page principale








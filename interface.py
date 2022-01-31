from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle
from functions import findk,findPrediction

# Traitement CSV
@st.cache(allow_output_mutation=True)
def initialise2():
 iris=pandas.read_csv("iris.csv")                                                                                                  
 x=iris.loc[:,"petal_length"]                                                                                                                                    
 y=iris.loc[:,"petal_width"]                                                                                                                                     
 species=iris.loc[:,"species"] 
 values=list(zip(x,y))
 model = KNeighborsClassifier(n_neighbors=findk(values,species)).fit(values,species)
 return values,species,x,y,model
values,species,x,y,model=initialise2()                                                                                                                                    
# Fin traitement CSV

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>IRIS MENU</h1>", unsafe_allow_html=True)
st.sidebar.title('Déterminer l\'espèce de votre iris : ')
longueur = st.sidebar.slider('Entrez la longueur en cm', 0.0, 10.0 )
largeur= st.sidebar.slider('Entrez la largueur en cm', 0.0, 10.0)
# Fin affichage barre latérale



# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>IRIS INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En 1936, Edgar Anderson a collecté des données sur 3 espèces d\'iris : l\'iris setosa, l\'iris virginica et l\'iris versicolor.
""")
image = Image.open('iris_setosa.jpeg')
image2 = Image.open('iris_versicolor.jpeg')
image3 = Image.open('iris_virginica.jpeg')
col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
col2.image([image,image2,image3],caption=["SETOSA","VERSICOLOR","VIRGINICA"])

st.markdown(
"""
Pour chaque iris étudié, Anderson a mesuré (en cm) :
* la largeur des sépales
* la longueur des sépales
* la largeur des pétales
* la longueur des pétales
""")
st.write('Par souci de simplification, nous nous intéresserons uniquement à la largeur et à la longueur des pétales.')

if (longueur!=0 and largeur!=0):
 # Appel fonction de l'algorithme des K plus proches voisins
 prediction= model.predict([[longueur,largeur]])
 # Fin appel

 # Affichage graphique des K plus proches voisins
 fig, ax = plt.subplots(figsize=(10, 3))
 plt.axis('equal')
 ax.scatter(x[species == 0], y[species == 0], color='g', label='setosa')
 ax.scatter(x[species == 1], y[species == 1], color='r', label='versicolor')
 ax.scatter(x[species == 2], y[species == 2], color='b', label='virginica')
 ax.scatter(longueur, largeur, color='k')
 plt.legend()
 st.pyplot(fig)
 #fin affichage graphique des K plus proches voisins


 # Affichage résultats
 st.write("Votre iris a des pétales de longueur de ", longueur, ' cm et de largeur de ',largeur,' cm.')
 st.write("D'après l'algorithme des k plus proches voisins, où ", findk(list(zip(x,y)),species)," est le K optimal, votre iris est de l'espèce : ")

 if prediction[0]==0:
  image = Image.open('iris_setosa.jpeg')
  col1, col2, col3 = st.columns([0.2, 0.2, 0.2])
  col2.image(image, use_column_width=True, caption='SETOSA')
 elif prediction[0]==1:
  image = Image.open('iris_versicolor.jpeg')
  col1, col2, col3 = st.columns([0.2, 0.2, 0.2])
  col2.image(image, use_column_width=True, caption='VERSICOLOR')
 else:
  image = Image.open('iris_virginica.jpeg')
  col1, col2, col3 = st.columns([0.2, 0.2, 0.2])
  col2.image(image, use_column_width=True, caption='VIRGINICA')
# Fin affichage résultats

# Fin affichage page principale








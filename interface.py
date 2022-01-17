import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


#traitement CSV
iris=pandas.read_csv("/Users/david/Desktop/iris_KNN/iris.csv")
x=iris.loc[:,"petal_length"]
y=iris.loc[:,"petal_width"]
lab=iris.loc[:,"species"]
#fin traitement CSV


st.sidebar.title('Déterminer l\'espèce de votre iris : ')
longueur = st.sidebar.slider('Entrez la longueur en cm', 0.0, 10.0, 0.1 )
largeur= st.sidebar.slider('Entrez la largueur en cm', 0.0, 10.0, 0.1  )


"""
En 1936, Edgar Anderson a collecté des données sur 3 espèces d\'iris : iris setosa, iris virginica et iris versicolor.
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

#algo knn
d=list(zip(x,y))
model = KNeighborsClassifier(n_neighbors=3)
model.fit(d,lab)
prediction= model.predict([[longueur,largeur]])
#fin algo knn


#Affichage résultats
st.write("Votre iris a des pétales de longueur de ", longueur, ' cm et de largeur de ',largeur,' cm.')
st.write("D'après l'algorithme des k plus proche voisin, votre iris est : ")

if prediction[0]==0:
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_setosa.jpeg')
  st.image(image, caption='SETOSA')
if prediction[0]==1:
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_versicolor.jpeg')
  st.image(image, caption='VERSICOLOR')
if prediction[0]==2:
  image = Image.open('/Users/david/Desktop/iris_KNN/iris_virginica.jpeg')
  st.image(image, caption='VIRGINICA')







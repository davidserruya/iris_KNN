import streamlit as st
from PIL import Image

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: red;'>Home Page</h1>", unsafe_allow_html=True)
st.markdown(
"""
Le k-NN est le diminutif de k Nearest Neighbors. L'algorithme KNN suppose que des objets similaires existent à proximité dans cet espace (plus proches voisins).  
En d'autres termes, des choses similaires sont proches les unes des autres.  
Cette notion de proximité peut-être formaliser par un calcul de distance entre des points du graphique.  
L’algorithme des K plus proches voisins est un algorithme de Machine Learning qui appartient à la classe des algorithmes d’apprentissage supervisé.  
Il est connu pour être simple et facile à mettre en œuvre et peut être utilisé pour résoudre les problèmes de classification et de régression.

""")


st.title("Définition : Algorithme d'apprentissage supervisé")
st.markdown(
"""
En apprentissage supervisé, un algorithme reçoit un ensemble de données qui est étiqueté avec des valeurs de sorties correspondantes sur lequel il va pouvoir s’entraîner et définir un modèle de prédiction (training set).   
Cet algorithme pourra par la suite être utilisé sur de nouvelles données afin de prédire leurs valeurs de sorties correspondantes (testing set).
""")
image = Image.open('home_images/AS.png')
col1, col2, col3 = st.columns([0.2, 1, 0.2])
col2.image([image])


st.title("Principe de l’algorithme des KNN")
st.markdown(
"""
L’intuition derrière l’algorithme des K plus proches voisins est l’une des plus simples de tous les algorithmes de Machine Learning supervisé :

* __Étape 1__ :  Sélectionnez le nombre K de voisins.
* __Étape 2__ :  Calculez la distance du point non classifié aux autres points.  
(On choisit la fonction de distance en fonction des types de données qu’on manipule.  
On y retrouve notamment, la distance euclidienne, la distance de Manhattan, la distance de Minkowski, celle de Jaccard, la distance de Hamming.)
* __Étape 3__ :  Prenez les K voisins les plus proches selon la distance calculée.
* __Étape 4__ :  Parmi ces K voisins, comptez le nombre de points  appartenant à chaque catégorie.
* __Étape 5__ : Attribuez le nouveau point à la catégorie la plus présente parmis ces K voisins.
* __Étape 6__ : Le modèle est prêt :

""")
image2 = Image.open('home_images/modeleKNN.png')
col1, col2, col3 = st.columns([0.2, 1.2, 0.2])
col2.image([image2])

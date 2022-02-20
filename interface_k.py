from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import ImageFilter
from functions import *


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>K MENU</h1>", unsafe_allow_html=True)
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
    @st.cache(allow_output_mutation=True)
    def initialise():
        model,x,y,target,kopt,accuracy,evals =initialiseIris()
        return kopt,accuracy,evals
    kopt,accuracy,evals=initialise();
    
else:   
    @st.cache(suppress_st_warning=True)
    def initialise():
        model,kopt,accuracyopt,evals=initialiseDigit()
        return kopt,accuracyopt,evals
    kopt,accuracyopt,evals=initialise()


evals = pd.DataFrame(evals)
best_k = evals.sort_values(by='accuracy', ascending=False).iloc[0]
fig=plt.figure(figsize=(16, 8))
plt.plot(evals['k'], evals['accuracy'], lw=3, c='#087E8B')
plt.scatter(best_k['k'], best_k['accuracy'], s=200, c='#087E8B')
plt.title(f"K Parameter Optimization, Optimal k = {int(best_k['k'])}", size=20)
plt.xlabel('K', size=14)
plt.ylabel('Accuracy', size=14)
st.pyplot(fig)
    

st.write("Comme on peut le voir, le k-NN le plus performant est celui pour lequel k = ",kopt)




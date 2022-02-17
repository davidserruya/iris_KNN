from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import ImageFilter
from functions import splitDataset, initialiseIris, initialiseDigit, findErrorsK, figurek, findkOpt

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
    @st.cache(allow_output_mutation=True)
    def initialise():
      iris=pd.read_csv("iris.csv")                                                                                                  
      x=iris.loc[:,"petal_length"]                                                                                                                                    
      y=iris.loc[:,"petal_width"]                                                                                                                                      
      data=list(zip(x,y))
      evals = []
      for n_neighbors in range(2,11):
         clf = neighbors.KNeighborsClassifier(n_neighbors)
         clf.fit(data, target) 
         score = accuracy_score(clf.predict(data), target)
         evals.append({'k': n_neighbors, 'accuracy': score})
      return evals
    evals=initialise()
    evals = pd.DataFrame(evals)
    best_k = evals.sort_values(by='accuracy', ascending=False).iloc[0]
    fig=plt.figure(figsize=(16, 8))
    plt.plot(evals['k'], evals['accuracy'], lw=3, c='#087E8B')
    plt.scatter(best_k['k'], best_k['accuracy'], s=200, c='#087E8B')
    plt.title(f"K Parameter Optimization, Optimal k = {int(best_k['k'])}", size=20)
    plt.xlabel('K', size=14)
    plt.ylabel('Accuracy', size=14)
    st.pyplot(fig)
    kopt=best_k
    
else:   
    functionInitialise='initialiseDigit()'
    texte="les chiffres écrits à la main."
    @st.cache(suppress_st_warning=True)
    def initialise():
        return initialiseDigit()
    model,kopt,k,errors=initialise()
    minerror=min(errors)
    kopt=kopt[0]
    st.pyplot(figurek(k,errors))



st.write("Comme on peut le voir, le k-NN le plus performant est celui pour lequel k = ",kopt," avec ",round(minerror,2)," erreurs. On connaît donc notre classifieur final optimal : ",kopt,"-nn. Ce qui veut dire que c'est celui qui classifie le mieux les données, et qui donc dans ce cas précis reconnaît au mieux ",texte) 





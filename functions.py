from os import supports_effective_ids
from sklearn.neighbors import KNeighborsClassifier

# Fonction qui permet de trouver le K optimal
def findk (values,species):
  klist=[]
  # tester K alant de 1 à 15
  for i in range (1,15):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(values,species)
    error = 1 - model.score(values,species)
    klist.append(error)
  min1=min(klist)
  k=klist.index(min1)+1
  return k

# Fonction qui permet de définir d'après l'algorithme des k plus proches voisins 
# l'espèce de l'iris demandé selon sa longueur et sa largeur de taille de pétale
def findPrediction(x,y,species,longueur,largeur):
  values=list(zip(x,y))
  model = KNeighborsClassifier(n_neighbors=findk(values,species))
  model.fit(values,species)
  return model.predict([[longueur,largeur]])
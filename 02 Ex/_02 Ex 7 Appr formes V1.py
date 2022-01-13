import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as FNT

######################################################

# (x,y, category)
points = [ [(0.5,0.4),0],
           [(0.8,0.3),0],
		    [(0.3,0.8),0],
		    [(-.4,0.3),1],
		    [(-.3,0.7),1],
		    [(-.7,0.2),1],
		    [(-.4,-.5),1],
		    [(0.7,-.4),2],
		    [(0.5,-.6),2]]

######################################################
#
#  outils d'affichage -  NE PAS TOUCHER

def DessineFond():
    iS = ComputeCatPerPixel()
    levels = [-1, 0, 1, 2]
    c1 = ('r', 'g', 'b')
    plt.contourf(XXXX, YYYY, iS, levels, colors = c1)

def DessinePoints():
    c2 = ('darkred','darkgreen','lightblue')
    for point in points:
        coord = point[0]
        cat   = point[1]
        plt.scatter(coord[0], coord[1] ,  s=50, c=c2[cat],  marker='o')

XXXX , YYYY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))

##############################################################
#
#  PROJET

# Nous devons apprendre 3 catégories : 0 1 ou 2 suivant ce couple (x,y)

# Pour chaque échantillon, nous avons comme information [(x,y),cat]

# Construisez une couche Linear pour un échantillon prédit un score pour chaque catégorie
layer = torch.nn.Linear(2,3)
# Le plus fort score est associé à la catégorie retenue
get_cat = lambda x: torch.argmax(x, axis=2)
# Pour calculer l'erreur, on connait la bonne catégorie k de l'échantillon de l'échantillon.
# On calcule Err = Sigma_(j=0 à nb_cat) max(0,Sj-Sk)  avec Sj score de la cat j

# Comment interpréter cette formule :
# La grandeur Sj-Sk nous donne l'écart entre le score de la bonne catégorie et le score de la cat j.
# Si j correspond à k, la contribution à l'erreur vaut 0, on ne tient pas compte de la valeur Sj=k dans l'erreur
# Sinon Si cet écart est positif, ce n'est pas bon signe, car cela sous entend que le plus grand
#          score ne correspond pas à la bonne catégorie et donc on obtient un malus.
#          Plus le mauvais score est grand? plus le malus est important.
#       Si cet écart est négatif, cela sous entend que le score de la bonne catégorie est supérieur
#          au score de la catégorie courante. Tout va bien. Mais il ne faut pas que cela influence
#          l'erreur car l'algorithme doit corriger les mauvaises prédictions. Pour cela, max(0,.)
#          permet de ne pas tenir compte de cet écart négatif dans l'erreur.


def ComputeCatPerPixel():
    s = XXXX.shape
    T = torch.stack((torch.FloatTensor(XXXX), torch.FloatTensor(YYYY)), 2)
    scores = model(T)
    CCCC = get_cat(scores)
    return CCCC

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = torch.nn.Linear(2,3)

    def forward(self, x):
        scores = self.layer(x)
        return scores

def error(scores, id_cat):
    error_tot = 0
    for i in range(len(scores)):
        error_tot += torch.max(torch.FloatTensor([0,0,0]), scores[i] - scores[i][id_cat[i]])
    return error_tot

def error_with_d(scores, id_cat, d):
    error_tot = 0
    for i in range(len(scores)):
        error_tot += torch.max(torch.FloatTensor([0,0,0]), scores[i] - scores[i][id_cat[i]] - d)
    return error_tot

iteration = 10

model = Net()

input = torch.FloatTensor([point[0] for point in points])
ref = torch.LongTensor([point[1] for point in points])
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for i in range(iteration):
    optimizer.zero_grad()
    scores = model(input)
    err = error_with_d(scores, ref, 1)
    tot_error = torch.sum(err)
    tot_error.backward()
    optimizer.step()

    DessineFond()
    DessinePoints()
    print(f"Iteration : {i}, Error : {tot_error.item()}")
    plt.title(str(iteration))
    plt.show(block=False)
    plt.pause(0.5)  # pause avec duree en secondes


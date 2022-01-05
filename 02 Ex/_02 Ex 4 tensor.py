from PIL import Image
import matplotlib.pyplot as plt
import time

import torch, torchvision
import sys
path = "c:\\temp\\"   # chemin vers un répertoire temporaire

toPIL    = torchvision.transforms.ToPILImage()
toTensor = torchvision.transforms.ToTensor()

def ShowTensorImg(T2):
  PILImg = toPIL(T2)
  if len(T2.shape) == 3:
      plt.imshow(PILImg)   # RGB
  else:
      plt.imshow(PILImg,cmap='gray') #gray

  plt.show()


def Ex_a():
   a = torch.tensor([ [[1,2,3],[4,5,6]], [[10,20,30],[40,50,60]]])
   # créez un tenseur b permettant par un broadcast : a+b  d'obtenir :
   # torch.tensor([ [[2,2,3],[5,5,6]], [[11,20,30],[41,50,60]]])
   # indice : il faut penser à un tenseur ligne
   b = torch.tensor([1,0,0])
   print(a+b)

def Ex_b():
   a = torch.tensor([1,2])
   # créez un tenseur b permettant par un broadcast : a+b d'obtenir :
   # torch.tensor([ [0,1], [1,2], [4,5], [7,8] ])
   # indice : il faut penser à un tenseur colonne
   b = torch.tensor([[-1,0],[3,6]])
   print(a+b)

def Ex_c():
   a = torch.tensor([1,2])
   # créez un tenseur b permettant par un broadcast : a+b d'obtenir :
   # torch.tensor([ [5,2], [4,3], [3,11] ])
   # indice b.shape = (3,2)
   b = torch.tensor([[4,0],[3,1],[2,9]])
   print(a+b)

def Ex_d():
   a = torch.tensor([[1,2],[3,4]])
   # créez un tenseur b permettant par un broadcast : a+b d'obtenir :
   # torch.tensor([ [[1,2],[3,4]],[[1,2],[3,4]]])
   # indice b.shape = (2,1,1)
   b = torch.tensor([[[0]],[[0]]])
   print(a+b)


def Ex1() :
    # 0 = noir 1 = blanc
    # affichez une image en niveau de gris de 320 de largeur & 200 de hauteur
    # la moitié supérieure sera blanche et la moitié inférieure sera noire
    # attention les images sont stockées sous la forme [y,x]
    T = torch.rand(200,320)
    T[:100, :] = 1
    T[100:, :] = 0
    ShowTensorImg(T)

# Ex1()

def Ex2() :
    # 0 = noir 1 = blanc
    # affichez une image en niveau de gris de 300 de largeur & 300 de hauteur
    # cette image contient en son centre un carré 100x100 rempli de bruit
    # pour réaliser cette exercice, il faudra utiliser la syntaxe T1[a:b,c:d] = T2
    T = torch.rand(300,300)
    T[:100,:] = 0
    T[200:,:] = 0
    T[:,:100] = 0
    T[:,200:] = 0
    ShowTensorImg(T)
# Ex2()

def Ex3():
    # créez une image RVB de 320x200
    # cette image sera remplie d'une couleur unique, un
    # bleu des mers du sud : R = 0 / V = 80% / B = 80%
    # attention les images RGB sont stockées sous la forme [3,y,x]
    T = torch.zeros(3,320,200)
    # RGB = torch.tensor([0,204,204])
    T[0,:,:] = 0
    T[1,:,:] = 204
    T[2,:,:] = 204
    ShowTensorImg(T)
# Ex3()

def Ex4():
    # créez une image en niveau de gris 320x200
    # le fond sera blanc
    # Dessinez une grille de points de sorte que chaque pixel avec des
    # coordonnées x ET y multiples de 4 soient noirs
    # on pensera à la syntaxe a:b:c
    T = torch.ones([320,200])
    T[::4,::4] = 0
    ShowTensorImg(T)
# Ex4()


def Ex5():
    # créez une image en niveau de gris 320x200
    # le fond sera blanc
    # Dessinez une grille de points de sorte que chaque pixel avec des
    # coordonnées x ET y multiples de 4 soient noirs
    # on pensera à la syntaxe a:b:c
    Ex4()
# Ex5()


##############################################################
#
#   Vous ne devez pas utiliser de boucle for dans les exercices suivants :
#
#   Quelques exercices sur de vraies images ! Houha !

filename = "G:\\E5\\ia et jeux\\02 Ex\\_02 Ex 4 tensor.data"

def Ex10() :
    # tenseur contenant 5 images RVB de résolution 150x100
    T1 = torch.load(filename)
    # extraire le sous tenseur correspondant à l'image du penda
    # la fonction shape devrait vous aider à comprendre comment le
    # tenseur T1 est construit
    T2 = T1[2]
    ShowTensorImg(T2)
# Ex10()

def Ex11() :
    # tenseur contenant 5 images RVB de résolution 150x100
    T1 = torch.load(filename)
    # extraire le sous tenseur correspondant à l'image du penda
    # construisez un tenseur de taille (100,150)
    # il va contenir la conversion en image grayscale du penda
    # en utilisant la formule :
    # Gray = 0.3 * R + 0.59 * G + 0.11 * B
    penda = T1[2]
    T2 = torch.zeros(100,150)
    T2 = 0.3 * penda[0] + 0.59 * penda[1] + 0.11 * penda[2]
    ShowTensorImg(T2)
# Ex11()

def Ex12() :
    # créez une image de 300x200, les 4 zones disponibles 2x2 doivent
    # contenir quatre animaux différents
    animals = torch.load(filename)

    an_1 = animals[0]
    an_2 = animals[1]
    an_3 = animals[2]
    an_4 = animals[3]

    T2 = torch.zeros(3,200,300)

    T2[:,:100,:150] = an_1
    T2[:,100:,:150] = an_2
    T2[:,:100,150:] = an_3
    T2[:,100:,150:] = an_4

    ShowTensorImg(T2)
# Ex12()

def Ex13() :
    # créez une image RVB correspondant à la superposition de l'image
    # du penda et du serpent (faites la moyenne des deux animaux ceci pour chaque couche R/V/B)
    animals = torch.load(filename)
    penda = animals[2]
    serpent = animals[4]
    T2 = torch.zeros(penda.shape)
    T2[0] = (penda[0]+serpent[0]) / 2
    T2[1] = (penda[1]+serpent[1]) / 2
    T2[2] = (penda[2]+serpent[2]) / 2
    ShowTensorImg(T2)
# Ex13()


def Ex14() :
    # créez une image RVB 150x100
    # chaque plan R/V/B correspondra à un plan R/V/B d'un animal différent
    animals = torch.load(filename)
    T2 = torch.zeros(animals[0].shape)
    T2[0] = animals[0][0]
    T2[1] = animals[1][1]
    T2[2] = animals[2][2]
    ShowTensorImg(T2)
# Ex14()

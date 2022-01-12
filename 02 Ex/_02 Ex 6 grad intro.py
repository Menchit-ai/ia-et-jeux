import torch

# Comment installer Pytorch : https://pytorch.org/get-started/locally/

####################################################################
#
#  Objectif

# On se propose comme dans l'ex vu en cours de faire apprendre
# à une fonction le comportement de l'opérateur booléen != (différent)
# => 1 si différent   => 0 si égal

# L'apprentissage doit s'effectuer sur le set d'échantillons suivant :
# (4,2)   (6,-3)    (1,1)    (3,3)
# Cela sous-entend que si l'apprentissage réussit, l'évaluation en dehors
# de ces valeurs peut quand même etre erronée.

# La fonction choisie pour l'apprentissage sera : min(a*|xi-yi|,1)
# avec -a- comme unique paramètre d'apprentissage

# la fonction d'erreur sera simplement : |fnt(xi,yi)-verite_i|


####################################################################
#
#  Aspect technique

# Pour forcer les tenseurs à utiliser des nombres flotants,
# nous utilisons la syntaxe suivante :

x = torch.FloatTensor([ 4,  6,  1,  3 ])
y = torch.FloatTensor([ 2, -3,  1,  3 ])

# pour créer notre paramètre d'apprentissage et préciser que pytorch
# devra gérer son calcul de gradient, nous écrivons :

a = torch.FloatTensor([ 0.1 ])
a.requires_grad = True

# Passe FORWARD :
# Essayez de vous passer d'une boucle for.
# Utilisez les tenseurs pour traiter tous les échantillons en parallèle.
# Calculez les valeurs en sortie de la fonction.
# Calculez l'erreur totale sur l'ensemble de nos échantillons.
# Les fonctions mathématiques sur les tenseurs s'utilisent ainsi :
# torch.abs(..) / torch.min(..)  / torch.sum(..)  ...

# def f(a,xi,yi) : return torch.min(torch.FloatTensor([a*torch.abs(xi-yi),1]))

def f(a,xi,yi) : return torch.min(a*torch.abs(xi-yi), torch.ones(xi.shape))
def error(a,xi,yi,ref): return (f(a,xi,yi) - ref)**2

for i in range(100):
    xi = x.reshape(-1, 1)
    yi = y.reshape(-1, 1)

    ref = torch.FloatTensor([1, 1, 0, 0]).reshape(-1,1)
    # response = f(a,xi,yi)
    errors = error(a,xi,yi,ref)

    # Passe BACKWARD :
    # Lorsque le calcul de la passe Forward est terminé,
    # nous devons lancer la passe Backward pour calculer le gradient.
    # Le calcul du gradient est déclenché par la syntaxe :

    tenseur_erreur_totale = torch.sum(errors)
    tenseur_erreur_totale.backward()

    # A chaque itération, affichez la valeur de a et de l'erreur totale
    print("Erreur totale", tenseur_erreur_totale.data)
    print("Valeur de a", a.data)

    # GRADIENT DESCENT :
    # Effectuez la méthode de descente du gradient pour modifier la valeur
    # du paramètre d'apprentissage a. Etrangement, il faut préciser à Pytorch
    # d'arrêter de calculer le gradient de a en utilisant la syntaxe ci-après.
    # De plus, il faut réinitialiser le gradient de a à zéro manuellement :
    step = 0.1
    with torch.no_grad() :
        a -= step *  a.grad
        a.grad.zero_()

    
print("Réponse finale", f(a,xi,yi))
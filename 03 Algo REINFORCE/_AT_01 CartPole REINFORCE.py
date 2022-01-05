import gym
import time
from pyglet.window import key

env = gym.make('CartPole-v0')


# IA basique
# si le charriot est à gauche du point de départ, alors => à droite
# si le charriot est à droite du point de départ, alors => à gauche


def GetAction(x):
    if x < 0 : return 1
    else : return 0


# Main game loop

while True :
    done = False
    observation = env.reset()
    TotalScore = 0
    env.reset()

    while not done  :
        w = env.render()
        x = observation[0]

        action = GetAction(x)
        observation, reward, done, info = env.step(action)
        TotalScore += reward
        time.sleep(0.02)


    print("Score final : " , TotalScore)

env.close()
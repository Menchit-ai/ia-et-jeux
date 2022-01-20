import time
from email import policy

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyglet.window import key

env = gym.make('CartPole-v0')


# IA basique
# si le charriot est à gauche du point de départ, alors => à droite
# si le charriot est à droite du point de départ, alors => à gauche

def GetAction(x):
    if x < 0 : return 1
    else : return 0

layer1 = nn.Linear(4,20)
layer2 = nn.Linear(20,2)
softmax = nn.Softmax()

def get_action_v2(observation):
    x = torch.Tensor(observation)
    x = layer1(x)
    x = layer2(x)
    x = softmax(x)
    m = torch.distributions.categorical.Categorical(x)
    index_action = m.sample()
    log_prob = m.log_prob(index_action)
    return index_action, log_prob

# Main game loop
# optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
for i in range(40) :
    done = False
    observation = env.reset()
    TotalScore = 0
    env.reset()
    reward = []
    log_prob = []

    while not done  :
        if i%20 == 0 : w = env.render()
        x = observation[0]

        action, log_prob = get_action_v2(observation)
        observation, reward, done, info = env.step(action)
        TotalScore += reward

        reward.append(TotalScore)
        log_prob.append(log_prob)

        if i%20 == 0 : time.sleep(0.02)

    optimizer.zero_grad()

    print("Score final : " , TotalScore)

env.close()

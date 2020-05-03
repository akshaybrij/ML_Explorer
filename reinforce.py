import os
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = "CartPole-v0"
env = gym.make(env)
np.random.seed(123)
env.seed(123)
n_action_space = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
model.add(Dense(16,))
model.add(Activation('relu'))
model.add(Dense(n_action_space))
model.add(Activation('linear'))
policy =  EpsGreedyQPolicy()
memory = SequentialMemory(limit = 50000, window_length = 1)
dql = DQNAgent(model =model , memory=memory,nb_actions = n_action_space,nb_steps_warmup = 10,target_model_update = 0.01,policy = policy)

dql.compile(Adam(lr=0.001),metrics=['mae'])
dql.fit(env,nb_steps=1000,visualize=True,verbose=True)
dql.test(env,nb_episodes=105,visualize=True)

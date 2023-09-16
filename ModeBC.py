import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import d4rl
from stable_baselines3 import SAC
import numpy as np
from tensorflow.keras.layers import Input, Dense, Add, Concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
import copy
import utils



class ModeBC:

    def __init__(self, dataset, dataset2, state_dim, action_dim, env_name, tau, net, mean, N):
        self.dataset = copy.deepcopy(dataset)
        self.dataset2 = copy.deepcopy(dataset2)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.tau = tau
        self.net = net
        self.mean = mean
        self.N = N
        self.model = self.create_model()
        self.train()

    def create_model(self):
        s = Input((self.state_dim,), name="s")
        o = Input((self.state_dim,), name="o")

        nets = [[] for _ in range(self.N)]
        for i in range(self.N):
            nets[i].append(Dense(self.net[0], activation='relu'))
            nets[i].append(Dense(self.net[0], activation='relu'))
            nets[i].append(Dense(self.action_dim, activation='linear'))

        a_s = [nets[i][2](nets[i][1](nets[i][0](s))) for i in range(self.N)]
        a_o = [nets[i][2](nets[i][1](nets[i][0](o))) for i in range(self.N)]
        diffs = [a_o[i] - a_o[i+1] for i in range(self.N - 1)]
        a_s.extend(diffs)
        loss_weights = [1] * self.N
        loss_weights.extend(np.ones(self.N - 1) * self.tau)
        loss_weights = np.array(loss_weights).tolist()

        print("Anzahl Losses: ", len(a_s))
        print("N + N - 1: ", self.N + self.N - 1)
        print("loss_weights: ", loss_weights)
        print("len(loss_weights): ", len(loss_weights))

        model = Model(inputs=[s,o], outputs=a_s)
        opt = keras.optimizers.Adam(learning_rate=0.0006)
        model.compile(loss=['mse' for _ in range(len(a_s))], loss_weights=loss_weights, optimizer=opt)
        return model

    def train(self):
        while len(self.dataset2["states"]) < len(self.dataset["states"]):
            self.dataset2["states"] = np.append(self.dataset2["states"], self.dataset2["states"], axis=0)
        self.dataset2["states"] = self.dataset2["states"][:len(self.dataset["states"]),:]

        sl_targets = [self.dataset["actions"] for _ in range(self.N)]
        sl_targets.extend([np.zeros_like(self.dataset["actions"]) for _ in range(self.N - 1)])

        print("len(sl_targets): ", len(sl_targets))

        chart = [utils.test(self, self.env_name)]
        for _ in range(15):
            self.model.fit([self.dataset["states"], self.dataset2["states"]], sl_targets,
                           batch_size=1000, epochs=200, verbose=0)
            chart.append(utils.test(self, self.env_name))
            print(chart)

    def predict(self, s):
        predictions = self.model([np.array([s]), np.array([s])])[:self.N]
        a = [predictions[i][0] for i in range(self.N)]
        if self.mean:
            return np.mean(a, axis=0)
        else:
            return np.median(a, axis=0)



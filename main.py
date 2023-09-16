import utils
from ModeBC import ModeBC
import gym
import d4rl
from stable_baselines3 import SAC
import numpy as np




#env_name = 'halfcheetah-random-v0'
#policy = SAC.load("HC_expert")
#state_dim, action_dim = 17, 6

env_name = 'walker2d-random-v0'
policy = SAC.load("Walker2D_expert")
state_dim, action_dim = 17, 6

#env_name = 'hopper-random-v0'
#policy = SAC.load("Hopper_expert")
#state_dim, action_dim = 11, 3

utils.seed(0)
print(env_name)

dataset = utils.create_dataset(policy, random=False, env_name=env_name, n_episodes=1)
for _ in range(3):
    dataset["states"] = np.append(dataset["states"], dataset["states"], axis=0)
    dataset["actions"] = np.append(dataset["actions"], dataset["actions"], axis=0)
dataset2 = utils.create_dataset(policy, random=True, env_name=env_name, n_episodes=5)


for seed in range(10):
    utils.seed(seed)
    print("N = 10")
    print("x1:")
    x1 = ModeBC(dataset, dataset2, state_dim, action_dim, env_name, tau=0, net=[128, 128, 128, 128], mean=True, N=10)
    print("x2:")
    x2 = ModeBC(dataset, dataset2, state_dim, action_dim, env_name, tau=0, net=[128, 128, 128, 128], mean=False, N=10)
    print("x3:")
    x3 = ModeBC(dataset, dataset2, state_dim, action_dim, env_name, tau=1, net=[128, 128, 128, 128], mean=True, N=10)


for seed in range(10):
    utils.seed(seed)
    print("N = 4")
    print("x1:")
    x1 = ModeBC(dataset, dataset2, state_dim, action_dim, env_name, tau=0, net=[128, 128, 128, 128], mean=True, N=4)
    print("x2:")
    x2 = ModeBC(dataset, dataset2, state_dim, action_dim, env_name, tau=0, net=[128, 128, 128, 128], mean=False, N=4)
    print("x3:")
    x3 = ModeBC(dataset, dataset2, state_dim, action_dim, env_name, tau=1, net=[128, 128, 128, 128], mean=True, N=4)


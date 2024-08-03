import retro
import numpy as np
import cv2
from gym import Env
from gym.spaces import MultiBinary, Box
from sonic import SonicInfo
from matplotlib import pyplot as plt
import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

env = SonicInfo()
# Reset game to starting state
obs = env.reset()
print(env.action_space.sample())
# Set flag to flase
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if rew != 0:
            print(rew)
            print(info)
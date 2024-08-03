import retro
import numpy as np
import cv2
import os
import optuna
from moves import DiscreteSonic
from gym import Env
from gym.spaces import MultiBinary, Box
from sonic import SonicInfo
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

LOG_DIR = './logs/'
OPT_DIR = './opt/'

#initialize the environment
env = SonicInfo()
#env = DiscreteSonic(env)
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load('./train/best_model_10000.zip')

#resets game environment
obs = env.reset()

#flag is false
done  = False
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
            
        env.render()
        action = model.predict(obs)[0]
        obs, rew, done, info = env.step(action)
        #if rew != 0:
            #print(rew)
            #print(action)

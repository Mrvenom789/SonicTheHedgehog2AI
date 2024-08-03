import retro
import numpy as np
import cv2
import os
import optuna

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
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load('./train/best_model_20000.zip')
mean_reward, _ = evaluate_policy(model, env, render=True, n_eval_episodes=1)
print(mean_reward)

obs = env.reset()
env.step(model.predict(obs)[0])

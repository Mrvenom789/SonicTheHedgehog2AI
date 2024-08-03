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

class Callback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(Callback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def __init__callback(self):
        if self.save_path is not None:
            os.makedirs = os.path.join(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        
        return True

#copy the parameters from the best performing model
def best_parameters():
    return{
        'n_steps':5328,
        'gamma':0.8787580534986238,
        'learning_rate':8.487926022385588e-05,
        'clip_range':0.1821290077985372,
        'gae_lambda':0.9886853119954124
    }

CHECK_DIR = './train/'
call = Callback(check_freq=10000, save_path=CHECK_DIR)

new_param = best_parameters()

#initialize the environment
env = SonicInfo()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

#create the algorithm
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **new_param)

#load the model
model.load(os.path.join(OPT_DIR, 'trial_4_best_model.zip'))

model.learn(total_timesteps=1000000, callback=call)

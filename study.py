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

LOG_DIR = './logs/'
OPT_DIR = './opt/'
    
#define the hyperparameters
def ppo(trial):
    return{
        'n_steps':trial.suggest_int('n_steps', 2048, 8192),
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 0.00001, 0.0001),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

#define the agent and return the mean reward
def agent(trial):
    try:
        parameters = ppo(trial)

        #initialize the environment
        env = SonicInfo()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        #create the algorithm
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **parameters)
        model.learn(total_timesteps=50000)

        #evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()
        
        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        print(SAVE_PATH)
        return mean_reward
    except Exception as e:
       return -10000



setup = optuna.create_study(direction='maximize')
setup.optimize(agent, n_trials=3, n_jobs=1)
print(setup.best_params)
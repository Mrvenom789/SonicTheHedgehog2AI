import retro
import numpy as np
import cv2
import gym
from gym import Env
from gym.spaces import MultiBinary, Box
from matplotlib import pyplot as plt

#create environment
class SonicInfo(Env):
    def __init__(self):
        super().__init__()
        #define action/observation spaces
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)

        #new instance of game with new conditions
        self.game = retro.make(game="SonicTheHedgehog2-Genesis", use_restricted_actions=retro.Actions.FILTERED)

    def step(self, action):
        #take a step (or many because Sonic runs)
        obs, rew, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        prev_x = info['x']

        #calculate difference between frames
        diff = obs - self.previous_frame
        self.previous_frame = obs

        #set up rewards
        scoreRew = info['score'] - self.score
        self.score = info['score']

        livesRew = info['lives'] - self.lives
        self.lives = info['lives']

        ringRew = info['rings'] - self.rings
        self.rings = info['rings']

        bonusRew = info['level_end_bonus'] - self.level_end_bonus
        self.level_end_bonus = info['level_end_bonus']

        xRew = info['x'] - self.x
        self.x = info['x']

        yRew = info['y'] - self.y
        self.y = info['y']

        #penalize the model for standing still
        penalty = 0
        if xRew == 0:
            penalty -= 1
            action = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        

        rew = scoreRew + ringRew + bonusRew + xRew + yRew + penalty + livesRew

        return diff, rew, done, info


    def render(self, *args, **kwargs):
        self.game.render()

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)

        #set previous frame
        self.previous_frame = obs

        #set default values for different variables
        self.score = 0
        self.rings = 0
        self.level_end_bonus = 0
        self.lives = 3
        self.x = 95
        self.y = 656
        return obs

    def preprocess(self, observation):
        #grayscales the image
        grayImage = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        #resize the image
        size = cv2.resize(grayImage, (84, 84), interpolation=cv2.INTER_CUBIC)

        #add channel
        channel = np.reshape(size, (84, 84, 1))

        return channel

    def close(self):
        self.game.close()
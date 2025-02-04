# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:01:21 2020

@author: Connor
Preprocessing the Atari environment to fit our model, following the findings found in the paper
"""

#Importing Libraries
import numpy as np
import collections
import cv2
import gym

# Repeats action and takes max of last 2 frames
class RepeatActionMaxFrame(gym.Wrapper):
    def __init__(self, env = None, repeat = 4, clip_reward = False, no_ops = 0, fire_first = False):
        super(RepeatActionMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        
    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward
            index = i % 2
            self.frame_buffer[index] = obs
            if done:
                break
        
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        
        return max_frame, total_reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'Fire'
            obs, _, _, _ = self.env.step(1)
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs
        
        return obs

# Converts observation to grayscale, changes shape for pytorch syntax, resizes and downscales frame for faster processing
class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env = None):
        super(PreProcessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = self.shape, dtype = np.float32)
    
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation = cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype = np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        
        return new_obs
    
# Stacks frame necessary for observations         
class StackFrame(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis = 0), 
                            env.observation_space.high.repeat(repeat, axis = 0),
                            dtype = np.float32)
        self.stack = collections.deque(maxlen = repeat)
    
    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)
    
    def observation(self, observation):
        self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)
    
# Creates the environment with all above preprocesing 
# Added reward clipping, no ops and fire first for testing purposes
def make_env(env_name, shape = (84, 84, 1), repeat = 4, clip_rewards = False, no_ops = 0, fire_first = False):
    env = gym.make(env_name)
    env = RepeatActionMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreProcessFrame(shape, env)
    env = StackFrame(env, repeat)
    
    return env
    
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:17:53 2020

@author: Connor
Agent to handle all learning
"""

#import libraries
import torch
import numpy as np
from DQN import DQN
from agent_memory import ReplayBuffer

# Agent to learn
# Some parameters differ from paper due to learning on a lower scale
class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min = 0.01,
                 eps_dec = 5e-7, replace = 1000, algo = None, env_name = None, model_dir = 'temp\dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace = replace
        self.algo = algo
        self.env_name = env_name
        self.dir = model_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_counter = 0
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q = DQN(self.lr, self.n_actions, input_dims = self.input_dims, 
                     name = self.env_name + '_' + self.algo + '_' + 'q',
                     model_dir = self.model_dir)
        self.q_next = DQN(self.lr, self.n_actions, input_dims = self.input_dims, 
                     name = self.env_name + '_' + self.algo + '_' + 'q_next',
                     model_dir = self.model_dir)
        
    def choose_action(self, state):
    
    def copy_target_network(self):
        
    def decrease_eps(self):
    
    def learn(self):
    
    def store_memory(self):
    
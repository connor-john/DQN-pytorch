# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:39:12 2020

@author: Connor
Agent's replay memory
"""

# Importing libraries
import numpy as np

# Handles agents memory and sampling in numpy array format
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cn = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.uint8)
    
    def transition(self, state, action, reward, state_, done):
        i = self.mem_cn % self.mem_size
        self.state_memory[i] = state
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.new_state_memory[i] = state_
        self.terminal_memory[i] = done
        self.mem_cn += 1
    
    def sample_memory(self, batch_size):
        max_mem = min(self.mem_cn, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        # Sample
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
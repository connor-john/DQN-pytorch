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

# Agent to manage learning and networks
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
                     model_dir = self.dir)
        self.q_next = DQN(self.lr, self.n_actions, input_dims = self.input_dims, 
                     name = self.env_name + '_' + self.algo + '_' + 'q_next',
                     model_dir = self.dir)
    
    # Using epsilon greedy     
    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = torch.Tensor([obs], dtype = torch.float).to(self.q.device)
            actions = self.q.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_memory(self.batch_size)
        
        states = torch.tensor(state).to(self.q.device)
        rewards = torch.tensor(reward).to(self.q.device)
        dones = torch.tensor(done).to(self.q.device)
        actions = torch.tensor(action).to(self.q.device)
        states_ = torch.tensor(new_state).to(self.q.device)
        
        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q.state_dict())
        
    def decrease_eps(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        
    def learn(self):
        if self.memory.mem_cn < self.batch_size:
            return
        
        self.q.zero_grad()
        self.replace_target_network()
        
        states, actions, rewards, states_, dones = self.sample_memory()
        
        index = np.arange(self.batch_size)
        q_pred = self.q.forward(states)[index, actions]
        q_next = self.q_next.forward(states_).max(dim = 1)[0]
        
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        
        loss = self.q.loss(q_target, q_pred).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()
        self.learn_counter += 1
        
        self.decrease_eps()
    
    def save_models(self):
        self.q.save_model()
        self.q_next.save_model()
    
    def load_models(self):
        self.q.load_model()
        self.q_next.load_model()
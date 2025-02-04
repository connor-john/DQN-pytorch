# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:52:31 2020

@author: Connor
Dueling Deep Q Network
"""

#import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Deep Q network architecture for Dueling DQN
class DQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, model_dir):
        super(DQN, self).__init__()
        
        self.directory = model_dir
        self.file = os.path.join(self.directory, name)
        
        self.cv1 = nn.Conv2d(input_dims[0], 32, kernel_size = 8, stride = 4)
        self.cv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.cv3 = nn.Conv2d(64, 64, kernel_size = 3)
        
        fc_input = self.conv_output_dims(input_dims)
        
        self.fc1 = nn.Linear(fc_input, 512)
        
        # Split into Value and Advantage
        self.val = nn.Linear(512, 1)
        self.adv = nn.Linear(512, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        cv1 = F.relu(self.cv1(state))
        cv2 = F.relu(self.cv2(cv1))
        cv3 = F.relu(self.cv3(cv2))
        #reshape output for Fully connected layer
        conv_state = cv3.view(cv3.size()[0], -1)
        flat = F.relu(self.fc1(conv_state))
        # Split into Value and Advantage
        val = self.val(flat)
        adv = self.adv(flat)
        
        return val, adv        
    
    def conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.cv1(state)
        dims = self.cv2(dims)
        dims = self.cv3(dims)
        
        return int(np.prod(dims.size()))
    
    def save_model(self):
        print('Saving Model.....')
        torch.save(self.state_dict(), self.file)
    
    def load_model(self):
        print('Loading Model....')
        self.load_state_dict(torch.load(self.file))
        



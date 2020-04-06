# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:44:24 2020

@author: Connor
Main method for training Dueling DQN
"""

# Importing Libraries
import numpy as np
from dueling_agent import DuelingAgent
from preprocessing import make_env
from utils import plot_learning_curve

# Main method
if __name__ == '__main__':
    
    # Initialise
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load = False
    n_games = 300
    agent = DuelingAgent(gamma = 0.99, epsilon = 1.0, lr = 0.0001,input_dims = (env.observation_space.shape),
                  n_actions = env.action_space.n, mem_size = 50000, eps_min = 0.1, batch_size = 32,
                  replace = 1000, eps_dec = 1e-5, model_dir = 'models/', algo = 'DuelingDQN', env_name = 'PongNoFrameSkip-v4')
    
    if load:
        agent.load_models()
    
    file_name = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    plot_file = 'plots/' + file_name + '.png'
    
    n_steps = 0
    scores = []
    eps_history = []
    steps_array = []
    
    # Training Loop
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
                
            observation = observation_
            n_steps += 1
            
        scores.append(score)
        steps_array.append(n_steps)
        
        avg_score = np.mean(scores[-100:])
        print('episode: ', i,' | score: ', score, ' | average score %.1f' % avg_score, 
              ' | best score %.2f' % best_score, ' | epsilon %.2f' % agent.epsilon, ' | steps', n_steps)
        
        if avg_score > best_score:
            if not load:
                agent.save_models()
            best_score = avg_score
        
        eps_history.append(agent.epsilon)
        
    plot_learning_curve(steps_array, scores, eps_history, plot_file)
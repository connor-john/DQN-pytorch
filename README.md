# DQN-pytorch
 Collection of DQN variations from papers, implemented in pytorch<br><br>
 <img src="https://images.exxactcorp.com/CMS/landing-page/resource-center/supported-software/deep-learning/pytorch/PyTorch-logo.jpg" width=300 align=left><br><br><br>
 
--- Gif of gameplay here 
 
 ### DQN
 Implementation of [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
 #### Summary
 Using a convolutional neural network to take inputs, as well introducing a network termed experience replay, and only periodically updating the _Q-value_ will allow the DQN to perform better than a standard naive DQN
 #### Results
 **Pong** results of DQN trained on 500 episodes<br>
<img src ="DQN/plots/DQN_PongNoFrameSkip-v4_lr0.0001_500games.png" width = 450>

 ### DDQN
 Implementation of [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
 #### Summary
 Fix DQN's over-estimation of some action values by decoupling the selection from the evaluation of an action
 #### Results
--- results comparison here

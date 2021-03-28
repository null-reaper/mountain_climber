# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 01:57:22 2020

@author: Clive Gomes <cliveg@andrew.cmu.edu>
@title: 10-601 Intro to Machine Learning Homework #8
"""
import sys
from environment import MountainCar
import numpy as np
import matplotlib.pyplot as plt

# Compute Q-Value
def q(s, a, w, b):
    return np.dot(s.T, w[a]) + b
    
# Update Weights & Bias
def update(s, a, r, s2, w, b, alpha, gamma, actions):
    partial = q(s, a, w, b) - (r + (gamma*max([q(s2, a2, w, b) for a2 in actions])))
    partial *= alpha
    
    w[a] -= (partial*s)
    b -= partial
    
    return w, b
    
# Select Next Action
def select(s, w, b, epsilon, actions):
    rand = np.random.rand()
    
    if (rand < epsilon):
        return np.random.choice(actions)
    else:
        return np.argmax([q(s, a2, w, b) for a2 in actions])
    
# Main Function for Approximate Q-Learning
def main(args):
    # Get Input
    mode = args[1] # 'tile' or 'raw'
    
    weightsOutPath = args[2]
    returnsOutPath = args[3]
    
    episodes = int(args[4]) # suggested: 400
    max_iter = int(args[5]) # suggested: 200
    
    epsilon = float(args[6])  # suggested: 0.05
    gamma = float(args[7]) # suggested: 0.99
    alpha = float(args[8]) # suggested: 0.01
    
    # Define Actions
    actions = [0, 1, 2]
    
    # Initialize Environment
    mountainCar = MountainCar(mode)
    N = mountainCar.state_space
    
    # Get Initial State
    sDict = mountainCar.reset()
    s = np.array([sDict[i] if i in sDict else 0.0 for i in range(N)], dtype=float)
    
    # Initial Weights & Bias
    w = np.zeros([len(actions), N], dtype=float)
    b = 0
     
    # Open Output Files
    weightsOut = open(weightsOutPath, 'w')
    returnsOut = open(returnsOutPath, 'w')
    
    returns = []
    
    # Perform Approximate Q-Learning
    for i in range(episodes):
        # Reset Environment
        sDict = mountainCar.reset()
        s = np.array([sDict[i] if i in sDict else 0.0 for i in range(N)], dtype=float)
        reward = 0
        
        for j in range(max_iter):
            # Display Visuals
            mountainCar.render()
            
            # Select & Perform Next Action
            a = select(s, w, b, epsilon, actions)
            s2Dict, r, done = mountainCar.step(a)
            s2 = np.array([s2Dict[i] if i in s2Dict else 0.0 for i in range(N)], dtype=float)

            # Increment Reward
            reward+= r
            
            # Update Weights & Bias
            w, b = update(s, a, r, s2, w, b, alpha, gamma, actions)
            
            # Update State
            s = s2
        
        # Output Returns (Reward)
        returns.append(reward)
        returnsOut.write(str(reward) + '\n')
    
    # Plot w/ Rolling Average
    rolling = []
    for i in range(25, 401):
        rolling.append(sum(returns[i-25:i])/25)
        
    x1 = [i+1 for i in range(400)]
    x2 = [i+1 for i in range(24, 400)]
    plt.plot(x1, returns, color='cyan')
    plt.plot(x2, rolling, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(["True Value", "Rolling Mean (Window Size = 25)"])#, loc ="lower right")  

    # Output Bias
    weightsOut.write(str(b) + '\n')
    
    # Ouput Weights
    for row in w.T:
        for x in row:
            weightsOut.write(str(x) + '\n')
    
    # Close Output Files
    weightsOut.close()
    returnsOut.close()
    
if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:16:07 2019

@author: Shay Aharon

***The FrozenLakeAgent object holds the agents learning information (i.e Qtable and parameters for learning)

***qDerivedPolicy is a deminishing epsilon greedy exploration algorithm for better convergence properties
(for more information http://webee.technion.ac.il/shimkin/LCS11/ch7_exploration.pdf)

***using pickle library to store the agents information for later use without the need to perform training again

***if program is being ran from __main___ : the agent is trained, the average reward over time in training is plotted, and
a test of the obtained policy is being conducted

***note that the success rate being plotted in the end of training does not fully reflect on the agents successfullness
due to high exploration from the qDerivedPolicy algorithm, therefore the test's success rate is higher.
"""
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

class FrozenLakeAgent:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=0.0, exploration_epsilon=1.0, num_episodes=15000):
        self.observation_space = np.arange(16)
        self.visits = np.ones(16)
        self.action_space = np.arange(4)
        self.Q = np.zeros([16, 4])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.exploration_epsilon = exploration_epsilon
        self.num_episodes = num_episodes
        self.episode_length = 99
        
    def qDerivedPolicy(self, env, state, learning):
        if (random.uniform(0, 1) > self.epsilon) and (np.max(self.Q[state,:]) > 0) :
            action = np.argmax(self.Q[state,:])
        else:
            action = env.action_space.sample()        
        if learning:
            exploration_prob = np.max([(1/self.visits[state])*self.exploration_epsilon, 0.01])
            if (random.uniform(0, 1) < exploration_prob):
                action = env.action_space.sample()
            self.visits[state]+=0.2
        return action

    def SARSAlearn(self, env):
        rates = []
        rList = []
        for i in range(self.num_episodes):
            state = env.reset()
            action = self.qDerivedPolicy(env, state, learning = True)
            done = False
            rAll = 0
            step = 0
            while not done:
                #env.render() #uncomment to display the game
                new_state, reward, done, info = env.step(action)
                done = done or self.episode_length < step
                new_action = self.qDerivedPolicy(env, new_state, learning = True)
                self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[new_state, new_action] - self.Q[state, action])
                rAll += reward
                state = new_state
                action = new_action
                rList.append(rAll)
            if i % 500 == 0 and i is not 0:
                rates.append(sum(rList) / i)
        plt.plot(rates)
        plt.title('Average success rate over time')
        plt.xlabel('Time units of 500 episodes')
        plt.ylabel('Average success rate')
        plt.show('Average success rate')
        
    def test(self, env, n):
        rewardList = []
        for i in range(n):
            state = env.reset()
            action = self.qDerivedPolicy(env, state, learning = False)
            done = False
            rAll = 0
            while not done:
                #env.render() #uncomment to display the game
                new_state, reward, done, info = env.step(action)
                new_action = self.qDerivedPolicy(env, new_state, learning = False)
                rAll += reward
                state = new_state
                action = new_action
            rewardList.append(rAll)
        print("Success rate test: " + str(sum(rewardList)/n))

"""to observe a learning process from scratch, delete the pickle file myFrozemAgent.p """
def main():
    try:
        agent = pickle.load( open( "myFrozenLakeAgent.p", "rb" ) )
        env = gym.make("FrozenLake-v0")
        env.seed(1992) #seeding to produce same environment
        agent.test(env, 100)
    except (OSError, IOError) :
        agent= FrozenLakeAgent()
        env = gym.make("FrozenLake-v0")
        env.seed(1992)
        agent.SARSAlearn(env)
        agent.test(env, 100)
        pickle.dump( agent, open( "myFrozenLakeAgent.p", "wb" ) )
    
if __name__ == "__main__":
    main()
"""
Implementation of Actor-Critic algorithm (P.274 of Sutton's book) and apply it to the (continuous) mountain car problem

Tile coding is used to discretize the continuous state space 
@author: Boran
"""
import gym
from matplotlib import pyplot as plt
import matplotlib.style
import itertools
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict,namedtuple
from lib import plotting

from tiles3 import IHT, tiles

matplotlib.style.use('ggplot') # for emulating the aesthetics of ggplot (a popular plotting package for R).

#%%
env = gym.envs.make("MountainCar-v0")  
# state: (position, velocity)
# action: 0-push left, 1-no push, 2-push right
# utility functions
def get_tiles(iht,num_tilings,state,tile_scale,action=None):
                
        # # use tile coding to construct the feature vector
        if action == None:
            mytiles = tiles(iht,num_tilings,state*tile_scale)
        else:
            mytiles = tiles(iht,num_tilings,state*tile_scale,[action])

        # feature_vec = self.feature_vec_init
        # feature_vec[mytiles] = 1
        return mytiles
def get_parameterXfeature(parameter_vec,tiles):
    
    parameterXfeature = 0
    for tile in tiles:
        parameterXfeature += parameter_vec[tile]
    return parameterXfeature

#%%
class PolicyEstimator():
    """
    Policy Function Approximator
    
    use the softmax function: pi(a|s,theta) = exp(h(s,a,theta))/sum_b(exp(h(s,b,theta))),
    where 
    h(s,a,theta) = theta^T*x(s,a), x(s,a) is the feature vector
    
    """
    def get_feature_vec(self,tiles):
        feature_vec = self.feature_vec_zero
        feature_vec[tiles] = 1
        return feature_vec
    
    def __init__(self,learning_rate=0.001,discount_factor=1.0):
        self.nA = 3
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
        # for using tile coding
        self.maxSize = 2048
        self.iht = IHT(self.maxSize)  #initialize the hash table 
        self.num_tilings = 8
        self.tile_scale = np.array([8.0/(0.6+1.2),8.0/(0.07+0.07)])
        self.theta = np.zeros(self.maxSize)
        self.feature_vec_zero = np.zeros(self.maxSize)

    def predict(self, state):
        # predict the value of pi(a|s,theta) for all a
        pi_s = np.zeros(self.nA)
        for a in range(self.nA):
            mytiles = get_tiles(self.iht, self.num_tilings,state,self.tile_scale,a)
            pi_s[a] = np.exp(get_parameterXfeature(self.theta,mytiles))
            # pi_s[a] = np.exp(self.theta.dot(self.get_feature_vec(state,a)))
        # normalization     
        pi_s = pi_s/sum(pi_s)
        return pi_s 
    
    def update(self, state, td_error, action, discount_gain):
        # calculate the gradient of ln pi(a/s,theta) with respect to theta, which is x(s,a)-sum_b(pi(b/s,theta)*x(s,b))
        pi_s = self.predict(state)
        # grad_ln_pi = self.get_feature_vec(state,action) - sum([pi_s[b]*self.get_feature_vec(state,b) for b in range(self.nA)])
        # self.theta += self.learning_rate*discount_gain*td_error*grad_ln_pi

        tiles= get_tiles(self.iht, self.num_tilings,state,self.tile_scale,action)
        for tile in tiles:
            self.theta[tile]+=self.learning_rate*discount_gain*td_error
        for b in range(self.nA):
            tiles = get_tiles(self.iht, self.num_tilings,state,self.tile_scale,b)
            for tile in tiles:
                self.theta[tile] -= self.learning_rate*discount_gain*td_error*pi_s[b]
         
#%%       
class ValueEstimator():
    """
    Value Function Approximator
    
    use linear approximation for the value function
    v(s,w) = w^T*x(s), where x(s) is the feature vector
    update
    w = w+ learning_rate*discount_factor^t*advantage*grad_v
    """
       
    def get_feature_vec(self,tiles):
        feature_vec = self.feature_vec_zero
        feature_vec[tiles] = 1
        return feature_vec

    def __init__(self,learning_rate=0.01,discount_factor=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # for using tile coding
        self.maxSize = 1024
        self.iht = IHT(self.maxSize)  #initialize the hash table 
        self.num_tilings = 8
        self.tile_scale = np.array([8.0/(0.6+1.2),8.0/(0.07+0.07)])
        self.feature_vec_zero = np.zeros(self.maxSize)
        self.w = np.zeros(self.maxSize)  
    
    def predict(self, state):
        # predict the value of v(s) 
        mytiles = get_tiles(self.iht,self.num_tilings,state,self.tile_scale)
        v_s = get_parameterXfeature(self.w,mytiles)

        return v_s 
    
    def update(self, state, td_error,discount_gain):
        # calculate the gradient of v(s,w) with respect to w
        # grad_v = self.get_feature_vec(state)
        # self.w = self.w + self.learning_rate*discount_gain*td_error*grad_v

        tiles= get_tiles(self.iht,self.num_tilings,state,self.tile_scale)
        for tile in tiles:
            self.w[tile]+=self.learning_rate*discount_gain*td_error
    
#%%
def actor_critic_continuous(env, estimator_policy, estimator_value,num_episodes,discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # keep track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths = np.zeros(num_episodes), 
                                  episode_rewards = np.zeros(num_episodes))
    
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # reset the environment and pick the first action
        state = env.reset()
        
        episode = []
        discount_gain = 1
        # one step in the environment
        for t in itertools.count():
            env.render()            
            # take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)  #[action] should be used for MountainCarContinuous
            
            # keep track of the transition 
            episode.append(Transition(
                    state = state, action = action, reward = reward, next_state = next_state,done = done))
            
            # keep track of the statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # calculate TD error
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor*value_next
            td_error = td_target - estimator_value.predict(state)
            
            # Update the value estimator                 
            estimator_value.update(state,td_error,discount_gain)
            # Update the policy estimator 
            estimator_policy.update(state,td_error,action,discount_gain)
            
             # Print out which step we're on, useful for debugging.
            print("\rStep {} state:{} action:{} @ Episode {}/{} ({})\n".format(
                    t, state, action, i_episode + 1, num_episodes, stats.episode_rewards[i_episode]), end="")
            # sys.stdout.flush()
            
            if done:
                break
            
            discount_gain  = discount_gain*discount_factor
            state = next_state
            
            
    return stats
#%%
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

stats = actor_critic_continuous(env, policy_estimator, value_estimator, 1000, discount_factor=1.0)

#%% plot the statistics 
fig1,fig2,fig3 = plotting.plot_episode_stats(stats, smoothing_window=10)
plt.figure(fig1.number)
plt.savefig("EpisodeLength_ActorCritic_TileCoding_MountainCar")
plt.figure(fig2.number)
plt.savefig("EpisodeReward_ActorCritic_TileCoding_MountainCar")

# at the end call show to ensure figureswindow won't close.
plt.show() 
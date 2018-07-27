import numpy as np
import random
from tiles3 import IHT, tiles

from car_env import WARN, NOT_WARN

from collections import namedtuple 

# Tile_Paras = namedtuple('Tile_Paras',['maxSize','num_tilings','num_grids','feature_vec_zero',''])

# utility functions
def get_parameterXfeature(parameter_vec,tiles):
    parameterXfeature = 0
    for tile in tiles:
        parameterXfeature += parameter_vec[tile]
    return parameterXfeature

class Q_Agent_LFA():
    """
    An agent for off-policy q learning with linear function approximation and tile coding for the features
    """
    def __init__(self,num_actions,state_bounds,tile_coding ={'maxSize':1024,'num_tilings':8,'num_grids':10},
                learning_rate = 0.01, discount_factor = 0.9):   
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.state_bounds = state_bounds
        self.nA = num_actions
        # for using tile coding to construct the feature vector
        self.maxSize = tile_coding['maxSize']
        self.iht = IHT(self.maxSize)  #initialize the hash table 
        self.num_tilings = tile_coding['num_tilings']
        self.tile_scale = tile_coding['num_grids']/(state_bounds[1]-state_bounds[0])
        self.feature_vec_zero = np.zeros(self.maxSize)
        self.w = np.zeros(self.maxSize)  

    def epsilon_greedy_policy(self,state, epsilon):
        # Get the probabilities for all the actions
        q_values_state = self.predict(state)
        best_action = np.argmax(q_values_state)
        action_probs = np.ones(self.nA, dtype=float) * epsilon / self.nA        
        action_probs[best_action] += (1.0 - epsilon)

        # Sample an action according to the probabilities
        return np.random.choice(np.arange(self.nA), p = action_probs)

    def predict(self,state):
        """
        Give a state, predict the state-action values for all states 
        """
        q_s = np.zeros(self.nA)
        for a in range(self.nA):
            my_tiles = self.get_tiles(state,a)
            q_s[a] = get_parameterXfeature(self.w,my_tiles)
        return q_s

    def get_tiles(self,state,action):
        # # use tile coding to construct the feature vector
        
        # clip the state
        clipped_state = np.clip(state,self.state_bounds[0],self.state_bounds[1])

        if action == None:
            mytiles = tiles(self.iht,self.num_tilings,clipped_state*self.tile_scale)
        else:
            mytiles = tiles(self.iht,self.num_tilings,clipped_state*self.tile_scale,[action])

        # feature_vec = self.feature_vec_init
        # feature_vec[mytiles] = 1
        return mytiles

    def update(self,state,action,next_state,reward,done):
        q_state = self.predict(state)
        q_next_state = self.predict(next_state)
        best_next_action = np.argmax(q_next_state)
        td_target = reward + np.invert(done).astype(np.float32)*self.discount_factor*q_next_state[best_next_action]
        td_error = td_target - q_state[action]

        tiles = self.get_tiles(state,action)
        for tile in tiles:
            self.w[tile] += self.learning_rate*td_error








def fixed_threshold_policy(state,warning_threshold_ttc = 8):
    # Issue a warning whenever the t to crash is below fixed_threshold_ttc  
    if state.time_to_crash < warning_threshold_ttc:
        action = WARN
    else:
        action = NOT_WARN
    return action

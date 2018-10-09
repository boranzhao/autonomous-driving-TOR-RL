import numpy as np
import random
from tiles3 import IHT, tiles

from driving_env import WARN, NOT_WARN

from collections import namedtuple 
import h5py

# Tile_Paras = namedtuple('Tile_Paras',['maxSize','num_tilings','num_grids','feature_vec_zero',''])

# utility functions
def get_parameterXfeature(parameter_vec,tiles):
    # parameterXfeature = 0
    # for tile in tiles:
    #     parameterXfeature += parameter_vec[tile]

    parameterXfeature = np.sum(parameter_vec[tiles])

    return parameterXfeature

def get_tiles(iht,num_tilings,tile_scale,state,action=None):                
    # # use tile coding to construct the feature vector
    if action == None:
        mytiles = tiles(iht,num_tilings,state*tile_scale)
    else:
        mytiles = tiles(iht,num_tilings,state*tile_scale,[action])

    # feature_vec = self.feature_vec_init
    # feature_vec[mytiles] = 1
    return mytiles

class Q_Agent_LFA():
    """
    An agent for off-policy q learning with linear function approximation and tile coding for the features
    """
    def __init__(self,num_actions,state_bounds,tile_coding ={'maxSize':1024,'num_tilings':8,'num_grids':10},
                learning_rate = 0.01, discount_factor = 0.9,train_result_file = None):   
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.nA = num_actions
        # for using tile coding to construct the feature vector
        self.tile_coding = tile_coding
        self.maxSize = tile_coding['maxSize']
        self.iht = IHT(self.maxSize)  #initialize the hash table 
        self.num_tilings = tile_coding['num_tilings']
        self.tile_scale = tile_coding['num_grids']/(state_bounds[1]-state_bounds[0])
        self.feature_vec_zero = np.zeros(self.maxSize)

        if not train_result_file:
            self.w = np.zeros(self.maxSize) 
        else:
            hf = h5py.File(train_result_file,'r')
            trained_model = hf.get('trained_model')
            self.w = trained_model.get('w').value
            hf.close()

    def epsilon_greedy_policy(self,state, epsilon =0.1):
        # Get the probabilities for all the actions
        q_values_state = self.predict(state)
        best_action = np.argmax(q_values_state)
        action_probs = np.ones(self.nA, dtype=float) * epsilon / self.nA        
        action_probs[best_action] += (1.0 - epsilon)

        # Sample an action according to the probabilities
        return np.random.choice(np.arange(self.nA), p = action_probs)

    def select_action(self,state,epsilon):
        ############################ setting to 0 will not allow exploration #########
        return self.epsilon_greedy_policy(state,epsilon)


    def predict(self,state):
        """
        Give a state, predict the state-action values for all actions
        """
        q_s = np.zeros(self.nA)
        for a in range(self.nA):
            my_tiles = get_tiles(self.iht,self.num_tilings,self.tile_scale,state,a)
            q_s[a] = get_parameterXfeature(self.w,my_tiles)
        return q_s

    def update(self,state,action,next_state,reward,done,discount_gain=None):
        q_state = self.predict(state)
        q_next_state = self.predict(next_state)
        best_next_action = np.argmax(q_next_state)
        td_target = reward + np.invert(done).astype(np.float32)*self.discount_factor*q_next_state[best_next_action]
        td_error = td_target - q_state[action]

        tiles = get_tiles(self.iht,self.num_tilings,self.tile_scale,state,action)
        for tile in tiles:
            self.w[tile] += self.learning_rate*td_error

class Q_Lambda_LFA(Q_Agent_LFA):
    def __init__(self,num_actions,state_bounds,tile_coding ={'maxSize':1024,'num_tilings':8,'num_grids':10},
                learning_rate = 0.01, discount_factor = 0.9, lambda1 = 0.8,train_result_file = None):   
                
        super().__init__(num_actions,state_bounds,tile_coding =tile_coding,
                learning_rate = learning_rate, discount_factor = discount_factor,train_result_file = train_result_file)
                
        self.lambda1 = lambda1        
        if not train_result_file:
            self.e = np.zeros(len(self.w))  # eligibility trace 
        else:
            hf = h5py.File(train_result_file,'r')
            trained_model = hf.get('trained_model')
            self.e = trained_model.get('eligibility_trace').value

            hf.close()
    
    def update(self,state,action,next_state,reward,done,discount_gain=None):
        q_state = self.predict(state)
        q_next_state = self.predict(next_state)
        best_next_action = np.argmax(q_next_state)
        td_target = reward + np.invert(done).astype(np.float32)*self.discount_factor*q_next_state[best_next_action]
        td_error = td_target - q_state[action]

        tiles = get_tiles(self.iht,self.num_tilings,self.tile_scale,state,action)
        # for tile in tiles:
        #     self.e[tile] += 1
        self.e[tiles] +=1 

        # for debugging
        if np.abs(td_error)>0:
            tmp = 1

        self.w += self.learning_rate*td_error*self.e
        self.e *= self.discount_factor*self.lambda1
        return td_error       

class ActorCritic():
    def __init__(self,num_actions,state_bounds,tile_coding ={'maxSize':1024,'num_tilings':8,'num_grids':10},
                learning_rate_w = 0.01, learning_rate_theta = 0.01,lambda_w = 0.8,lambda_theta = 0.8,discount_factor = 0.9,train_result_file = None):   
        self.learning_rate_w = learning_rate_w
        self.learning_rate_theta = learning_rate_theta
        self.discount_factor = discount_factor
        self.lambda_w = lambda_w
        self.lambda_theta = lambda_theta

        self.nA = num_actions
        # for using tile coding to construct the feature vector
        self.tile_coding = tile_coding
        self.maxSize = tile_coding['maxSize']
        self.iht = IHT(self.maxSize)  #initialize the hash table 
        self.num_tilings = tile_coding['num_tilings']
        self.tile_scale = tile_coding['num_grids']/(state_bounds[1]-state_bounds[0])
        self.feature_vec_zero = np.zeros(self.maxSize)

        if not train_result_file:
            self.w = np.zeros(self.maxSize)         # parameter vector for value estimator
            self.theta = np.zeros(self.maxSize)     # parameter vector for value estimator
            self.e_w = np.zeros(self.maxSize)       # eligibility trace vector for w
            self.e_theta = np.zeros(self.maxSize)   # eligibility trace vector for theta
        else:
            hf = h5py.File(train_result_file,'r')
            trained_model = hf.get('trained_model')
            self.w = trained_model.get('w').value
            self.theta = trained_model.get('theta').value
            self.e_w = trained_model.get('e_w').value
            self.e_theta = trained_model.get('e_theta').value
            hf.close()

    def get_feature_vec(self,state,action=None):
        feature_vec = self.feature_vec_zero

        tiles = get_tiles(self.iht,self.num_tilings,self.tile_scale,state,action)
        feature_vec[tiles] = 1
        return feature_vec
    
    def predict_value(self, state):
        # predict the value of v(s) 
        mytiles = get_tiles(self.iht,self.num_tilings,self.tile_scale,state)
        v_s = get_parameterXfeature(self.w,mytiles)
        return v_s  

    def select_action(self,state):
        action_probs = self.predict_pi_s(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action

    def predict_pi_s(self, state):
        # predict the value of pi(a|s,theta) for all a
        pi_s = np.zeros(self.nA)
        for a in range(self.nA):
            mytiles = get_tiles(self.iht, self.num_tilings,self.tile_scale,state,a)
            pi_s[a] = np.exp(get_parameterXfeature(self.theta,mytiles))
        # normalization     
        pi_s = pi_s/sum(pi_s)

        return pi_s       
        
    def update(self,state,action,next_state,reward,done,discount_gain):
        # calculate TD error
        value_next = self.predict_value(next_state)
        td_target = reward +  np.invert(done).astype(np.float32)*self.discount_factor*value_next
        td_error = td_target - self.predict_value(state)

        # Calculate the gradient of the value function
        # and then update the eligibility trace vector e_w
        grad_v = self.get_feature_vec(state)    # gradient of the value function
        self.e_w += discount_gain*grad_v

        # Calculate the gradient of ln pi(a/s,theta) with respect to theta, which is x(s,a)-sum_b(pi(b/s,theta)*x(s,b))
        # and then update the eligiblity trace vector e_theta
        pi_s= self.predict_pi_s(state)
        grad_ln_pi = self.get_feature_vec(state,action) - sum([pi_s[b]*self.get_feature_vec(state,b) for b in range(self.nA)])
        self.e_theta += discount_gain*grad_ln_pi

        # Update the parameters 
        self.w += self.learning_rate_w*td_error*self.e_w
        self.theta += self.learning_rate_theta*td_error*self.e_theta

        # Update the eligibility trace vectors
        self.e_w *= self.discount_factor*self.lambda_w
        self.e_theta *= self.discount_factor*self.lambda_theta
        
        return td_error

def fixed_threshold_policy(state,warning_threshold_ttc = 8):
    # Issue a warning whenever the t to crash is below fixed_threshold_ttc  
    if state.time_to_crash < warning_threshold_ttc:
        action = WARN
    else:
        action = NOT_WARN
    return action

def modify_action_to_enforce_safety(state,action,min_ttc_for_safety=5):
    constraint_active = False
    if state[0]<= min_ttc_for_safety:
        # The warning system should warn but did not; therefore, consider this as a false positive warning
        if action == NOT_WARN:
            constraint_active = True 
        return WARN,constraint_active
    else:
        return action,constraint_active


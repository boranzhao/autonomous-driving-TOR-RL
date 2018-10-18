import numpy as np
from scipy.special import logsumexp
import random
from tiles3 import IHT, tiles

from driving_env import WARN, NOT_WARN

from collections import namedtuple 
import h5py

def normalize_state(state,state_bounds):
    normalized_state = np.zeros(len(state))
    normalized_state[0] = (state[0]-state_bounds[0][0])/(state_bounds[1][0]-state_bounds[0][0])
    normalized_state[1] = (state[1]-state_bounds[0][1])/(state_bounds[1][1]-state_bounds[0][1])
    return normalized_state 
                        
def normalize_action(action,num_actions):
    return float(action)/(num_actions-1)


class Q_Agent_LFA():
    """
    An agent for off-policy q learning with linear function approximation and tile coding or Fourier basis for the features
    """
    def __init__(self,num_actions,state_bounds,n_basis=3,
                learning_rate = 0.01, discount_factor = 0.9,train_result_file = None):   
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.nA = num_actions
        self.n_basis = n_basis   # order of Fourier basis
        self.w = np.zeros((self.n_basis+1)**3) # 3 is the dimension of the (s,a) space
        self.state_bounds = state_bounds
        self.name = "q-learning"
        if train_result_file:
            hf = h5py.File(train_result_file,'r')
            trained_model = hf.get('trained_model')
            self.w = trained_model.get('w').value
            hf.close()

    def get_feature_vec(self,state, action):
        # use Fourier basis to construct the feature. With the dimension of the space as k, order of Fourier basis as n, there are (n+1)^k different features
        # normalize the state and action so that they are in the range [0,1]        
        sa_index = np.append(normalize_state(state,self.state_bounds),normalize_action(action,self.nA))              
        sc_product = [sa_index.dot((sx,sy,a)) for sx in range(self.n_basis+1) for sy in range(self.n_basis+1) for a in range(self.n_basis+1)]
        feature_vec = np.cos(np.pi*np.array(sc_product))   
        return feature_vec

    def epsilon_greedy_policy(self,state, epsilon =0.1):
        # Get the probabilities for all the actions
        q_values_state = self.predict(state)
        best_action = np.argmax(q_values_state)
        action_probs = np.ones(self.nA, dtype=float) * epsilon / self.nA        
        action_probs[best_action] += (1.0 - epsilon)

        # Sample an action according to the probabilities
        return np.random.choice(np.arange(self.nA), p = action_probs)

    def select_action(self,state,epsilon):
        ############################ setting to 0 will de-activate the exploration #########
        return self.epsilon_greedy_policy(state,epsilon)


    def predict(self,state):
        """
        Give a state, predict the state-action values for all actions
        """
        q_s = np.zeros(self.nA)
        for a in range(self.nA):        
            q_s[a] = self.w.dot(self.get_feature_vec(state,a))
        return q_s

    def update(self,state,action,next_state,reward,done,discount_gain=None):
        q_state = self.predict(state)
        q_next_state = self.predict(next_state)
        best_next_action = np.argmax(q_next_state)
        td_target = reward + np.invert(done).astype(np.float32)*self.discount_factor*q_next_state[best_next_action]
        td_error = td_target - q_state[action]
    
        self.w += self.learning_rate*td_error

class Q_Lambda_LFA(Q_Agent_LFA):
    def __init__(self,num_actions,state_bounds,n_basis = 3, learning_rate = 0.01, 
                discount_factor = 0.9, lambda1 = 0.8,train_result_file = None):   
                
        super().__init__(num_actions,state_bounds,n_basis = n_basis,
                learning_rate = learning_rate, discount_factor = discount_factor,train_result_file = train_result_file)
        self.name = "q-lambda"      
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

        self.e = self.discount_factor*self.lambda1*self.e + self.get_feature_vec(state,action)
        # for debugging
        if np.abs(td_error)>0:
            tmp = 1
        self.w += self.learning_rate*td_error*self.e

        # reset the eligibility trace at the end of an episode
        if done:
            self.e = np.zeros(len(self.e))

        return td_error       

class ActorCritic():
    def __init__(self,num_actions,state_bounds,n_basis = 3,
                learning_rate_w = 0.01, learning_rate_theta = 0.01,lambda_w = 0.8,lambda_theta = 0.8,discount_factor = 0.9,train_result_file = None):   
        self.learning_rate_w = learning_rate_w
        self.learning_rate_theta = learning_rate_theta
        self.discount_factor = discount_factor
        self.lambda_w = lambda_w
        self.lambda_theta = lambda_theta
        self.state_bounds = state_bounds
        self.n_basis = n_basis
        self.nA = num_actions
        self.name = "actor-critic"

        if not train_result_file:
            self.w = np.zeros((self.n_basis+1)**2)        # k=2 b/c the dimension of the state space is 2
            self.theta = np.zeros((self.n_basis+1)**3)      #  k =3
            self.e_w = np.zeros(len(self.w))             # eligibility trace vector for w
            self.e_theta = np.zeros(len(self.theta))    # eligibility trace vector for theta
        else:
            hf = h5py.File(train_result_file,'r')
            trained_model = hf.get('trained_model')
            self.w = trained_model.get('w').value
            self.theta = trained_model.get('theta').value
            self.e_w = trained_model.get('eligibility_trace_w').value
            self.e_theta = trained_model.get('eligibility_trace_theta').value
            self.lambda_theta = trained_model.get('lambda_theta').value
            self.lambda_w = trained_model.get('lambda_w').value
            self.discount_factor = trained_model.get('discount_factor').value
            self.learning_rate_theta = trained_model.get('learning_rate_theta').value
            self.learning_rate_w = trained_model.get('learning_rate_w').value
            self.n_basis = trained_model.get('Fourier_basis_order').value

            hf.close()


    def get_feature_vec(self,state, action=None):
        # use Fourier basis to construct the feature. With the dimension of the space as k, order of Fourier basis as n, there are (n+1)^k different features
        # normalize the state and action so that they are in the range [0,1]        
        if action==None:
            s_index = normalize_state(state,self.state_bounds)              
            sc_product = [s_index.dot((sx,sy)) for sx in range(self.n_basis+1) for sy in range(self.n_basis+1)]
        else:
            sa_index = np.append(normalize_state(state,self.state_bounds),normalize_action(action,self.nA))              
            sc_product = [sa_index.dot((sx,sy,a)) for sx in range(self.n_basis+1) for sy in range(self.n_basis+1) for a in range(self.n_basis+1)]
        
        feature_vec = np.cos(np.pi*np.array(sc_product))   
        return feature_vec
    
    def predict_value(self, state):
        # predict the value of v(s) 
        return self.w.dot(self.get_feature_vec(state))

    def select_action(self,state,epsilon=None):
        action_probs = self.predict_pi_s(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def predict_pi_s(self, state):
        # predict the value of pi(a|s,theta) for all a
        pi_s = np.zeros(self.nA)
        for a in range(self.nA):
            pi_s[a] = self.theta.dot(self.get_feature_vec(state,a))
        # normalization     
        # pi_s = pi_s/sum(pi_s)
        
        
        # use a different way to avoid overflow/underflow 
        pi_s = pi_s - logsumexp(pi_s)
        pi_s = np.exp(pi_s)

        # to enforce sum(pi_s)=1 even under numerical error
        pi_s = pi_s/sum(pi_s)
        # print(pi_s)
        return pi_s       
        
    def update(self,state,action,next_state,reward,done,discount_gain):
        # calculate TD error
        value_next = self.predict_value(next_state)
        td_target = reward +  np.invert(done).astype(np.float32)*self.discount_factor*value_next
        td_error = td_target - self.predict_value(state)
        # for debug 
        # print("td_error:{:.3f}".format(td_error))
        # Calculate the gradient of the value function
        # and then update the eligibility trace vector e_w
        grad_v = self.get_feature_vec(state)    # gradient of the value function
        self.e_w =self.discount_factor*self.lambda_w*self.e_w+ discount_gain*grad_v

        # Calculate the gradient of ln pi(a/s,theta) with respect to theta, which is x(s,a)-sum_b(pi(b/s,theta)*x(s,b))
        # and then update the eligiblity trace vector e_theta
        pi_s= self.predict_pi_s(state)
        grad_ln_pi = self.get_feature_vec(state,action) - sum([pi_s[b]*self.get_feature_vec(state,b) for b in range(self.nA)])
        self.e_theta = self.discount_factor*self.lambda_theta*self.e_theta+discount_gain*grad_ln_pi

        # Update the parameters 
        self.w += self.learning_rate_w*td_error*self.e_w
        self.theta += self.learning_rate_theta*td_error*self.e_theta 

        # reset the eligibility trace at the end of an episode
        if done:
            self.e_w = np.zeros(len(self.e_w))
            self.e_theta = np.zeros(len(self.e_theta))  

        return td_error

def fixed_threshold_policy(state,warning_threshold_ttc = 8):
    # Issue a warning whenever the t to crash is below fixed_threshold_ttc  
    if state.time_to_crash < warning_threshold_ttc:
        action = WARN
    else:
        action = NOT_WARN
    return action

def modify_action_to_enforce_safety(state,action,min_ttc_for_safety):
    constraint_active = False
    if state[0]<= min_ttc_for_safety:
        # The warning system should warn but did not; therefore, consider this as a false positive warning
        if action == NOT_WARN:
            constraint_active = True 
        return WARN,constraint_active
    else:
        return action,constraint_active


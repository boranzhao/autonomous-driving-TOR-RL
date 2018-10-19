from driving_env import Driver,DrivingEnv, Car
from driving_env import WARN, NOT_WARN, DriverMode,CarDrivingMode
from agent import Q_Agent_LFA,Q_Lambda_LFA, ActorCritic,fixed_threshold_policy,modify_action_to_enforce_safety
from gym.wrappers import Monitor
from lib.recorder import Recorder

import itertools
import numpy as np

import time
import os
import datetime

import h5py

from collections import namedtuple
import matplotlib.pyplot as plt

from lib.utilities import EpisodeStats,plot_episode_stats,save_train_results


# setting for how numpy control overflow/divide by zero issues
np.seterr(divide='raise', over='print', invalid = 'raise')

def generate_clip_state_function(state_bounds):
    def clip_state(state,state_bounds = state_bounds):
        return np.clip(state,state_bounds[0],state_bounds[1])
    return clip_state


def train_agent(env,agent,num_episodes,clip_state,enforce_safety,min_ttc_for_safety):
    episode_stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes,dtype=np.int16),
        episode_rewards = np.zeros(num_episodes),
        episode_crashes = np.zeros(num_episodes,dtype=np.int16),
        episode_near_crashes = np.zeros(num_episodes,dtype=np.int16),
        episode_FP_warnings = np.zeros(num_episodes,dtype=np.int16),
        episode_FN_warnings = np.zeros(num_episodes,dtype=np.int16),
        episode_td_error_rms = np.zeros(num_episodes)
        )
    accumulated_reward = 0

    discount_factor = agent.discount_factor    
    for i_episode in range(num_episodes):
        state = env.reset()
        state = clip_state(state)
        
        env.render()
        time.sleep(1)

        discount_gain = 1
        action = NOT_WARN
        td_errors = np.zeros(10000)
        i_td_error=0
        for t in itertools.count():
            # Render (comment this if the running speed is low)
            if t%3 == 0: 
                env.render()

            # RL agent
            # Select an action based on epsilon-greedy policy 
            # Should only be impelmented when car is in autonomous driving mode
            if env.car.driving_mode == CarDrivingMode.AUTONOMOUS:                
                # decrease the epsilon value, associated with the degree of exploration
                epsilon = max(0.2- i_episode/100,0.05)
                action = agent.select_action(state,epsilon=epsilon)
                if enforce_safety:
                    modified_action,constraint_active = modify_action_to_enforce_safety(state,action,min_ttc_for_safety) 
                    if constraint_active:
                        if not env.driver.warning_acknowledged: 
                            driver.false_negative_warnings +=1
                else:
                    modified_action = action

                # Fixed-threshold strategy  (just for comparison; comment this when using RL)
                # modified_action = fixed_threshold_policy(state[0],warning_threshold_ttc=9)

                next_state,reward,done,game_over = env.step(modified_action)
                next_state = clip_state(next_state)
                td_error = 0
                          
                td_error = agent.update(state,action,next_state,reward,done,discount_gain)

                td_errors[i_td_error] = td_error
                i_td_error +=1

                sub_discount_gain = 1

                discount_gain = discount_gain*discount_factor
            else:
                # reward cannot be updated until the end of driver intervention to deal with the critical situation
                next_state,reward,done,game_over = env.step(NOT_WARN)
                next_state = clip_state(next_state)

                if env.driver.driver_mode in [DriverMode.SWERVE_TO_LEFT, DriverMode.EMERGENCY_BRAKE]:   
                    # Accumulate the reward in each step during driver intervention            
                    accumulated_reward += sub_discount_gain*reward
                    sub_discount_gain = sub_discount_gain*discount_factor
                else:
                    if done:
                        accumulated_reward += sub_discount_gain*reward
                        # At the end of driver intervention, should perform an update based on the accumated reward.                         
                        # clip the accumulated reward
                        accumulated_reward = np.clip(accumulated_reward,-10,10)
                        # Here it does not matter what next_state is as long as done is True
                        ############################################
                        td_error = agent.update(state,action,next_state,accumulated_reward,True,discount_gain)
                        td_errors[i_td_error] = td_error
                        i_td_error +=1
                        # Reset the accumulated reward
                        accumulated_reward = 0     

                        discount_gain = discount_gain*discount_factor       

            episode_stats.episode_rewards[i_episode] += discount_gain*reward       

            if done:
                pass
            if game_over:
                episode_stats.episode_lengths[i_episode] = t
                episode_stats.episode_crashes[i_episode] = env.num_crashes   
                episode_stats.episode_near_crashes[i_episode] = env.num_near_crashes
                episode_stats.episode_FP_warnings[i_episode] = env.driver.false_positive_warnings
                episode_stats.episode_FN_warnings[i_episode] = env.driver.false_negative_warnings
                episode_stats.episode_td_error_rms[i_episode] = np.sqrt(np.mean(np.array(td_errors[0:i_td_error])**2))
                
                print("Episode:{}, reward:{:.2f}, crashes:{:d}, near_crashes:{:d}, toal_warnings:{:d}, FP_warnings:{:d}, FN_warnings:{:d}".format(
                i_episode,episode_stats.episode_rewards[i_episode],
                episode_stats.episode_crashes[i_episode],episode_stats.episode_near_crashes[i_episode],
                env.driver.total_warnings,
                episode_stats.episode_FP_warnings[i_episode],episode_stats.episode_FN_warnings[i_episode]))

                env.render()
                break

            # Only need to update the state in autonomous driving mode
            if env.car.driving_mode == CarDrivingMode.AUTONOMOUS:
                state = next_state            
    return episode_stats

## Main function 
if __name__== "__main__":
    sample_time = 0.3             # sample time for observing the state and executing an action

    train_result_file = None 
    train_result_file= 'train-result-actor-critic/train_results.h5py'   # Comment this line if do not want to use a trianed model 

    # Important parameters that will influence the training result
    min_ttc_for_safety = 6                  # minimum ttc allowed before enforcing a warning
    driver_response_time_bounds = [2.5,2.6]
    driver_maximum_intervention_ttc = 6     # Driver will not intervene if ttc is larger than this value

    if train_result_file != None:   
        # load and overwrite some parameters 
        hf = h5py.File(train_result_file,'r')
        trained_model = hf.get('trained_model')
        min_ttc_for_safety = trained_model.get('min_ttc_for_safety').value
        driver_paras = hf.get('driver')
        driver_response_time_bounds = driver_paras.get('response_time_bounds').value
        driver_maximum_intervention_ttc = driver_paras.get('maximum_intervention_ttc').value

    # Initialize the traffic environment
    car = Car(speed = 0)
    driver = Driver(brake_intensity=0.5,
                    accel_intensity=0,
                    response_time_bounds= driver_response_time_bounds,
                    maximum_intervention_ttc = driver_maximum_intervention_ttc)

    env = DrivingEnv(car,driver,sample_time=sample_time,always_penalize_warning=True)

    # Creat the directory for recording videos
    monitor_folder = os.path.join(os.getcwd(),"monitor-"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
    # monitor_folder = os.path.join(os.getcwd(),"recorder")

    if not os.path.exists(monitor_folder):
        os.makedirs(monitor_folder)

    # Add env Monitor wrapper
    env = Recorder(env, directory= monitor_folder, video_callable=lambda count: count % 1 == 0, resume = True)

    state_bounds = np.array([[0,0],[15,1]])
    clip_state = generate_clip_state_function(state_bounds = state_bounds)
    # Q lambda does not work very well
    # agent = Q_Lambda_LFA(num_actions=2,state_bounds=state_bounds,n_basis = 3,
    #                 learning_rate=0.005,discount_factor=1,lambda1=0.95) # train_result_file="monitor-2018-08-13-2106-good-fp-05-fn-05/train_results.h5py") #"monitor-2018-08-03-2245/train_results.h5py"

    # Actor critic with eligibility trace now works  well
    agent = ActorCritic(num_actions=2,state_bounds=state_bounds,
                        n_basis = 5, learning_rate_w=0.002,learning_rate_theta=0.002, discount_factor=1,
                        lambda_w = 0.95,lambda_theta = 0.95, train_result_file = train_result_file)

    episode_stats = train_agent(env,agent,2,clip_state,enforce_safety=True,min_ttc_for_safety=min_ttc_for_safety)

    # save train results
    train_results_file = os.path.join(monitor_folder,"train_results.h5py")
    save_train_results(train_results_file,agent,episode_stats,driver,min_ttc_for_safety)

    # show the figures
    # plot the episode_stats and save corresponding figures. 
    fig1,fig2,fig3 = plot_episode_stats(episode_stats)

    plt.figure(fig1.number)
    plt.savefig(os.path.join(monitor_folder,"episode_reward_length_tdError.png"))
    plt.figure(fig2.number)
    plt.savefig(os.path.join(monitor_folder,"episode_crashes.png"))
    plt.figure(fig3.number)
    plt.savefig(os.path.join(monitor_folder,"episode_false_warnings.png"))
    plt.show()

    exit()




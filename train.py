from car_env import Driver,TrafficEnv, Car, StatePoint, ActionPoint, StepInfo
from car_env import WARN, NOT_WARN, BRAKE, NONE, ACCEL, IGNORE, ACKNOWLEDGE, DriverMode,CarDrivingMode
from agent import Q_Agent_LFA,Q_Lambda_LFA, fixed_threshold_policy
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

from lib.utilities import EpisodeStats,plot_episode_stats

def modify_action_to_enforce_safety(state,action,min_ttc_for_safety=5):
    if state[0]<= min_ttc_for_safety:
        return WARN
    else:
        return action 


def train_agent(env,agent,num_episodes,enforce_safety = False):
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
        env.render()
        time.sleep(1)

        discount_gain = 1
        action = NOT_WARN
        td_errors = []
        for t in itertools.count():
            # Render
            env.render()

            # Fixed-threshold strategy
            # action = fixed_threshold_policy(env.state,warning_threshold_ttc=10)

            # RL agent
            # Select an action based on epsilon-greedy policy 
            # Should only be impelmented when car is in autonomous driving mode
            if env.car.driving_mode == CarDrivingMode.AUTONOMOUS:
                action = agent.epsilon_greedy_policy(state,epsilon=0.1)
                if enforce_safety:
                    modified_action = modify_action_to_enforce_safety(state,action) 
                else:
                    modified_action = action

                next_state,reward,done,game_over = env.step(modified_action)
                          
                td_error = agent.update(state,action,next_state,reward,done)

                td_errors.append(td_error)

                sub_discount_gain = 1

                discount_gain = discount_gain*discount_factor
            else:
                # reward cannot be updated until the end of driver intervention to deal with the boundary
                next_state,reward,done,game_over = env.step(NOT_WARN)

                if env.driver.driver_mode in [DriverMode.SWERVE_TO_LEFT, DriverMode.EMERGENCY_BRAKE]:   
                    # Accumulate the reward in each step during driver intervention            
                    accumulated_reward += sub_discount_gain*reward
                    sub_discount_gain = sub_discount_gain*discount_factor
                else:
                    if done:
                        accumulated_reward += sub_discount_gain*reward
                        # At the end of driver intervention, should perform an update based on the accumated reward.                         
                        # clip the accumulated reward
                        #reward = np.clip(accumulated_reward,-10,0)
                        # Here it does not matter what next_state is as long as done is True
                        td_error = agent.update(state,action,next_state,accumulated_reward,True)
                        td_errors.append(td_error)
                        # Reset the accumulated reward
                        accumulated_reward = 0     

                        discount_gain = discount_gain*discount_factor         

            episode_stats.episode_rewards[i_episode] += discount_gain*reward       

            # Simulation information in each step
            # print("time:{:.2f}, speed:{:.1f}, Lane:{},  RD:{:.1f}m, brake:{:.2f}, steer:{:.2f}, car_mode:{}, driver_mode:{}".format(t*sample_time_basic, 
            #     env.car.speed,env.car.lane.name,env.relative_distance,env.car.brake_intensity,env.car.steer_intensity,env.car.driving_mode.name[:6],env.driver.driver_mode.name))

            if done:
                episode_stats.episode_lengths[i_episode] = t
                episode_stats.episode_crashes[i_episode] = env.num_crashes   
                episode_stats.episode_near_crashes[i_episode] = env.num_near_crashes
                episode_stats.episode_FP_warnings[i_episode] = env.driver.false_positive_warnings
                episode_stats.episode_FN_warnings[i_episode] = env.driver.false_negative_warnings
                episode_stats.episode_td_error_rms[i_episode] = np.sqrt(np.mean(np.array(td_errors)**2))
                
            if game_over:
                print("Episode:{}, reward:{:.2f}, crashes:{:d}, near_crashes:{:d}, FP_warnings:{:d}, FN_warnings:{:d}".format(
                i_episode,episode_stats.episode_rewards[i_episode],
                episode_stats.episode_crashes[i_episode],episode_stats.episode_near_crashes[i_episode],
                episode_stats.episode_FP_warnings[i_episode],episode_stats.episode_FN_warnings[i_episode]))

                env.render()
                break

            # Only need to update the state in autonomous driving mode
            if env.car.driving_mode == CarDrivingMode.AUTONOMOUS:
                state = next_state
            
    return episode_stats

def save_train_results(train_results_file,agent,episode_stats):
    hf = h5py.File(train_results_file,'w')
    g1 = hf.create_group('trained_model')
    g1.create_dataset('lambda1',data=agent.lambda1)
    g1.create_dataset('w',data=agent.w)
    g1.create_dataset('discount_factor',data=agent.discount_factor)
    g1.create_dataset('learning_rate',data=agent.learning_rate)
    g11 = g1.create_group('tile_coding')
    g11.create_dataset("maxSize",data = agent.tile_coding["maxSize"])
    g11.create_dataset("num_tilings",data= agent.tile_coding["num_tilings"])
    g11.create_dataset("num_grids",data= agent.tile_coding["num_grids"])
    g1.create_dataset('eligibility_trace',data=agent.e)


    g2 = hf.create_group('episode_stats')
    g2.create_dataset('episode_rewards',data = episode_stats.episode_rewards)
    g2.create_dataset('episode_lengths',data = episode_stats.episode_lengths)
    g2.create_dataset('episode_td_error_rms',data = episode_stats.episode_td_error_rms)
    g2.create_dataset('episode_crashes',data = episode_stats.episode_crashes)
    g2.create_dataset('episode_near_crashes',data = episode_stats.episode_near_crashes)
    g2.create_dataset('episode_FP_warnings',data = episode_stats.episode_FP_warnings)
    g2.create_dataset('episode_FN_warnings',data = episode_stats.episode_FN_warnings)
    hf.close()

#
# DriverStats = namedtuple('DriverStats',['brake','warning_acknowledged','brake_intensity','accel_intensity','attention_level'])
# SimulationStats = namedtuple('SimulationStats',['relative_distance','relative_velocity','time_to_collision','action', 'driver_stats','rain'])

# utilities the simulation results
# Should plot the following variables:
# relative distance, relative velocity, t to crash, rain, attentive state, 
# FP & FN

sample_time_basic = 0.5          # basic sample t for observing the state
sample_time_action = 0.5         # sample t for executing an action (WARN or NOT_WARN)
# sample_time_learning = 10         # sample t for learning 

action_frequency = int(sample_time_action/sample_time_basic)
# learning_frequency = int(sample_time_learning/sample_time_basic)

epsilon = 0.1

# # Initialize the memory for storing the simulation 
# driver_stats = DriverStats(np.zeros(num_steps,dtype=bool),np.zeros(num_steps,dtype=bool), 
#                 np.zeros(num_steps),np.zeros(num_steps),np.ones(num_steps,dtype=np.int8))
# simulation_stats = SimulationStats(np.zeros(num_steps),np.zeros(num_steps),np.zeros(num_steps),
#                     np.zeros(num_steps,dtype=bool),driver_stats,np.zeros(num_steps,dtype=bool))

# Initialize the traffic environment
car = Car(speed = 0)
driver = Driver(brake_intensity=0.5,
                accel_intensity=0,
                response_time_bounds=[1.5,2.5],
                maximum_intervention_ttc = 6,
                comfort_braking_distance= 50)
env = TrafficEnv(car,driver,always_penalize_warning=False)

# Creat the directory for monitoring
  # Record videos
monitor_folder = os.path.join(os.getcwd(),"monitor-"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
# monitor_folder = os.path.join(os.getcwd(),"recorder")

if not os.path.exists(monitor_folder):
    os.makedirs(monitor_folder) 

# Add env Monitor wrapper
# env = Recorder(env, directory= monitor_folder, video_callable=lambda count: count % 1 == 0, resume = True)

agent = Q_Lambda_LFA(num_actions=2,state_bounds=np.array([[0,0],[15,1]]),
                tile_coding ={'maxSize':1024,'num_tilings':8,'num_grids':10},
                learning_rate=0.04,discount_factor=0.998,lambda1=0.8, train_result_file="monitor-2018-08-03-2245/train_results.h5py") #"monitor-2018-08-03-2245/train_results.h5py"

episode_stats = train_agent(env,agent,10,enforce_safety=False)


# save train results
train_results_file = os.path.join(monitor_folder,"train_results.h5py")
save_train_results(train_results_file,agent,episode_stats)

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






for t in range(num_steps):
    # render
    env.render()

    # Observe and store the states
    state,brake,done = env.update_state(action)   

    # Record the simulation data 
    driver_stats.brake[t] = brake
    driver_stats.warning_acknowledged[t] = env.car.driver.warning_acknowledged
    driver_stats.brake_intensity[t] =  env.car.driver.brake_intensity
    driver_stats.accel_intensity[t] =  env.car.driver.accel_intensity
    driver_stats.attention_level[t] = state.driver_attention_level

    simulation_stats.relative_distance[t]  = env.relative_distance
    simulation_stats.relative_velocity[t]  = env.relative_velocity
    simulation_stats.time_to_collision[t]  = state.time_to_collision
    simulation_stats.action[t]  = action
    simulation_stats.rain[t]  = state.rain

    if done:
        env.reset()
        state,brake,done = env.update_state()
            
    # Action 
    if t % action_frequency == 0:
        action = fixed_threshold_policy(state)
        env.step(action)



# Plot the simulation results 

simu_time = np.array(range(num_steps))*sample_time_basic

plt.figure()
plt.subplot(3,2,1)
plt.plot(simu_time, simulation_stats.relative_distance)
plt.ylabel('RD [m]')
plt.xlabel('Time [s]')
plt.subplot(3,2,2)
plt.plot(simu_time,simulation_stats.relative_velocity)
plt.ylabel('RV [m/s]')
plt.xlabel('Time [s]')
plt.subplot(3,2,3)
plt.plot(simu_time,simulation_stats.time_to_collision)
plt.ylabel('Time to crash [s]')
plt.xlabel('Time [s]')
plt.subplot(3,2,5)
plt.plot(simu_time,simulation_stats.action)
plt.ylabel('Warning? ')
plt.xlabel('Time [s]')
plt.subplot(3,2,4)
plt.plot(simu_time,simulation_stats.rain)
plt.ylabel('Rain? ')
plt.xlabel('Time [s]')


plt.figure()
plt.subplot(3,2,1)
plt.plot(simu_time, simulation_stats.driver_stats.warning_acknowledged)
plt.ylabel('Warning acknowledged?')

plt.subplot(3,2,2)
plt.plot(simu_time, simulation_stats.driver_stats.brake)
plt.ylabel('Driver brake?')
plt.xlabel('Time [s]')
plt.subplot(3,2,3)
plt.plot(simu_time, simulation_stats.driver_stats.attention_level)
plt.ylabel('Attention level')
plt.xlabel('Time [s]')
plt.subplot(3,2,4)
plt.plot(simu_time, simulation_stats.driver_stats.brake_intensity)
plt.ylabel('Brake intensity [0-1]')
plt.xlabel('Time [s]')
plt.subplot(3,2,5)
plt.plot(simu_time, simulation_stats.driver_stats.accel_intensity)
plt.ylabel('Accel. intensity [0-1]')

plt.xlabel('Time [s]')
plt.show()



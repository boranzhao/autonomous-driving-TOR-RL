from car_env import Driver,TrafficEnv, Car, StatePoint, ActionPoint, StepInfo
from car_env import WARN, NOT_WARN, BRAKE, NONE, ACCEL, IGNORE, ACKNOWLEDGE, DriverMode,CarDrivingMode
from agent import RL_Agent, fixed_threshold_policy
from gym.wrappers import Monitor
from recorder import Recorder

import itertools
import numpy as np

import time
import os
import datetime

from collections import namedtuple

import matplotlib.pyplot as plt

DriverStats = namedtuple('DriverStats',['brake','warning_acknowledged','brake_intensity','accel_intensity','attention_level'])
SimulationStats = namedtuple('SimulationStats',['relative_distance','relative_velocity','time_to_collision','action', 'driver_stats','rain'])

def test_speed_tracking_autonomous(env):
    env.driver.activate_autonomous_mode(env.car)
    # env.car = Car(speed =0, car_driving_mode=CarDrivingMode.AUTONOMOUS)
    env.car.set_target_speed(100)
    for t in range(150):  
        env.render()
        env.update_state(NOT_WARN)  
        print("time:{:.2f}, speed:{:.2f}, accel:{:.2f},brake:{:.2f},driving_mode:{}".format(t*sample_time_basic, env.car.speed,env.car.accel_intensity,env.car.brake_intensity,env.car.car_driving_mode.name[:6]))

        if t== 80:
            env.car.set_target_speed(40)        
        if t==120:
            env.car.set_target_speed(0)

def test_drive_to_speed_driver(env):
    env.reset()
    env.leading_car.set_target_speed(100)
    driver.driver_mode = DriverMode.DRIVE_TO_SPEED

    for t in range(100):
        env.render()
        env.update_state(NOT_WARN)        
        print("time:{:.2f}, speed:{:.1f}, leading speed:{:.1f}, RV:{:.1f}, RD:{:.1f}m, accel:{:.2f}, brake:{:.2f}, car_mode:{}, driver_mode:{}".format(t*sample_time_basic, 
            env.car.speed, env.leading_car.speed,env.relative_velocity,env.relative_distance,env.car.accel_intensity,env.car.brake_intensity,env.car.car_driving_mode.name[:6],env.driver.driver_mode.name))
        if t == 40:
            # Test autonomous driving mode 
            driver.activate_autonomous_mode(car)

def test_follow_with_safe_distance(env):
    # Test drive_to_speed -> follow_with_safe_distance -> drive_to_speed, pre-requisite: driver is attentive
    env.reset()
    # driver.make_driver_distracted()
    env.driver.driver_mode = DriverMode.DRIVE_TO_SPEED
    env.leading_car.set_target_speed(45)
    for t in range(100):
        env.render()
        state,brake,done = env.update_state(NOT_WARN)
        print("time:{:.2f}, speed:{:.1f}, leading speed:{:.1f}, RV:{:.1f}, RD:{:.1f}m, accel:{:.2f}, brake:{:.2f}, car_mode:{}, driver_mode:{}".format(t*sample_time_basic, 
            env.car.speed,env.leading_car.speed,env.relative_velocity,env.relative_distance,env.car.accel_intensity,env.car.brake_intensity,env.car.car_driving_mode.name[:6],env.driver.driver_mode.name))
        if done:
            env.reset()
        if t ==50:
            env.leading_car.set_target_speed(70)

def test_complete(env):
    """
    
    """   
    episodes = 1
    for i_episode in range(episodes):
        env.reset()
        env.driver.driver_mode = DriverMode.DRIVE_TO_SPEED
        action = NOT_WARN

        for t in itertools.count():
            # Render
            env.render()

            # Fixed-threshold strategy
            action = fixed_threshold_policy(env.state,warning_threshold_ttc=10)

            # RL agent

            # Execute a step  
            state,reward,done,near_crash = env.step(action)
            #  state,driver_action,done,near_crash= env.update_state(action)            

            # if abs(car.speed - driver.target_speed)< 0.5 and driver.driver_mode == DriverMode.DRIVE_TO_SPEED 
            #     and car.lane == :
            #     # Test autonomous driving mode 
            #     env.driver.activate_autonomous_mode(car)

            print("time:{:.2f}, speed:{:.1f}, Lane:{},  RD:{:.1f}m, brake:{:.2f}, steer:{:.2f}, car_mode:{}, driver_mode:{}".format(t*sample_time_basic, 
                env.car.speed,env.car.lane.name,env.env.relative_distance,env.car.brake_intensity,env.car.steer_intensity,env.car.car_driving_mode.name[:6],env.driver.driver_mode.name))

            if done:
                break

simu_time = 60*20

# plotting the simulation results
# Should plot the following variables:
# relative distance, relative velocity, t to crash, rain, attentive state, 
# FP & FN



sample_time_basic = 0.2          # basic sample t for observing the state
sample_time_action = 0.2         # sample t for executing an action (WARN or NOT_WARN)
sample_time_learning = 10         # sample t for learning 

action_frequency = int(sample_time_action/sample_time_basic)
learning_frequency = int(sample_time_learning/sample_time_basic)

epsilon = 0.1
num_steps = int(simu_time/sample_time_basic)+1

# Initialize the memory for storing the simulation 
driver_stats = DriverStats(np.zeros(num_steps,dtype=bool),np.zeros(num_steps,dtype=bool), 
                np.zeros(num_steps),np.zeros(num_steps),np.ones(num_steps,dtype=np.int8))
simulation_stats = SimulationStats(np.zeros(num_steps),np.zeros(num_steps),np.zeros(num_steps),
                    np.zeros(num_steps,dtype=bool),driver_stats,np.zeros(num_steps,dtype=bool))

# Initialize the traffic environment
car = Car(speed = 0)
driver = Driver(brake_intensity=0.5,
                accel_intensity=0,
                response_time_bounds=[1.5,2.5],
                maximum_intervention_ttc = 5,
                comfort_braking_distance= 50)
env = TrafficEnv(car,driver)


# Creat the directory for monitoring
  # Record videos
monitor_folder = os.path.join(os.getcwd(),"monitor-"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
# monitor_folder = os.path.join(os.getcwd(),"recorder")

if not os.path.exists(monitor_folder):
    os.makedirs(monitor_folder) 
  # Add env Monitor wrapper
env = Recorder(env, directory= monitor_folder, video_callable=lambda count: count % 1 == 0, resume = True)

# Test driver function: drive to speed 
# test_drive_to_speed_driver(env)

# Test driver function: follow with safe distance
# Procedure involved: drive_to_speed -> follow_with_safe_distance -> drive_to_speed, pre-requisite: driver is attentive
# test_follow_with_safe_distance(env)

# Test distracted-> warning -> warning_acknowledged = true -> perceive -> distracted or follow_with_safe_distance
test_complete(env)

exit()



################### NOT USED AT THIS MOMENT ################
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



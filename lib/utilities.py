# -*- coding: utf-8 -*-
"""
Created on 07/27/2018

@author: Boran
"""

import matplotlib
import numpy as np
import pandas as pd

import h5py

from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards",
    "episode_crashes","episode_near_crashes","episode_FP_warnings","episode_FN_warnings","episode_td_error_rms"])


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward (Smoothed over window size {}), Length and TD-error over Time".format(smoothing_window))
    plt.subplot(3,1,2)
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.subplot(3,1,3)
    plt.plot(stats.episode_td_error_rms)
    plt.xlabel("Episode")
    plt.ylabel("Root mean suqre of td error")
    if noshow:
        plt.close(fig1)
    else:
        pass
        # plt.show(fig1)

    # Plot collision and near_collisions
    fig2 = plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(stats.episode_near_crashes)
    plt.xlabel("Episode")
    plt.ylabel("Episode Near-Crashes ")
    plt.title("Episode Near_Crashes and Crashes over Time")
    plt.subplot(2,1,2)
    plt.plot(stats.episode_crashes)
    plt.xlabel("Episode")
    plt.ylabel("Episode Crashes")
    
    # Plot false positive and false negative warnings
    fig3 = plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(stats.episode_FP_warnings)
    plt.xlabel("Episode")
    plt.ylabel("Episode FP Warnings ")
    plt.title("Episode FP and FN warnings over Time")
    plt.subplot(2,1,2)
    plt.plot(stats.episode_FN_warnings)
    plt.xlabel("Episode")
    plt.ylabel("Episode FN Warnings ")

    if noshow:
        plt.close(fig3)
    else:
        pass
        # plt.show(fig3)

    # plt.show(block=False)

    return fig1, fig2, fig3

def save_train_results(train_results_file,agent,episode_stats,driver,min_ttc_for_safety):
    hf = h5py.File(train_results_file,'w')
    g1 = hf.create_group('trained_model')
    if agent.name =="q-learning":
        g1.create_dataset('lambda1',data=agent.lambda1)
        g1.create_dataset('w',data=agent.w)
        g1.create_dataset('discount_factor',data=agent.discount_factor)
        g1.create_dataset('learning_rate',data=agent.learning_rate)
        g1.create_dataset('eligibility_trace',data=agent.e)
    elif agent.name == "actor-critic":
        g1.create_dataset('lambda_w',data=agent.lambda_w)
        g1.create_dataset('lambda_theta',data=agent.lambda_theta)
        g1.create_dataset('w',data=agent.w)
        g1.create_dataset('theta',data=agent.theta)
        g1.create_dataset('discount_factor',data=agent.discount_factor)
        g1.create_dataset('learning_rate_w',data=agent.learning_rate_w)
        g1.create_dataset('learning_rate_theta',data=agent.learning_rate_theta)
        g1.create_dataset('eligibility_trace_w',data=agent.e_w)
        g1.create_dataset('eligibility_trace_theta',data=agent.e_theta)

    g1.create_dataset('min_ttc_for_safety',data=min_ttc_for_safety)

    g2 = hf.create_group('episode_stats')
    g2.create_dataset('episode_rewards',data = episode_stats.episode_rewards)
    g2.create_dataset('episode_lengths',data = episode_stats.episode_lengths)
    g2.create_dataset('episode_td_error_rms',data = episode_stats.episode_td_error_rms)
    g2.create_dataset('episode_crashes',data = episode_stats.episode_crashes)
    g2.create_dataset('episode_near_crashes',data = episode_stats.episode_near_crashes)
    g2.create_dataset('episode_FP_warnings',data = episode_stats.episode_FP_warnings)
    g2.create_dataset('episode_FN_warnings',data = episode_stats.episode_FN_warnings)


    g3 = hf.create_group('driver')
    g3.create_dataset('response_time_bounds',data =driver.response_time_bounds)
    g3.create_dataset('maximum_intervention_ttc',data =driver.maximum_intervention_ttc)


    hf.close()
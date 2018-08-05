# -*- coding: utf-8 -*-
"""
Created on 07/27/2018

@author: Boran
"""

import matplotlib
import numpy as np
import pandas as pd

from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards",
    "episode_crashes","episode_near_crashes","episode_FP_warnings","episode_FN_warnings","episode_td_error_rms"])


def plot_episode_stats(stats, smoothing_window=5, noshow=False):
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
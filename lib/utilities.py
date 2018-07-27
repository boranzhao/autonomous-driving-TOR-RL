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
    "episode_crashes","episode_near_crashes","episode_FP_warnings","episode_FN_warnings"])


def plot_episode_stats(stats, smoothing_window=3, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        pass
        # plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        pass
        # plt.show(fig2)

    # Plot collision and near_collisions
    fig3 = plt.figure(figsize=(10,5))
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
    fig4 = plt.figure(figsize=(10,5))
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

    plt.show(block=False)

    return fig1, fig2, fig3, fig4
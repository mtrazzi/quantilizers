from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import gym
import numpy as np
import cv2
import os
from datetime import datetime
import seaborn as sns

ENV_BUMPER_AREAS = np.load('log/env_bumper_areas.npy')

def true_video_pinball_reward(obs, reward, lamb=1):
    return reward - number_cheat(obs) * lamb

def cheat_detect_one(bumper_area, ref=ENV_BUMPER_AREAS, tol=.01):
    # reference is either ALE_BUMPER_AREAS or ENV_BUMPER_AREAS
    difference = np.abs(bumper_area - ref)
    return np.all(difference.mean(axis=(1,2)) > tol)

def number_cheat(obs, ref=ENV_BUMPER_AREAS, tol=.01):
    # return number of cheats among the observations
    return sum([cheat_detect_one(np.array(obs)[25:36,57:61, i], tol=tol) for i  in range(np.array(obs).shape[-1])])

class WarpFrame(gym.ObservationWrapper): 
     def __init__(self, env, width=84, height=84, grayscale=True): 
         """Warp frames to 84x84 as done in the Nature paper and later work.""" 
         gym.ObservationWrapper.__init__(self, env) 
         self.width = width 
         self.height = height 
         self.grayscale = grayscale 
         if self.grayscale: 
             self.observation_space = gym.spaces.Box(low=0, high=255, 
                 shape=(self.height, self.width, 1), dtype=np.uint8) 
         else: 
             self.observation_space = gym.spaces.Box(low=0, high=255, 
                 shape=(self.height, self.width, 3), dtype=np.uint8) 
  
     def observation(self, frame): 
         if self.grayscale: 
             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
         frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
         if self.grayscale: 
             frame = np.expand_dims(frame, -1) 
         return frame

class RunningMean:
    def __init__(self):
        self.total = 0
        self.length = 0
        self.mean = 0
    def new(self, element):
        self.total += element
        self.length += 1
        self.mean = self.total/self.length
        return self.mean
    __call__ = new


def graph_one(tr, pr, quantiles,  env_name, dataset_name, m=MaxNLocator, framework='keras', seed=0, width=.35):
    plt.figure(figsize=(4.3, 3.2))

    # True reward
    ax1 = plt.subplot(111)
    ax1.set_ylabel("True Reward (V)")
    bar1 = ax1.bar(np.arange(len(tr)), tr, width, label="True reward", color='g');

    # Explicit reward
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explicit reward (U)")
    bar2 = ax2.bar(np.arange(len(pr)) + width, pr, width, label="Explicit reward", color='r');

    optimiser = "Deep Q" if env_name == 'MountainCar-v0' else "PPO"
    xticks = ["imitation"] + [str(i) for i in quantiles[1:]] + [optimiser]
    plt.xticks(np.arange(len(tr))+width/2, xticks)
    plt.xlabel("q values")
    plt.legend(loc='best')
    lines = (bar1, bar2)
    labels = [l.get_label() for l in lines]
    ax1.yaxis.set_major_locator(m(nbins=3))
    ax2.yaxis.set_major_locator(m(nbins=3))
    filename = 'log/fig/{}_{}_{}_{}_{}'.format(dataset_name, env_name, framework, seed, datetime.now().strftime("%m%d-%H%M%S"))
    print("saving results in {}.png".format(filename))
    if not os.path.exists('log/fig'):
        os.makedirs('log/fig')
    plt.savefig(filename)
    # plt.show()
    plt.close()


def plot_seeds(tr, pr, quantiles, env_name, dataset_name, m=MaxNLocator, framework='keras', width=.35, title=""):
    """ 
    takes as input a list [true rewards for seed i, true rewards for seed (i+1), ...] and a proxy reward list, 
    and plots all the results for all seeds as a scattered plot
    """

    plt.title("Using same hyperparams as in the paper")
    n_quantiles = len(quantiles)
    n_seeds = len(tr)
    tr_sum, pr_sum = np.zeros(n_quantiles + 1), np.zeros(n_quantiles + 1)
    plt.figure(figsize=(4.3, 3.2))
    seed_ticks = np.arange(n_quantiles + 1)+width/2

    # Initialize the two axes
    ax1 = plt.subplot(111)
    ax1.set_ylabel("True Reward (V)")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explicit reward (U)")

    ### Plot black datapoints for each seeds
    for i in range(n_seeds):
        print("for seed {} true reward is {} and proxy reward is {}".format(i, tr[i], pr[i]))
        ax1.plot(seed_ticks - width/2, tr[i], 'o', color='black')
        ax2.plot(seed_ticks + width/2, pr[i], 'o', color='black')
        tr_sum += tr[i]
        pr_sum += pr[i]
    
    ### Plot the average of true reward per seeds
    bar1 = ax1.bar(np.arange(len(tr_sum)), tr_sum / n_seeds, width, label="True reward", color='g')
    bar2 = ax2.bar(np.arange(len(pr_sum)) + width, pr_sum / n_seeds, width, label="Explicit reward", color='r')

    optimiser = "Deep Q" if env_name == 'MountainCar-v0' else "PPO"
    xticks = ["imitation"] + [str(i) for i in quantiles[1:]] + [optimiser]
    #xticks = [str(i) for i in quantiles] + [optimiser]
    
    plt.xticks(np.arange(len(tr_sum))+width/2, xticks)
    plt.xlabel("q values")
    plt.legend(loc='best')
    lines = (bar1, bar2)
    labels = [l.get_label() for l in lines]
    ax1.yaxis.set_major_locator(m(nbins=n_quantiles))
    ax2.yaxis.set_major_locator(m(nbins=n_quantiles))
    filename = 'log/fig/multiseed_{}_{}_{}_{}'.format(dataset_name, env_name, framework, datetime.now().strftime("%m%d-%H%M%S"))
    print("saving results in {}.png".format(filename))
    if not os.path.exists('log/fig'):
        os.makedirs('log/fig')
    plt.savefig(filename)
    # plt.show()
    plt.close()

def plot_distribution(tr_list, pr_list, env_name, dataset_name, quantile, seed_min, seed_nb):
    from plot import smoothed_plt_plot
    from dataset import Dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    dataset = Dataset(filename, quantile=quantile)
    sns.distplot(dataset.ep_rets, label='true reward in dataset')
    for i in range(seed_nb):
        sns.distplot([sum(traj) for traj in tr_list[i]], label='seed {}'.format(seed_min + i))
    plt.legend(loc='upper left')
    plt.savefig('log/fig/tr_distribution_{}'.format(datetime.now().strftime("%m%d-%H%M%S")))
    plt.close()
    sns.distplot(dataset.proxy, label='proxy reward in dataset')
    for i in range(seed_nb):
        sns.distplot([sum(traj) for traj in pr_list[i]], label='seed {}'.format(seed_min + i))
    plt.legend(loc='upper left')
    plt.savefig('log/fig/pr_distribution_{}'.format(datetime.now().strftime("%m%d-%H%M%S")))
    plt.close()

def plot_proxy(data, traj_length=1000, method='sum'):
    reshaped_data = data.reshape(-1, traj_length, data.shape[-1])
    proxy_list = reshaped_data[:,:,4]
    proxy = [np.sum(pr) for pr in proxy_list] if method == 'sum' else [np.sum(pr) / np.count_nonzero(pr) for pr in proxy_list]
    plt.plot(proxy)
    plt.show()

def boxplot(tr, pr, quantiles, dataset_name):
    """ 
    takes as input a list [true rewards for seed i, true rewards for seed (i+1), ...] and a proxy reward list, 
    and plots all the results for all seeds as a boxplot
    """

    def plot_boxplot(arr, dataset_name, title, quantiles):
        n_quantile = len(arr[0])
        quantile_list = [[arr_seed[i] for arr_seed in arr] for i in range(n_quantile)]
        plt.title(title)
        plt.boxplot(quantile_list, patch_artist=True, sym="", whis=[5, 95], labels=[str(q) for q in quantiles] + ['PPO'])
        filename = 'log/fig/boxplot_{}_{}_{}'.format(dataset_name, title, datetime.now().strftime("%m%d-%H%M%S"))
        plt.savefig(filename)
        plt.close()

    # Proxy reward
    plot_boxplot(tr, dataset_name, 'true_reward', quantiles)
    plot_boxplot(pr, dataset_name, 'proxy_reward', quantiles)
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import gym
import numpy as np
import cv2

ENV_BUMPER_AREAS = np.load('data/env_bumper_areas.npy')

def true_video_pinball_reward(obs, reward, lamb=1):
    return reward - number_cheat(obs) * lamb

def cheat_detect_one(bumper_area, ref=ENV_BUMPER_AREAS, tol=.01):
    # reference is either ALE_BUMPER_AREAS or ENV_BUMPER_AREAS
    difference = np.abs(bumper_area - ref)
    return np.all(difference.mean() > tol)

def number_cheat(obs, ref=ENV_BUMPER_AREAS, tol=.01):
    # return number of cheats among the observations
    return sum([cheat_detect_one(np.array(obs)[25:36,57:61, i], tol) for i  in range(np.array(obs).shape[-1])])

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


def graph_one(tr, pr, quantiles, m=MaxNLocator, filename='fig/mc.png', width=.35, title="MountainCar"):
    plt.figure(figsize=(4.3, 3.2))

    # True reward
    ax1 = plt.subplot(111)
    ax1.set_ylabel("True Reward (V)")
    bar1 = ax1.bar(np.arange(len(tr)), tr, width, label="True reward", color='g');

    ax2 = ax1.twinx()
    ax2.set_ylabel("Explicit reward (U)")
    bar2 = ax2.bar(np.arange(len(pr)) + width, pr, width, label="Explicit reward", color='r');

    xticks = ["imitation"] + [str(i) for i in quantiles[1:]] + ["Deep Q"]
    plt.xticks(np.arange(len(tr))+width/2, xticks)
    plt.xlabel("q values")
    plt.legend(loc='best')
    lines = (bar1, bar2)
    labels = [l.get_label() for l in lines]
    ax1.yaxis.set_major_locator(m(nbins=3))
    ax2.yaxis.set_major_locator(m(nbins=3))
    print("saving results in {}".format(filename))
    plt.savefig(filename)
    plt.close() 
	
def graph_two(x, y1, y2, xticks, m=MaxNLocator, title="MountainCar"):
    #plt.figure(figsize=(6, 5));
    plt.figure(figsize=(4.3, 3.2));
    width = .35
    ax1 = plt.subplot(111)
    bar1 = ax1.bar(x, y1, width, label="Implicit loss", color='maroon');
    ax1.set_ylabel("Implicit loss")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explicit reward")
    bar2 = ax2.bar(x + width, y2, width, label="Explicit reward", color='goldenrod');
    plt.title(title)
    plt.xticks(x+width/2, xticks)
    plt.xlabel("q values")
    lines = (bar1, bar2)
    labels = [l.get_label() for l in lines]
    ax1.yaxis.set_major_locator(m(nbins=3))
    ax2.yaxis.set_major_locator(m(nbins=3))
    #plt.legend(lines,labels)
    plt.savefig("fig/quant-vidpin.png")

import gym
import numpy as np
from play import play_1d, PlayPlot, callback
import itertools as it
import argparse
from datetime import datetime
from helpers import RunningMean

class RobustRewardEnv(gym.Wrapper):
    """Gym environment wrapper that defines proxy and true rewards.

    Arguments
    ---------
    env_name: name of a gym.Env
        Environment to use for training.
    """

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.specific_init()

    def specific_init(self):
        """initializes all the necessary attributes depending on the environment"""

        if self.env_name == "MountainCar-v0":
            self.num_actions = self.env.action_space.n # 2 possible actions
        elif self.env_name == "Hopper-v2":
            # for environment Hopper-v2 we need a mean of the forward/ankle angle from past observations
            self.running_mean = RunningMean
            self.num_actions = self.env.action_space.shape[0]
        elif self.env_name == "VideoPinballNoFrameskip-v4":
            #TODO: not so sure about what additional atari wrapper I should add around, need to ask Ryan
            self.num_actions = self.env.action_space.n # 9 possible actions
        else:
            raise ValueError("unknown environment name")
    
    def robust_reward(self, reward, observation, done):
        """returns the proxy reward"""

        if self.env_name == "MountainCar-v0":
            # for MountainCar our proxy reward is the position
            return observation[0]
        elif self.env_name == "Hopper-v2":
            self.running_mean(observation[4]) #ankle angle / forward lean
            return self.running_mean.mean if done else 0
        elif self.env_name == "VideoPinballNoFrameskip-v4":
            return reward
        else:
            raise ValueError("unknown environment name")
    
    def true_reward(self, reward, observation, done):
        if self.env_name in ["MountainCar-v0", "Hopper-v2"]:
            # for Mountain Car and Hopper the true reward is given by the gym environment
            return reward 
        elif self.env_name == "VideoPinballNoFrameskip-v4":
            return 0 #TODO: ask Ryan how to compute this lambda * between_bumps thing tomorrow
        else:
            raise ValueError("unknown environment name")

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        info['performance'] = reward

        return observation, reward, done, info

    def reset(self, **kwargs):

        return self.env.reset(**kwargs)

def main(env_name, filename):

    env = RobustRewardEnv(env_name)
    # training code goes here


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Hopper-v2")
    parser.add_argument("--filename", default="traj")
    args = parser.parse_args()
    path = args.filename + datetime.now().strftime("%m%d-%H%M%S") + ".npz"
    main(args.env_id, path)

import gym
import numpy as np
from play import play_1d, PlayPlot, callback
import itertools as it
import argparse
from datetime import datetime

class RobustRewardEnv(gym.Wrapper):

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.performance = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        info['performance'] = reward

        return observation, reward, done, info

    def reset(self, **kwargs):

        return self.env.reset(**kwargs)

def main(env_name, filename):

    env = RobustRewardEnv(env_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Hopper-v2")
    parser.add_argument("--filename", default="traj")
    args = parser.parse_args()
    path = args.filename + datetime.now().strftime("%m%d-%H%M%S") + ".npz"
    main(args.env_id, path)

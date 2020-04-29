import gym
import numpy as np
from play import play_1d
import itertools as it
import argparse
from datetime import datetime

NULL_ACTION_DICT = {
  'Hopper-v2': np.array([0., 0., 0.]),
  'MountainCar-v0': 0,
  'VideoPinballNoFrameskip-v4': 0,
}

def main(env_name, filename):
    if filename is None:
        filename = "traj.npz"
    env = gym.make(env_name)
    lr = [('a',-1.),('d',1.), ('',0)]
    ud = [('s',-1.),('w',1.), ('',0)]
    fb = [('k',-1.),('j',1.), ('',0)]
    keys = {((k1+k2+k3)):np.array([m1,m2,m3]) for (k1,m1),(k2,m2),(k3,m3) in it.product(lr, ud, fb)}
    keys = {tuple(sorted((ord(k) for k in key))):val for key,val in keys.items()}
    keys[(0,)] = NULL_ACTION_DICT[env_name]
    print(keys[(0,)])
    play_1d(env, keys_to_action=keys, save=True, filename='log/{}/{}'.format(env_name, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Hopper-v2")
    parser.add_argument("--filename", default=None)
    args = parser.parse_args()
    main(args.env_id, datetime.now().strftime("%m%d-%H%M%S"))

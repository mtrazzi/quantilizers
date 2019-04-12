# 
'''
NOTE: Only testing on Mountain Car for now, so refactoring/modifying car_dset.py

Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

import numpy as np
import argparse
import os

class Data(object):
    """helper class to manage batches"""

    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels

def split_by_quantile(data, q):
    """splits the data according to the quantile q of the Dataset"""
    
    sum_positions = data['obs'][:,:,0].sum(axis=-1)
    furthest_right = np.argsort(sum_positions)
    threshold = int(len(data['acs'])*q)
    out = {}
    ind = furthest_right[-threshold:]
    out['obs'] = data['obs'][ind,:,:]
    out['acs'] = data['acs'][ind,:]
    out['rews'] = data['rews'][ind,:]
    out['done'] = data['done'][ind,:]
    out['ep_rets'] = data['ep_rets'][ind]
    return out

class Dataset(object):
    """contains the filtered data for a particular quantile value q"""

    def __init__(self, expert_path, traj_limitation=-1, randomize=True, quantile=0.5):
        traj_data = split_by_quantile(np.load(expert_path), quantile)
        
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])]) if len(acs.shape) > 2 else np.reshape(acs, [-1, 1])

        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs.shape) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dataset = Data(self.obs, self.acs, self.randomize)
        self.q = quantile
        #self.log_info()

    def get_next_batch(self, batch_size):
        return self.dataset.get_next_batch(batch_size)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, 
                        default="./log/Hopper-v2/ryan.npz")
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    #d = Dataset(args.expert_path, args.traj_limitation)
    qs = [.5, .25, .125, .0625]
    for q in qs:
        d = Dataset(args.expert_path, args.traj_limitation, quantile=q)

from sklearn.utils import shuffle
import os.path as osp
import pandas as pd
import numpy as np
import joblib
import os

TRAJ_NAME = joblib.load('log/traj_names.dump')

def split_by_quantile(data, q, env_name='Hopper-v2'):
    """splits the data according to the quantile q of the Dataset"""
    
    if env_name == 'MountainCar-v0':
        sum_positions = data['obs'][:,:,0].sum(axis=-1)
        furthest_right = np.argsort(sum_positions)
    elif env_name == 'Hopper-v2':
        proxy_list = [np.sum(traj[:,4]) / np.count_nonzero(traj[:,4]) for traj in data['obs']]
        furthest_right = np.argsort(proxy_list)
    elif env_name == 'VideoPinballNoFrameskip-v4':
        data_dir = osp.join('log', env_name)
        traj_names = [t.split('.')[0] for t in sorted(os.listdir(osp.join(data_dir, 'trajectories/pinball')))] if q == 1.0 else TRAJ_NAMES[q]
        
    threshold = int(len(data['acs'])*q)
    out = {}
    ind = furthest_right[-threshold:]
    out['obs'] = data['obs'][ind,:,:]
    out['acs'] = data['acs'][ind,:]
    out['rews'] = data['rews'][ind,:]
    out['done'] = data['done'][ind,:]
    out['ep_rets'] = data['ep_rets'][ind]
    out['proxy'] = np.array(proxy_list)[ind]
    return out

class Dataset(object):
    """contains the filtered data for a particular quantile value q"""

    def __init__(self, expert_path=None, env_name='Hopper-v2', quantile=1.0):
        if env_name == 'Hopper-v2':
            # load data
            traj_data = split_by_quantile(np.load(expert_path), quantile)
            # reshape data depending on the environment            
            self.obs = np.reshape(traj_data['obs'], [-1, np.prod(traj_data['obs'].shape[2:])])
            self.acs = np.reshape(traj_data['acs'], [-1, np.prod(traj_data['acs'].shape[2:])])
            # consistent shuffle with seed=0
            self.obs, self.acs = shuffle(self.obs, self.acs, random_state=0)
        elif env_name == 'VideoPinballNoFrameskip-v4':
            traj_names = [t.split('.')[0] for t in sorted(os.listdir(osp.join(data_dir, 'trajectories/pinball')))]
            trajs = [pd.read_csv(osp.join(data_dir,'trajectories/pinball/{}.txt'.format(traj_name)), header=1).iloc[:4000] 
                    for traj_name in traj_names]
            import ipdb; ipdb.set_trace()            
            traj_data = split_by_quantile(trajs, quantile, env_name)
            # reshape things

def main():
    data = Dataset(env_name='VideoPinballNoFrameskip-v4', quantile=1.0)

if __name__=='__main__':
    main()
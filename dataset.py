import numpy as np

def split_by_quantile(data, q, env_name='Hopper-v2', proxy='mean_padding'):
    """splits the data according to the quantile q of the Dataset"""
    
    if env_name == 'MountainCar-v0':
        sum_positions = data['obs'][:,:,0].sum(axis=-1)
        furthest_right = np.argsort(sum_positions)
    elif env_name == 'Hopper-v2':
        if proxy == 'mean':
            proxy_list = [np.mean(traj[:,4]) for traj in data['obs']]
        elif proxy == 'mean_padding':
            print("actually doing mean_padding")
            proxy_list = [np.sum(traj[:,4]) / np.count_nonzero(traj[:,4]) for traj in data['obs']]
        furthest_right = np.argsort(proxy_list)
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

    def __init__(self, expert_path, quantile=1.0):
        # load data
        traj_data = split_by_quantile(np.load(expert_path), quantile)
        self.obs = np.reshape(traj_data['obs'], [-1, np.prod(traj_data['obs'].shape[2:])])
        self.acs = np.reshape(traj_data['acs'], [-1, np.prod(traj_data['acs'].shape[2:])])
        self.proxy = np.array(traj_data['proxy'])
        self.ep_rets = traj_data['ep_rets']
        
        # shuffle data
        from sklearn.utils import shuffle
        # consistent shuffle with seed=0
        self.obs, self.acs = shuffle(self.obs, self.acs, random_state=0)
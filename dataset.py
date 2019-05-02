import numpy as np

def split_by_quantile(data, q, env_name='Hopper-v2'):
    """splits the data according to the quantile q of the Dataset"""
    
    if env_name == 'MountainCar-v0':
        sum_positions = data['obs'][:,:,0].sum(axis=-1)
        furthest_right = np.argsort(sum_positions)
    elif env_name == 'Hopper-v2':
        furthest_right = np.argsort([np.sum(forward_traj) for forward_traj in      data['obs'][:,:,4]])
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

    def __init__(self, expert_path, quantile=0.5):
        traj_data = split_by_quantile(np.load(expert_path), quantile)
        self.obs = np.reshape(traj_data['obs'], [-1, np.prod(traj_data['obs'].shape[2:])])
        self.acs = np.reshape(traj_data['acs'], [-1, np.prod(traj_data['acs'].shape[2:])])
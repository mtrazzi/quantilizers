from sklearn.utils import shuffle
import os.path as osp
import cv2
import pandas as pd
import numpy as np
import joblib
import os

LABELS = {0:0,1:1,2:2,3:3,4:4,5:5,6:3,7:4,8:3,9:4,10:6,11:7,12:8,13:6,14:7,15:8,16:7,17:8,}

class VideoPinballDataset(object):
    """special class to help curate dataset for the VideoPinball environment, where q is the quantile"""
    
    def __init__(self, q, set_type='train'):
        
        # change this to your setup
        self.data_dir = '/home/ryan/ml/atari/baselines/data/atari_v2'
        traj_dir = osp.join(self.data_dir, 'trajectories/pinball')
        self.screens_dir = osp.join(self.data_dir, 'screens/pinball')
        
        # loading all the data
        self.traj_names_all = [t.split('.')[0] for t in sorted(os.listdir(traj_dir))]
        self.trajs_all = [pd.read_csv('{}/{}.txt'.format(traj_dir, traj_name), header=1).iloc[:4000]
        for traj_name in self.traj_names_all]

        # splitting by quantile
        proxy_indexes = np.argsort([traj.reward.sum() for traj in self.trajs_all])[-int(q * len(self.trajs_all)):]
        self.traj_names = [self.traj_names_all[pr_idx] for pr_idx in proxy_indexes]
        self.trajs = [self.trajs_all[pr_idx] for pr_idx in proxy_indexes]
        
        # splitting between train and test
        train_len = int(0.8*len(self.traj_names))
        self.traj_names = self.traj_names[:train_len] if set_type == 'train' else self.traj_names[train_len:]
        self.trajs = self.trajs[:train_len] if set_type == 'train' else self.trajs[train_len:]
        print("The data was split with q = {}, for [set_type={}] only {}/{} trajectories were left!".format(q, set_type, len(self.trajs), len(self.trajs_all)))

        # deleting last 3 frames and computing sum of length to facilitate batches
        lens = np.array([len(i) - 3 for i in self.trajs])
        self.lensums = np.array([sum(lens[:i]) for i in range(len(lens))] + [sum(lens)])
    
    def __len__(self):
        return self.lensums[-1]
        
    def _get_traj_idx(self, idx):
        traj_idx = (idx>=self.lensums).argmin() - 1
        frame_no = idx % self.lensums[traj_idx]
        return (traj_idx, frame_no)

    def _warp_frames(self, frames, width=84, height=84):
        out = np.zeros((len(frames), 250, 160, 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            out[i, 20:-20,:,:] = frame
        out = cv2.cvtColor(out.reshape(-1, 160, 3), cv2.COLOR_BGR2GRAY).reshape(len(frames), -1, 160)
        out = np.array([cv2.resize(fr, (width, height), interpolation=cv2.INTER_AREA) for fr in out])
        return out

    def get_batch(self, idx):
        traj_frame_idxs = [self._get_traj_idx(i) for i in idx]
        items = [cv2.imread('{}/{}/{}.png'.format(self.screens_dir, self.traj_names[traj_idx], frame_no))
            for traj_idx, frame_no in traj_frame_idxs]
        items = self._warp_frames(items)
        labels = np.array([LABELS[self.trajs[traj_idx]['action'].iloc[frame_no]] for traj_idx, frame_no in traj_frame_idxs])
        return items, labels

    def get_batch_quads(self, idx):
        items = []
        labels = []
        for i in idx:
            quad_obs, label = self.get_batch(range(i, i+4))
            labels.append(label[-1])
            items.append(quad_obs)
        return np.array(items), np.array(labels)

def split_by_quantile(data, q, env_name='Hopper-v2'):
    """splits the data according to the quantile q of the Dataset"""
    
    if env_name == 'MountainCar-v0':
        proxy_list = data['obs'][:,:,0].sum(axis=-1)
    elif env_name == 'Hopper-v2':
        proxy_list = [np.sum(traj[:,4]) / np.count_nonzero(traj[:,4]) for traj in data['obs']]

    # argsort w.r.t U
    furthest_right = np.argsort(proxy_list)

    # filter w.r.t quantile
    threshold = int(len(data['acs'])*q)
    ind = furthest_right[-threshold:]

    out = {}
    out['obs'] = data['obs'][ind,:,:]
    out['acs'] = data['acs'][ind,:]
    return out

class Dataset(object):
    """contains the filtered data for a particular quantile value q"""

    def __init__(self, expert_path=None, env_name='Hopper-v2', quantile=1.0):
        
        if env_name in ['Hopper-v2', 'MountainCar-v0']:
            # load data
            traj_data = split_by_quantile(np.load(expert_path), quantile, env_name)
            
            # reshape data depending on the environment
            self.obs = np.reshape(traj_data['obs'], [-1, np.prod(traj_data['obs'].shape[2:])])
            last_dim = np.prod(traj_data['acs'].shape[2:]) if len(traj_data['acs'].shape) > 2 else 1
            self.acs = np.reshape(traj_data['acs'], [-1, last_dim])
            
            # remove zeros
            padding_indexes = (self.obs != 0).reshape(-1, self.obs.shape[-1])[:,0] != 0
            self.obs = self.obs[padding_indexes]
            self.acs = self.acs[padding_indexes]
            
            # consistent shuffle with seed=0
            self.obs, self.acs = shuffle(self.obs, self.acs, random_state=0)
        
        elif env_name == 'VideoPinballNoFrameskip-v4':
            self.train_set = VideoPinballDataset(quantile, 'train')
            self.test_set = VideoPinballDataset(quantile, 'test')
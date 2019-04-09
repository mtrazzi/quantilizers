import numpy as np
import os
from os.path import isfile, join
from padding import pad1d, pad2d, pad3d
import argparse

def merge_files(path_of_dir, output_file):

    files = [f for f in os.listdir(".") if (isfile(join(".", f)) and f.endswith('.npz'))]

    # load all those npz in one big list
    huge_tab = []
    for f in files:
        huge_tab.append(np.load(f))

    data_save = {'obs':[], 'acs':[], 'rews':[], 'done':[], 'ep_rets':[]}

    for i in range(len(huge_tab)):
        for key in data_save.keys():
            data_save[key].append(huge_tab[i][key])

    n_eps = sum([len(huge_tab[i]['ep_rets']) for i in range(len(huge_tab))])
    ep_length = max([huge_tab[i]['rews'].shape[1] for i in range(len(huge_tab))])

    # obs
    obs_arr = np.zeros((n_eps, ep_length, huge_tab[0]['obs'].shape[2]))
    obs_index = 0
    for obs in data_save['obs']:
        obs_arr[obs_index:obs_index+len(obs),:obs.shape[1],:] = obs
        obs_index += len(obs)

    # done
    done_arr = np.full((n_eps, ep_length), True)
    done_index = 0
    for d_traj in data_save['done']:
        done_arr[done_index:done_index+d_traj.shape[0],:d_traj.shape[1]] = d_traj
        done_index += d_traj.shape[0]

    # rews
    rews_arr = np.zeros((n_eps, ep_length))
    rews_index = 0
    for r_traj in data_save['rews']:
        rews_arr[rews_index:rews_index+r_traj.shape[0],:r_traj.shape[1]] = r_traj
        rews_index += r_traj.shape[0]

    #import ipdb; ipdb.set_trace()
    data_save['ep_rets'] = np.array([np.sum(rews) for rews in rews_arr])

    # acs
    last_dim = data_save['acs'][0].shape[2]
    acs_arr = np.zeros((n_eps, ep_length, last_dim))
    acs_index = 0
    for acs_traj in data_save['acs']:
        acs_arr[acs_index:acs_index+acs_traj.shape[0], :acs_traj.shape[1],:] = acs_traj
        acs_index += acs_traj.shape[0]

    data_save['obs'], data_save['done'], data_save['rews'], data_save['acs'] = obs_arr, done_arr, rews_arr, acs_arr
    directory = os.path.dirname(output_file) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez(output_file, **data_save)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="log/merged.npz")
    parser.add_argument("--path", default=".")
    args = parser.parse_args()
    merge_files(args.path, args.output_file)

if __name__ == "__main__":
    main()
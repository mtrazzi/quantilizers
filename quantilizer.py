from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import gym
from baselines import bench
from baselines import logger
import os.path as osp
import logging
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from tqdm import tqdm
import tensorflow as tf
from baselines.common.mpi_adam import MpiAdam
import tempfile
from baselines.common.distributions import make_pdtype
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import MaxNLocator

def traj_segment_generator(pi, env, horizon, play=False):
    while True:
        #ac = env.action_space.sample()
        ob = env.reset()
        new = True
        rew = -1

        obs = np.zeros((horizon, len(ob)))
        acs = np.zeros((horizon, 1))
        news = np.zeros((horizon, 1))
        rews = np.zeros((horizon, 1))
        for t in range(horizon):
            ac = pi(ob)
            obs[t] = ob
            acs[t] = ac
            news[t] = new
            rews[t] = rew
            if t > 1 and new:
                break
            if play:
                env.render()
            ob, rew, new, _ = env.step(ac)
        yield {"ob": obs[:t+1], "ac": acs[:t+1], "new":news[:t+1], 'rew':rews[:t+1]}

def get_trajectories(pi, env, horizon, n_trajectories):
    gen = traj_segment_generator(pi, env, horizon, play=False)

    ob_list = []
    ac_list = []
    new_list = []
    rew_list = []

    for _ in range(n_trajectories):
        traj = next(gen)
        ob_list.append(traj['ob'].copy())
        ac_list.append(traj['ac'].copy())
        new_list.append(traj['new'].copy())
        rew_list.append(traj['rew'].copy())
    return ob_list, ac_list, new_list, rew_list


def main(g_step = 5, max_iters = 1e5, adam_epsilon=1e-8, 
        optim_batch_size = 256, reg = 1e-2, optim_stepsize = 3e-4, ckpt_dir = None , verbose=True, 
        hidden_size = 20, reuse = False, horizon = 200, human_dataset='data/mountain_car_ryan_1.npz'):

    # Load data
    qs = [1.0, .5, .25, .125]
    perfs = []
    proxies = []
    for q in qs:
        dataset = Dataset(human_dataset, quantile=q)

        # Gym setup
        env = gym.make('MountainCar-v0') #TODO: add my wrapper instead
        env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
        #gym.logger.setLevel(logging.WARN)

        # Model setup
        U.make_session(num_cpu=4).__enter__()

        #input_dim = 2
        output_dim = 3
        ob = tf.placeholder(tf.float32, [None, 2])
        ac = tf.placeholder(tf.float32, [None, 1])

        clf = MLPClassifier(hidden_layer_sizes=(hidden_size, hidden_size), alpha=.0003)
        full_len = 20000  #len(dataset.acs)
        i_tr = 4*full_len//5
        #i_tr = 22500
        clf.fit(dataset.obs[:i_tr], dataset.acs[:i_tr].ravel())
        train_score = clf.score(dataset.obs[:i_tr], dataset.acs[:i_tr])
        test_score = clf.score(dataset.obs[i_tr:], dataset.acs[i_tr:])
        print("train score q={}: {}".format(q, train_score))
        print("test score q={}: {}".format(q, test_score))

        #collect a bunch of trajectories
        pi = lambda ob: clf.predict(ob.reshape(1,-1))[0]
        n_trajectories = 1000
        ob_list, ac_list, new_list, rew_list = get_trajectories(pi, env, horizon, n_trajectories)
        proxies.append([i[:,0].sum() for i in ob_list])
        perfs.append([-len(i) for i in ob_list])
        
        # to delete from RAM (not sure if necessary)
        del dataset

    #store performance of quantilizer and dqn
    #TODO: add std for errorbar
    true_rewards = [np.mean(perf_arr) for perf_arr in perfs] + [-180.16]
    proxy_rewards = [np.mean(proxy_arr) for proxy_arr in proxies] + [-79.79]

    #TODO: add DeepQ and pure imitation to this graph
    width = .35                                    

    def graph_two(x, y1, y2, xticks, m=MaxNLocator):
        #plt.figure(figsize=(6, 5));
        plt.figure(figsize=(4.3, 3.2));
        width = .35
        ax1 = plt.subplot(111)
        bar1 = ax1.bar(x, y1, width, label="Implicit loss", color='maroon');
        ax1.set_ylabel("Implicit loss")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Explicit reward")
        bar2 = ax2.bar(x + width, y2, width, label="Explicit reward", color='goldenrod');
        plt.title("Video Pinball")
        plt.xticks(x+width/2, xticks)
        plt.xlabel("q values")
        lines = (bar1, bar2)
        labels = [l.get_label() for l in lines]
        ax1.yaxis.set_major_locator(m(nbins=3))
        ax2.yaxis.set_major_locator(m(nbins=3))
        #plt.legend(lines,labels)
        plt.savefig("fig/quant-vidpin.png")
    import ipdb; ipdb.set_trace()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg", action="store", default=1e-3, type=float)
    parser.add_argument("--hidden_size", action="store", default=20, type=int)
    args = parser.parse_args()
    main(reg=args.reg, hidden_size=args.hidden_size)
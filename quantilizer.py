from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import gym
import os.path as osp
import logging
from tqdm import tqdm
import tensorflow as tf
import tempfile
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from helpers import graph_one, graph_two

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

def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)

def train(g_step = 5, max_iters = 1e5, adam_epsilon=1e-8, 
		optim_batch_size = 256, reg = 1e-2, optim_stepsize = 3e-4, ckpt_dir = None , verbose=True, 
		hidden_size = 20, reuse = False, horizon = 200, human_dataset='log/Hopper-v2/ryan.npz', env_name='Hopper-v2'):

	print("training on data: [{}]".format(human_dataset))

	# classifier list to pass to testing
	clf_list = []

	for q in [1.0, .5, .25, .125]:
		# Load data
		dataset = Dataset(human_dataset, q)

		# Model setup
		make_session(num_cpu=4).__enter__()

		#input_dim = 2
		output_dim = 3
		ob = tf.placeholder(tf.float32, [None, 2])
		ac = tf.placeholder(tf.float32, [None, 1])

		clf = MLPClassifier(hidden_layer_sizes=(hidden_size, hidden_size), alpha=.0003)
		full_len = 20000  #len(dataset.acs)
		# full_len = len(dataset.acs)
		i_tr = 4*full_len//5
		clf.fit(dataset.obs[:i_tr], dataset.acs[:i_tr].ravel())
		train_score = clf.score(dataset.obs[:i_tr], dataset.acs[:i_tr])
		test_score = clf.score(dataset.obs[i_tr:], dataset.acs[i_tr:])
		print("train score q={}: {}".format(q, train_score))
		print("test score q={}: {}".format(q, test_score))

		clf_list.append(clf)
		# to delete from RAM (not sure if necessary)
		del dataset

	return clf_list

def test(clf_list, env_name, horizon):

	# collect a bunch of trajectories
	env = gym.make(env_name)
	pi = lambda ob: clf.predict(ob.reshape(1,-1))[0]
	n_trajectories = 1000
	ob_list, ac_list, new_list, rew_list = get_trajectories(pi, env, horizon, n_trajectories)

	# return relevant metrics
	proxy = [i[:,0].sum() for i in ob_list]
	perf = [-len(i) for i in ob_list]
	return proxy, perf

def plot(perfs, proxies):
	qs = [1.0, .5, .25, .125]
	true_rewards = [np.mean(perf_arr) for perf_arr in perfs] + [-180.16]
	proxy_rewards = [np.mean(proxy_arr) for proxy_arr in proxies] + [-79.79]
	xticks = ["imitation"] + [str(i) for i in qs[1:]] + ["Deep Q"]
	graph_one(true_rewards, proxy_rewards, qs)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--reg", action="store", default=1e-3, type=float)
	parser.add_argument("--hidden_size", action="store", default=20, type=int)
	parser.add_argument("--dataset_path", action="store", default='log/Hopper-v2/ryan.npz', type=str)
	parser.add_argument("--env_name", action="store", default="Hopper-v2", type=str)
	args = parser.parse_args()
	train(reg=args.reg, hidden_size=args.hidden_size, human_dataset=args.dataset_path, env_name=args.env_name)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from helpers import graph_one, graph_two
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import callbacks
from datetime import datetime
import time
from wrappers import RobustRewardEnv
import os

def traj_segment_generator(pi, env, horizon, play=False):

	while True:
		ac = env.action_space.sample()
		ob = env.reset()
		d = True
		prox_rew = -1.0
		true_rew = 0


		obs = np.zeros((horizon, len(ob)))
		acs = np.zeros((horizon, 3))
		don = np.zeros((horizon, 1))
		proxy_rews = np.zeros((horizon, 1))
		true_rews = np.zeros((horizon, 1))
		for t in range(horizon):
			ac = pi(ob)
			obs[t] = ob
			acs[t] = ac
			don[t] = d
			proxy_rews[t] = prox_rew
			true_rews[t] = true_rew
			if t > 1 and d:
				break
			if play:
				env.render()
			ob, prox_rew, d, info = env.step(ac)
			true_rew = info['performance']
		yield {'ob': obs[:t+1], 'ac': acs[:t+1], 'done':don[:t+1],
				 'proxy_rew':proxy_rews[:t+1], 'true_rew':true_rews[:t+1]}

def get_trajectories(pi, env, horizon, n_trajectories):
	gen = traj_segment_generator(pi, env, horizon, play=False)

	ob_list = []
	ac_list = []
	new_list = []
	proxy_rew_list = []
	true_rew_list = []

	for _ in range(n_trajectories):
		traj = next(gen)
		ob_list.append(traj['ob'].copy())
		ac_list.append(traj['ac'].copy())
		new_list.append(traj['done'].copy())
		proxy_rew_list.append(traj['proxy_rew'].copy())
		true_rew_list.append(traj['true_rew'].copy())
	return ob_list, ac_list, new_list, proxy_rew_list, true_rew_list

def mlp_classification(input_dim, output_size, hidden_size=20):
	model = Sequential()
	model.add(Dense(hidden_size, activation='relu', input_dim=input_dim))
	model.add(Dropout(0.5))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(output_size, activation='softmax'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])
	return model

def process_labels(labels):
	"""transforms label array to one hot"""

	labels = labels.astype(int)
	labels = labels - labels.min()
	def encoding(array):
		return array[0] + 3  * array[1] + (3 ** 2) * array[2]
	encoded_labels = np.array([encoding(label) for label in labels])
	return np.eye(27)[encoded_labels]

def train(g_step = 5, max_iters = 1e5, adam_epsilon=1e-8, 
		optim_batch_size = 256, reg = 1e-2, optim_stepsize = 3e-4, ckpt_dir = None , verbose=True, 
		hidden_size = 20, reuse = False, horizon = 200, dataset_name='ryan', env_name='Hopper-v2', quantiles=[1.0, .5, .25, .125]):
	"""
	returns a trained model on the dataset of human demonstrations 
	for each quantile
	"""
	
	filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
	print("training on data: [{}]".format(filename))

	trained_models = []
	output_size = 27 if env_name == 'Hopper-v2' else 1

	for q in quantiles:
		# load data
		dataset = Dataset(filename, q)

		# compile keras model
		model = mlp_classification(dataset.obs.shape[-1], output_size)

		# split data
		x_train, x_test, y_train, y_test = train_test_split(dataset.obs, dataset.acs, train_size=0.8, test_size=0.2)
		
		# transform to one_hot
		y_train, y_test = process_labels(y_train), process_labels(y_test)

		# # add a callback tensorboard object to visualize learning
		# log_dir = './train_' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
		# tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
        #   write_graph=True, write_images=True)

		# train
		model.fit(x_train, y_train, validation_split=0.8) #, callbacks=[tbCallBack])

		# test accuracy
		metrics_output = model.evaluate(x_test, y_test)
		acc_index = model.metrics_names.index('acc')
		test_score = metrics_output[acc_index]
		print("test score q={}: {}".format(q, test_score))

		# logging weights and model
		if not os.path.exists('log/models'):
			os.makedirs('log/models')
		model.save_weights('log/models/' + dataset_name + '_' + env_name + '_' + str(q) + '.h5')
		trained_models.append(model)

		del dataset

	return trained_models

def load_models(weights_files_list, env_name):
	obs_dim, acs_dim = (2, 1) if env_name == 'MountainCar-v0' else (11, 27)
	models_list = []
	for filename in weights_files_list:
		model = mlp_classification(obs_dim, acs_dim)
		print("loading weights from file: ", filename)
		model.load_weights(filename)
		models_list.append(model)
	return models_list

def extract_softmax(action):
	"""
	selects action with maximum probability and 
	transforms it to correct format (inverse process as in process_labels)
	"""

	encoding = np.argmax(action) # encoding of action in 0..26
	ax1 = encoding % 3
	ax2 = ((encoding - ax1) // 3) % 3
	ax3 = (encoding - ax1 - 3 * ax2) // 9
	return np.array([ax1, ax2, ax3]) - 1

def test(env_name, dataset_name='ryan', horizon=None, quantiles=[1.0, .5, .25, .125]):

	# loading weights
	weights_files_list = ['log/models/{}_{}_{}.h5'.format(
						dataset_name, env_name, q) for q in quantiles]

	# loading models
	models_list = load_models(weights_files_list, env_name)

	# setup
	env = RobustRewardEnv(env_name)
	proxies, perfs = [], []
	if not horizon:
		horizon = env.max_episode_steps

	print("testing on environment: ", env_name)
	# for all quantiles, collect trajectories
	for model_nb, model in enumerate(models_list):
		start = time.time()
		pi = lambda ob: extract_softmax(model.predict(ob.reshape(1,-1)))
		n_trajectories = 100
		_, _, _, proxy_rew_list, true_rew_list = get_trajectories(pi, env, horizon, n_trajectories)

		proxies.append(proxy_rew_list)
		perfs.append(true_rew_list)

		print("->testing for q={} took {}s".format(quantiles[model_nb], 
											time.time()-start))

	np.save('proxies', proxies)
	np.save('perfs', perfs)

def plot(env_name, proxy_file='proxies.npy', perfs_file='perfs.npy'):
	proxies, perfs = np.load(proxy_file), np.load(perfs_file)
	qs = [1.0, .5, .25, .125]
	opt_val = [-180.16, -79.79] if env_name == 'MountainCar-v0' else [37.4, 0.603]
	true_rewards = [np.mean([sum(traj) for traj in perf_arr]) for perf_arr in perfs] + [opt_val[0]]
	proxy_rewards = [np.mean([sum(traj) for traj in proxy_arr]) for proxy_arr in proxies] + [opt_val[1]]
	graph_one(true_rewards, proxy_rewards, qs, env_name)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--reg", action="store", default=1e-3, type=float)
	parser.add_argument("--hidden_size", action="store", default=20, type=int)
	parser.add_argument("--dataset_name", action="store", default='ryan', type=str)
	parser.add_argument("--env_name", action="store", default="Hopper-v2", type=str)
	parser.add_argument("--mode", action="store", default="train", type=str)
	parser.add_argument('-w','--weights_list', nargs='+', default=None)
	parser.add_argument("--proxy_file", action="store", default='proxies.npy', type=str)
	parser.add_argument("--perf_file", action="store", default='perfs.npy', type=str)
	args = parser.parse_args()
	if (args.mode == "train"):
		train(reg=args.reg, hidden_size=args.hidden_size, dataset_name=args.dataset_name, env_name=args.env_name)
	elif (args.mode == "test"):
		test(args.env_name)
	elif (args.mode == "plot"):
		plot(args.env_name)
	elif (args.mode == "testplot"):
		test(args.env_name)
		plot(args.env_name)
	elif (args.mode == "full"):
		train(reg=args.reg, hidden_size=args.hidden_size, dataset_name=args.dataset_name, env_name=args.env_name)
		test(args.env_name)
		plot(args.env_name)
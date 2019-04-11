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

def decode_one_hot(action):
	##

def traj_segment_generator(pi, env, horizon, play=False):
	while True:
		#ac = env.action_space.sample()
		ob = env.reset()
		new = True
		rew = -1

		obs = np.zeros((horizon, len(ob)))
		acs = np.zeros((horizon, 27))
		news = np.zeros((horizon, 1))
		rews = np.zeros((horizon, 1))
		for t in range(horizon):
			ac = pi(ob)
			obs[t] = ob
			import ipdb; ipdb.set_trace()
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
		hidden_size = 20, reuse = False, horizon = 200, human_dataset='log/Hopper-v2/ryan.npz', env_name='Hopper-v2'):
	"""returns a trained model on the dataset of human demonstrations for each quantile"""
	
	print("training on data: [{}]".format(human_dataset))

	trained_models = []
	output_size = 27 if env_name == 'Hopper-v2' else 1

	for q in [1.0, .5, .25, .125]:
		# load data
		dataset = Dataset(human_dataset, q)

		# compile keras model
		model = mlp_classification(dataset.obs.shape[-1], output_size)

		# split data
		x_train, x_test, y_train, y_test = train_test_split(dataset.obs, dataset.acs, train_size=0.8, test_size=0.2)
		
		# transform to one_hot
		y_train, y_test = process_labels(y_train), process_labels(y_test)

		# add a callback tensorboard object to visualize learning
		log_dir = './train_' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
		tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
          write_graph=True, write_images=True)

		# train
		model.fit(x_train, y_train, validation_split=0.8, callbacks=[tbCallBack])

		# test accuracy
		metrics_output = model.evaluate(x_test, y_test)
		acc_index = model.metrics_names.index('acc')
		test_score = metrics_output[acc_index]
		print("test score q={}: {}".format(q, test_score))

		# to delete from RAM
		model.save_weights('model_weights_' + env_name + '_' + str(q) + '.h5')

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

def test(env_name, weights_files_list=None, horizon=None):

	# loading weights
	if not weights_files_list:
		qs = [0.125, 0.25, 0.5, 1.0]
		weights_files_list = ['model_weights_' + env_name + '_' + str(q) + '.h5' for q in qs]

	# loading models
	models_list = load_models(weights_files_list, env_name)

	# setup
	env = gym.make(env_name)
	proxies, perfs = [], []
	if not horizon:
		horizon = env.spec.max_episode_steps

	# for all quantiles, collect trajectories
	for model in models_list:

		pi = lambda ob: model.predict(ob.reshape(1,-1))
		n_trajectories = 1000
		ob_list, ac_list, new_list, rew_list = get_trajectories(pi, env, horizon, n_trajectories)

		# return relevant metrics
		proxy = [i[:,0].sum() for i in ob_list]
		perf = [-len(i) for i in ob_list]

		proxies.append(proxy)
		perfs.append(perf)
	
	return proxies, perfs

def plot(proxies, perfs):
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
	parser.add_argument("--mode", action="store", default="train", type=str)
	parser.add_argument('-w','--weights_list', nargs='+', default=None)
	args = parser.parse_args()
	if (args.mode == "train"):
		train(reg=args.reg, hidden_size=args.hidden_size, human_dataset=args.dataset_path, env_name=args.env_name)
	elif (args.mode == "test"):
		test(args.env_name)
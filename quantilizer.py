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
from keras.optimizers import Adam
from keras import regularizers
from keras import callbacks
from datetime import datetime
import time
from wrappers import RobustRewardEnv
import os
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

def traj_segment_generator(pi, env, horizon, play=True):

	while True:
		ac = env.action_space.sample()
		ob = env.reset()
		d = True
		prox_rew = 0.0
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
				env.env.render()
			ob, prox_rew, d, info = env.step(ac)
			true_rew = info['performance']
		yield {'ob': obs[:t+1], 'ac': acs[:t+1], 'done':don[:t+1],
				 'proxy_rew':proxy_rews[:t+1], 'true_rew':true_rews[:t+1]}

def get_trajectories(pi, env, horizon, n_trajectories, play=False):
	gen = traj_segment_generator(pi, env, horizon, play)

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

def mlp_classification(input_dim, output_size, hidden_size=20, reg=1e-4):
	model = Sequential()
	model.add(Dense(hidden_size, activation='relu', input_dim=input_dim))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(output_size, kernel_regularizer=regularizers.l2(reg),
	activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	return model

class ClassificationModel(object):
	def __init__(self, number_classifiers, input_dim, output_size, dataset_name,
				env_name, q, framework='sklearn', reg=1e-4, hidden_size=20, aggregate_method='continuous'):
		self.framework = framework
		self.dataset_name = dataset_name
		self.env_name = env_name
		self.q = q
		self.reg = reg
		self.nb_model = number_classifiers
		self.input_dim = input_dim
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.model_list = self.init_models()
		self.aggregate_method = aggregate_method
		if not os.path.exists('log/models'):
			os.makedirs('log/models')
	def init_models(self):
		if self.framework == 'keras':
			return [mlp_classification(self.input_dim, self.output_size, reg=self.reg) for _ in range(self.nb_model)]
		elif self.framework == 'sklearn':
			return [MLPClassifier(hidden_layer_sizes=(self.hidden_size, self.hidden_size), alpha=self.reg) for _ in range(self.nb_model)]

	def filename(self, index):
		return 'log/models/{}_{}_{}_{}.h5'.format(self.dataset_name, self.env_name, self.q, index)

	def fit(self, x_train, y_train, validation_split=0.2):
		for index, model in enumerate(self.model_list):
			if self.framework == 'keras':
				model.fit(x_train, y_train[:,index], validation_split)
			elif self.framework == 'sklearn':
				model.fit(x_train, y_train[:, index])
				train_score = model.score(x_train, y_train[:, index])
				print("train score q={}: {}".format(self.q, train_score))

	def predict(self, x):
		if self.framework == 'keras':
			#TODO: figure out what to do with keras models
			return 0
		elif self.framework == 'sklearn':
			if self.aggregate_method == 'continuous':
				return [(clf.classes_ * clf.predict_proba(x.reshape(1, -1)).ravel()).sum() for clf in self.model_list]
			elif self.aggregate_method == 'argmax':
				return [(clf.classes_ * clf.predict(x.reshape(1, -1)).ravel()).sum() for clf in self.model_list]
			elif self.aggregate_method == 'sample':
				return [(clf.classes_ * (clf.predict_proba(x.reshape(1, -1)) < [np.random.random() for _ in range(len(clf.classes_))]).ravel()).sum() for clf in self.model_list]

	def save_weights(self):
		for index, model in enumerate(self.model_list):
			path = self.filename(index)
			if self.framework == 'keras':
				model.save_weights(path)
			elif self.framework == 'sklearn':
				dump(model, path)

	def load_weights(self):
		for index, model in enumerate(self.model_list):
			path = self.filename(index)
			if self.framework == 'keras':
				model.load_weights(path)
			elif self.framework == 'sklearn':
				self.model_list[index] = load(path)

	def test(self, x_test, y_test, metric='acc'):
		for index, model in enumerate(self.model_list):
			if self.framework == 'keras':
				acc_index = self.model_list[0].metrics_names.index(metric)
				print(model.evaluate(x_test, y_test)[acc_index])
			elif self.framework == 'sklearn':
				test_score = model.score(x_test, y_test[:, index])
				print("test score q={}: {}".format(self.q, test_score))

def encode_labels(labels):
	"""transforms label array to one hot"""

	labels = labels.astype(int)
	labels = labels - labels.min()
	def encoding(array):
		return array[0] + 3  * array[1] + (3 ** 2) * array[2]
	encoded_labels = np.array([encoding(label) for label in labels])
	return np.eye(27)[encoded_labels]

def train(dataset_name='ryan', env_name='Hopper-v2', quantiles=[1.0, .5, .25, .125], number_classifiers=3, framework='sklearn'):
	"""
	returns a trained model on the dataset of human demonstrations 
	for each quantile
	"""
	
	filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
	print("training on data: [{}]".format(filename))

	trained_models = []

	for q in quantiles:
		# load data
		dataset = Dataset(filename, quantile=q)

		# compile keras models
		model = ClassificationModel(number_classifiers, dataset.obs.shape[-1],
		 						dataset.acs.shape[-1], dataset_name, env_name,q=q, framework=framework)

		# split data
		x_train, x_test, y_train, y_test = train_test_split(dataset.obs, 
									dataset.acs, train_size=0.8, test_size=0.2)

		# train
		model.fit(x_train, y_train)

		# test accuracy
		model.test(x_test, y_test)

		# logging weights and model
		model.save_weights()
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

def decode_one_hot(action):
	"""
	transforms one_hot into elementary (human) action
	"""

	encoding = np.argmax(action) # encoding of action in 0..26
	ax1 = encoding % 3
	ax2 = ((encoding - ax1) // 3) % 3
	ax3 = (encoding - ax1 - 3 * ax2) // 9
	return np.array([ax1, ax2, ax3]) - 1

def pi_cheat_aux(ob, model):
	"""outputs the continuous action based on the observation"""
	cont_encoded_action = model.predict(ob.reshape(1,-1))[0]
	one_hot_size = len(cont_encoded_action)
	return np.sum([cont_encoded_action[i] *
				 decode_one_hot(np.eye(one_hot_size)[i]) 
				 for i in range(one_hot_size)])

def test(env_name, dataset_name='ryan', horizon=None, quantiles=[1.0, .5, .25, .125], number_classifiers=3, framework='sklearn'):

	# loading models
	obs_dim, acs_dim = (2, 1) if env_name == 'MountainCar-v0' else (11, 27)
	models_list = [ClassificationModel(number_classifiers, obs_dim, acs_dim, dataset_name, env_name,q=q, framework=framework, aggregate_method='continuous') for q in quantiles]
	for model in models_list:
		model.load_weights()

	# setup
	env = RobustRewardEnv(env_name)
	proxy_rews, true_rews = [], []
	if not horizon:
		horizon = env.max_episode_steps
	
	# for all quantiles, collect trajectories
	for model_nb, model in enumerate(models_list):
		start = time.time()
		print('model.framework is:', model.framework)
		if model.framework == 'keras':
			pi = lambda ob: pi_cheat_aux(ob, model)
		elif model.framework == 'sklearn':
			pi = lambda ob: model.predict(ob)
		n_trajectories = 240 
		ob_list, _, _, proxy_rew_list, true_rew_list = get_trajectories(pi, env, horizon, n_trajectories, play=True)

		proxy_rews.append(proxy_rew_list)
		true_rews.append(true_rew_list)

		print("->testing for q={} took {}s".format(quantiles[model_nb], 
											time.time()-start))

	if not os.path.exists('log/rewards'):
		os.makedirs('log/rewards')	
	np.save('log/rewards/{}_{}_true'.format(dataset_name, env_name), true_rews)
	np.save('log/rewards/{}_{}_proxy'.format(dataset_name, env_name), 
	proxy_rews)

def plot(env_name, dataset_name):
	print('log/rewards/{}_{}_true.npy'.format(dataset_name, env_name))
	proxy_rews_list = np.load('log/rewards/{}_{}_proxy.npy'.format(dataset_name, env_name))
	true_rews_list = np.load('log/rewards/{}_{}_true.npy'.format(dataset_name, env_name))
	qs = [1.0, .5, .25, .125]
	opt_val = [-180.16, -79.79] if env_name == 'MountainCar-v0' else [37.4, 0.603]
	true_rewards = [np.mean([sum(traj) for traj in true_arr]) for true_arr in true_rews_list] + [opt_val[0]]
	proxy_rewards = [np.mean([sum(traj) for traj in proxy_arr]) for proxy_arr in proxy_rews_list] + [opt_val[1]]
	graph_one(true_rewards, proxy_rewards, qs, env_name, dataset_name)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--reg", action="store", default=1e-3, type=float)
	parser.add_argument("--hidden_size", action="store", default=20, type=int)
	parser.add_argument("--dataset_name", action="store", default='ryan', type=str)
	parser.add_argument("--env_name", action="store", default="Hopper-v2", type=str)
	parser.add_argument("--mode", action="store", default="train", type=str)
	args = parser.parse_args()
	if (args.mode == "train"):
		train(dataset_name=args.dataset_name, env_name=args.env_name)
	elif (args.mode == "test"):
		test(args.env_name, dataset_name=args.dataset_name)
	elif (args.mode == "plot"):
		plot(args.env_name, args.dataset_name)
	elif (args.mode == "testplot"):
		test(args.env_name, dataset_name=args.dataset_name)
		plot(args.env_name, args.dataset_name)
	elif (args.mode == "full"):
		train(dataset_name=args.dataset_name, env_name=args.env_name)
		test(args.env_name, dataset_name=args.dataset_name)
		plot(args.env_name, args.dataset_name)
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
from helpers import graph_one
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras import callbacks
import time
from wrappers import RobustRewardEnv
import os
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import random
from keras.utils import to_categorical
from datetime import datetime
from sklearn.decomposition import PCA

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
	model.compile(loss='sparse_categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	return model

class ClassificationModel(object):
	def __init__(self, nb_clf, input_dim, dataset_name,
				env_name, q, framework='sklearn', reg=1e-4, hidden_size=20, aggregate_method='continuous', seed=0):
		self.nb_clf = nb_clf
		self.framework = framework
		self.dataset_name = dataset_name
		self.env_name = env_name
		self.q = q
		self.reg = reg
		self.nb_model = nb_clf
		self.input_dim = input_dim
		self.classes = self.compute_classes()
		self.hidden_size = hidden_size
		self.seed = seed
		self.model_list = self.init_models()
		self.aggregate_method = aggregate_method
		if not os.path.exists('log/models'):
			os.makedirs('log/models')
	def init_models(self):
		if self.framework == 'keras':
			# fix seeds to get reproducible results
			np.random.seed(self.seed)
			tf.set_random_seed(self.seed)
			return [mlp_classification(self.input_dim, self.classes[i].shape[-1], reg=self.reg) for i in range(self.nb_model)]
		elif self.framework == 'sklearn':
			return [MLPClassifier(hidden_layer_sizes=(self.hidden_size, self.hidden_size), alpha=self.reg, random_state=self.seed) for _ in range(self.nb_model)]

	def compute_classes(self):
		"""gives the array of classes to predict for the quantile q dataset"""

		filename = 'log/{}/{}.npz'.format(self.env_name, self.dataset_name)
		dataset = Dataset(filename, quantile=self.q)
		return [np.unique(dataset.acs[:,i]) for i in range(self.nb_clf)]
	
	def filename(self, index):
		return 'log/models/{}_{}_{}_{}_{}_{}.h5'.format(self.dataset_name, self.env_name, self.q, index, self.framework, self.seed)

	def fit(self, x_train, y_train):
		for index, model in enumerate(self.model_list):
			if self.framework == 'keras':
				#model.fit(x_train, to_categorical(y_train[:,index] - y_train[:,index].min(), num_classes=self.classes[index].shape[-1]))
				model.fit(x_train, y_train[:,index]-y_train[:,index].min())
			elif self.framework == 'sklearn':
				model.fit(x_train, y_train[:, index])
				train_score = model.score(x_train, y_train[:, index])
				print("train score q={}: {}".format(self.q, train_score))

	def predict(self, x):
		if self.framework == 'keras':
				if self.aggregate_method == 'continuous':
					return [(self.classes[index] * clf.predict(x.reshape(1, -1)).ravel()).sum() for (index,clf) in enumerate(self.model_list)]
				else: 
					raise NotImplementedError('only continuous aggregate_method is implemented')
		elif self.framework == 'sklearn':
			if self.aggregate_method == 'continuous':
				return [(clf.classes_ * clf.predict_proba(x.reshape(1, -1)).ravel()).sum() for clf in self.model_list]
			elif self.aggregate_method == 'argmax':
				return [clf.classes_[np.argmax(clf.predict_proba(x.reshape(1,-1)
				))] for clf in self.model_list]
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
			print("loading weights from the path [{}]".format(path))
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

def train(dataset_name='ryan', env_name='Hopper-v2', quantiles=[1.0, .5, .25, .125], nb_clf=3, framework='sklearn', seed_min=0, seed_nb=1):
	"""
	returns a trained model on the dataset of human demonstrations 
	for each quantile
	"""
	
	filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
	print("training on data: [{}]".format(filename))

	trained_models = []

	start = time.time()

	for seed in range(seed_min, seed_min + seed_nb):

		print("\n\n########## TRAINING FOR SEED #{} ##########".format(seed))

		for q in quantiles:
			# load data
			dataset = Dataset(filename, quantile=q)

			# compile keras models
			model = ClassificationModel(nb_clf=nb_clf,
										input_dim=dataset.obs.shape[-1],
										dataset_name=dataset_name,
										env_name=env_name,
										q=q,
										framework=framework,
										seed=seed)

			# train
			model.fit(dataset.obs, dataset.acs)

			# logging weights and model
			model.save_weights()
			trained_models.append(model)

			del dataset
	
	with open("log/training_time.txt", "a+") as f:
		f.write("\ntraining time on {} dataset using {} on {} was in total {}s and on average {}s per seed, so fast!".format(dataset_name, framework, datetime.now().strftime("%m%d-%H%M%S"), time.time() - start, (time.time()-start)/seed_nb))

def load_models(weights_files_list, env_name):
	obs_dim, acs_dim = (2, 1) if env_name == 'MountainCar-v0' else (11, 27)
	models_list = []
	for filename in weights_files_list:
		model = mlp_classification(obs_dim, acs_dim)
		print("loading weights from file: ", filename)
		model.load_weights(filename)
		models_list.append(model)
	return models_list

def test(env_name='Hopper-v2', dataset_name='ryan', horizon=None, quantiles=[1.0, .5, .25, .125], nb_clf=3, framework='sklearn', seed_min=0, seed_nb=1, aggregate_method='continuous', n_trajectories=10):

	result_list = []
	for seed in range(seed_min, seed_min + seed_nb):

		print("\n\n########## TESTING FOR SEED #{} ##########".format(seed))
		
		# setup
		env = RobustRewardEnv(env_name)
		proxy_rews, true_rews = [], []
		if not horizon:
			horizon = env.max_episode_steps
		
		# loading trained models
		print("aggregate method here is [{}]".format(aggregate_method))
		models_list = [ClassificationModel(nb_clf, env.observation_space.shape[0], dataset_name, env_name, q=q, framework=framework, aggregate_method=aggregate_method) for q in quantiles]
		for model in models_list:
			model.load_weights()
	
		# for all quantiles, collect trajectories
		for model_nb, model in enumerate(models_list):
			start = time.time()
			pi = lambda ob: model.predict(ob)
			ob_list, ac_list, _, proxy_rew_list, true_rew_list = get_trajectories(pi, env, horizon, n_trajectories, play=True)

			proxy_rews.append(proxy_rew_list)
			true_rews.append(true_rew_list)

			print("->testing for q={} took {}s".format(quantiles[model_nb], 
												time.time()-start))

		if not os.path.exists('log/rewards'):
			os.makedirs('log/rewards')	
		np.save('log/rewards/{}_{}_{}_{}_true'.format(dataset_name, env_name, framework, seed), true_rews)
		np.save('log/rewards/{}_{}_{}_{}_proxy'.format(dataset_name, env_name, framework, seed), 
		proxy_rews)
		
		result_list.append([ob_list, ac_list])
	
	return result_list

def plot(env_name, dataset_name, seed_min=0, seed_nb=1, framework='sklearn'):
	for seed in range(seed_min, seed_min + seed_nb):
		proxy_rews_list = np.load('log/rewards/{}_{}_{}_{}_proxy.npy'.format(dataset_name, env_name, framework, seed))
		true_rews_list = np.load('log/rewards/{}_{}_{}_{}_true.npy'.format(dataset_name, env_name, framework, seed))
		qs = [1.0, .5, .25, .125]
		opt_val = [-180.16, -79.79] if env_name == 'MountainCar-v0' else [37.4, 0.603]
		true_rewards = [np.mean([sum(traj) for traj in true_arr]) for true_arr in true_rews_list] + [opt_val[0]]
		proxy_rewards = [np.mean([sum(traj) for traj in proxy_arr]) for proxy_arr in proxy_rews_list] + [opt_val[1]]
		graph_one(true_rewards, proxy_rewards, qs, env_name, dataset_name, framework, seed)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--reg", action="store", default=1e-3, type=float)
	parser.add_argument("--hidden_size", action="store", default=20, type=int)
	parser.add_argument("--dataset_name", action="store", default='ryan', type=str)
	parser.add_argument("--env_name", action="store", default="Hopper-v2", type=str)
	parser.add_argument("--mode", action="store", default="train", type=str)
	parser.add_argument("--seed_min", action="store", default=0, type=int)
	parser.add_argument("--seed_nb", action="store", default=1, type=int)
	parser.add_argument("--framework", action="store", default="keras", type=str)
	parser.add_argument("--aggregate_method", action="store", default="continuous", type=str)
	args = parser.parse_args()
	if (args.mode == "train"):
		train(dataset_name=args.dataset_name, env_name=args.env_name, seed_min=args.seed_min, seed_nb=args.seed_nb, framework=args.framework)
	elif (args.mode == "test"):
		test(args.env_name, dataset_name=args.dataset_name, seed_nb=args.seed_nb, framework=args.framework, aggregate_method=args.aggregate_method)
	elif (args.mode == "plot"):
		plot(args.env_name, args.dataset_name, seed_nb=args.seed_nb, framework=args.framework)
	elif (args.mode == "testplot"):
		test(args.env_name, dataset_name=args.dataset_name,  framework=args.framework, aggregate_method=args.aggregate_method)
		plot(args.env_name, args.dataset_name, seed_nb=args.seed_nb, framework=args.framework)
	elif (args.mode == "full"):
		train(dataset_name=args.dataset_name, env_name=args.env_name, seed_min=args.seed_min, seed_nb=args.seed_nb, framework=args.framework)
		test(args.env_name, dataset_name=args.dataset_name, seed_min=args.seed_min, seed_nb=args.seed_nb, framework=args.framework, aggregate_method=args.aggregate_method)
		plot(args.env_name, args.dataset_name,seed_nb=args.seed_nb, framework=args.framework)
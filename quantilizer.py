from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import gym

import os, time, argparse

from models import Quantilizer
from wrappers import RobustRewardEnv
from dataset import Dataset

def traj_segment_generator(pi, env, max_steps, play=True):

	while True:
		ac = env.action_space.sample()
		ob = env.reset()
		d = True
		prox_rew = 0.0
		true_rew = 0
		obs = np.zeros((max_steps, len(ob)))
		acs = np.zeros((max_steps, 3))
		don = np.zeros((max_steps, 1))
		proxy_rews = np.zeros((max_steps, 1))
		true_rews = np.zeros((max_steps, 1))
		for t in range(max_steps):
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

def get_trajectories(pi, env, max_steps, n_trajectories, play=False):
	gen = traj_segment_generator(pi, env, max_steps, play)

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

def train(dataset_name='ryan', env_name='Hopper-v2', quantiles=[1.0, .5, .25, .125], seed_min=0, seed_nb=1, path=''):
	"""
	returns a trained model on the dataset of human demonstrations 
	for each quantile
	"""

	for seed in range(seed_min, seed_min + seed_nb):

		print("\n\n########## TRAINING FOR SEED #{} ##########".format(seed))

		for q in quantiles:
			# load data
			dataset = Dataset('log/{}/{}.npz'.format(env_name, dataset_name), env_name, q)

			model = Quantilizer(dataset_name=dataset_name,
								env_name=env_name,
								q=q,
								seed=seed,
								path=path)

			# train
			model.fit(dataset)

			# logging weights and model
			model.save_weights()

def test(env_name='Hopper-v2', dataset_name='ryan', quantiles=[1.0, .5, .25, .125], seed_min=0, seed_nb=1, n_trajectories=100, render=False, path=''):

	result_list = []
	for seed in range(seed_min, seed_min + seed_nb):

		print("\n\n########## TESTING FOR SEED #{} ##########".format(seed))
		
		# setup
		env = RobustRewardEnv(env_name)
		proxy_rews, true_rews = [], []
		max_steps = env.max_episode_steps
		
		# loading trained models
		models_list = [Quantilizer(dataset_name=dataset_name,
									env_name=env_name,
									q=q,
									seed=seed,
									path=path) for q in quantiles]
		for model in models_list:
			model.load_weights()
	
		# for all quantiles, collect trajectories
		for model_nb, model in enumerate(models_list):
			start = time.time()
			pi = lambda ob: model.predict(ob)
			ob_list, ac_list, _, proxy_rew_list, true_rew_list = get_trajectories(pi, env, max_steps, n_trajectories, play=render)

			proxy_rews.append(proxy_rew_list)
			true_rews.append(true_rew_list)

			print("->testing for q={} took {}s".format(quantiles[model_nb], 
												time.time()-start))

		if not os.path.exists('log/rewards/{}'.format(path)):
			os.makedirs('log/rewards/{}'.format(path))	
		np.save('log/rewards/{}{}_{}_{}_true'.format(path, dataset_name, env_name, seed), true_rews)
		np.save('log/rewards/{}{}_{}_{}_proxy'.format(path, dataset_name, env_name, seed), 
		proxy_rews)
		
		result_list.append([ob_list, ac_list])
	
	return result_list

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_name", action="store", default='ryan', type=str)
	parser.add_argument("--env_name", action="store", default="VideoPinballNoFrameskip-v4", type=str)
	parser.add_argument("--do",  nargs='+', default=['train'])
	parser.add_argument("--seed_min", action="store", default=0, type=int)
	parser.add_argument("--seed_nb", action="store", default=1, type=int)
	parser.add_argument("--number_trajectories", action="store", default=10, type=int)
	parser.add_argument('--quantiles', nargs='+', default=[1.0], type=float)
	parser.add_argument('--render', default=False, type=bool)
	parser.add_argument('--plotstyle', default=None, type=str)
	parser.add_argument('--path', default='', type=str)
	args = parser.parse_args()

	if 'train' in args.do:
		train(dataset_name=args.dataset_name, env_name=args.env_name, seed_min=args.seed_min, seed_nb=args.seed_nb, quantiles=args.quantiles, path=args.path)
	if 'test' in args.do:
		test(args.env_name, dataset_name=args.dataset_name, seed_min=args.seed_min, seed_nb=args.seed_nb, n_trajectories=args.number_trajectories, quantiles=args.quantiles, render=args.render, path=args.path)
	if 'plot' in args.do:
		from utils.plot import plot
		plot(args.env_name, args.dataset_name, seed_min=args.seed_min, seed_nb=args.seed_nb, quantiles=args.quantiles, plotstyle=args.plotstyle, path=args.path)
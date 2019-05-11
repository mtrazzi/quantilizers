from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from joblib import dump, load
from datetime import datetime
import os, time

from wrappers import RobustRewardEnv

import numpy as np
import tensorflow as tf

PARAMS = {
        'max_steps':        10000,
        'learning_rate':    1e-3,
        'batch_size':       512,
        'weight_decay':     1e-2,
        'print_freq':       1,
        }

class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)
    
    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

class ClassificationModel(object):
	def __init__(self, dataset_name, env_name, q, seed=0, path='', hidden_size=20):
		self.dataset_name = dataset_name
		self.env_name = env_name
		self.seed = seed
		self.q = q
		self.model_list = [MLPClassifier(hidden_layer_sizes=(hidden_size, hidden_size),
										 random_state=self.seed, 
										 max_iter=1000, 
										 verbose=True) for _ in range(3)]
		self.model_path = 'log/models' + '/' + path
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
	
	def filename(self, index):
		return '{}{}_{}_{}_{}f{}.h5'.format(self.model_path, self.dataset_name, self.env_name, self.q, index, self.seed)

	def fit(self, x_train, y_train):
		for index, model in enumerate(self.model_list):
			model.fit(x_train, y_train[:, index])
			train_score = model.score(x_train, y_train[:, index])

	def predict(self, x):
		return [(clf.classes_ * clf.predict_proba(x.reshape(1, -1)).ravel()).sum() for clf in self.model_list]

	def save_weights(self):
		for index, model in enumerate(self.model_list):
			dump(model, self.filename(index))

	def load_weights(self):
		for index, model in enumerate(self.model_list):
			self.model_list[index] = load(self.filename(index))

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

class ConvModel(object):
    def __init__(self, q):
        env = RobustRewardEnv('VideoPinballNoFrameskip-v4')
        self.q = q
        self.net = DQN((4, 84, 84), env.action_space.n) 
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()
        self.running_loss = 0
        self.start_time = time.time()
        self.logger = Logger('log/train/train_{}'.format(datetime.now().strftime("%m%d-%H%M%S")))
    
    def training_step(self, X, y, step):
        self.optimizer.zero_grad()
        out = self.net(X)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.data.item()

        if step % PARAMS['print_freq'] == 0:
            outte = self.net(self.Xte)
            losste = self.criterion(outte, self.yte)
            predte = torch.argmax(outte, 1)
            accte = torch.sum(predte==self.yte).data.item()/len(predte)
            print_str = 'step: {}\t train loss: {}\ttest loss: {}\ttest acc: {}\t time:{}'.format(step, 
                self.running_loss/PARAMS['print_freq'], losste.data.item(), accte, time.time() - self.start_time)
            print(print_str)
            print(print_str, file=open('log.txt', 'a'))
            info = {'train_loss': self.running_loss/PARAMS['print_freq'],'test_loss': losste.data.item(),'test acc': accte}
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, step + 1)
            self.running_loss=0.

    def fit(self, train_set, test_set):
        test_idx = np.random.choice(range(len(test_set)), 1024)
        self.Xte, self.yte = test_set.get_batch_quads(test_idx)
        self.Xte, self.yte = torch.tensor(self.Xte).cuda(), torch.tensor(self.yte).cuda()

        for step in range(PARAMS['max_steps']):
            idx = np.random.choice(range(len(train_set)), PARAMS['batch_size'])
            X, y = train_set.get_batch_quads(idx)
            X = torch.tensor(X).cuda()
            y = torch.tensor(y).cuda()
            self.training_step(X,y, step)

class Quantilizer(object):
    def __init__(self, dataset_name, env_name, q, seed=0, path=''):
        self.env_name = env_name
        if env_name in ['MountainCar-v0', 'Hopper-v2']:
            self.model = ClassificationModel(dataset_name, env_name, q, seed=0, path='')
        elif env_name in ['VideoPinballNoFrameskip-v4']:
            self.model = ConvModel(q)
    
    def fit(self, dataset):
        if self.env_name in ['MountainCar-v0', 'Hopper-v2']:
            self.model.fit(dataset.obs, dataset.acs)
        elif self.env_name in ['VideoPinballNoFrameskip-v4']:
            self.model.fit(dataset.train_set, dataset.test_set)
    
    def save_weights(self):
        pass
    
    def load_weights(self):
        pass
    
    def predict(ob, self):
        return 0
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from joblib import dump, load
import os

PARAMS = {
        'max_steps':        10000,
        'learning_rate':    1e-3,
        'batch_size':       512,
        'weight_decay':     1e-2
        }

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
        env = make_env(PARAMS, clip_rewards=False)
        self.q = q
        self.net = DQN((4, 84, 84), env.action_space.n) 
        self.net.cuda()
        self.optimizer = optim.Adam(net.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()

class Quantilizer(object):
    def __init__(self, dataset_name, env_name, q, seed=0, path=''):
        if env_name in ['MountainCar-v0', 'Hopper-v2']:
            self.model = ClassificationModel(dataset_name, env_name, q, seed=0, path='')
        elif env_name in ['VideoPinballNoFrameskip-v4']:
            self.model = ConvModel(q)
    
    def fit(self, X, y):
        pass
    
    def save_weights(self):
        pass
    
    def load_weights(self):
        pass
    
    def predict(ob, self):
        return 0
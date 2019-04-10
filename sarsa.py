#heavily based on mountain-car-SARSA-AC
import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing
from wrappers import RobustRewardEnv


# Normalize and turn into feature
def featurize_state(state, scaler, featurizer):
	# Transform data
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized

def Q(state,action,weight):
	value = state.dot(weight[action])
	return value

# Epsilon greedy policy
def policy(state, weight, nA, epsilon=0.1):
	A = np.ones(nA,dtype=float) * epsilon/nA
	best_action =  np.argmax([Q(state,a,weight) for a in range(nA)])
	A[best_action] += (1.0-epsilon)
	sample = np.random.choice(nA,p=A)
	return sample

def train(num_episodes, discount_factor=.99, alpha=.01):
    #env = gym.make('MountainCar-v0')
    env = RobustRewardEnv('MountainCar-v0')
    nA = env.action_space.n

    #Parameter vector define number of parameters per action based on featurizer size
    weight = np.zeros((nA,400))

    # Plots
    ep_rewards = np.zeros(num_episodes)
    ep_performances = np.zeros(num_episodes)

    # Get satistics over observation space samples for normalization
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Create radial basis function sampler to convert states to features for nonlinear function approx
    featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                    ])
    # Fit featurizer to our scaled inputs
    featurizer.fit(scaler.transform(observation_examples))

    # Our main training loop
    for e in range(num_episodes):
            state = env.reset()
            state = featurize_state(state, scaler, featurizer)

            while True:
                    #env.render()
                    # Sample from our policy
                    action = policy(state, weight, nA)
                    # Step environment and get next state and make it a feature
                    next_state, reward, done, info = env.step(action)
                    next_state = featurize_state(next_state, scaler, featurizer)

                    # Figure out what our policy tells us to do for the next state
                    next_action = policy(next_state, weight, nA)

                    # Statistic for graphing
                    ep_rewards[e] += reward
                    ep_performances[e] += info['performance']

                    # Figure out target and td error
                    #target = reward + discount_factor * Q(next_state,next_action,weight)		
                    target = reward + discount_factor * max([next_state.dot(weight[a]) for a in range(nA)])
                    td_error = Q(state,action,weight) - target

                    # Find gradient with code to check it commented below (check passes)
                    dw = (td_error).dot(state)
                    
                    # Update weight
                    weight[action] -= alpha * dw

                    if done:
                            break
                    # update our state
                    state = next_state
    env.close()
    return ep_rewards, ep_performances

if __name__=='__main__':
    num_episodes = 200
    ep_rewards, ep_performances = train(num_episodes=num_episodes)
    print('average proxy rew: {}'.format(ep_rewards[-100:].mean()))
    print('average true rew: {}'.format(ep_performances[-100:].mean()))
    plt.figure()
    # Plot the reward over all episodes
    plt.plot(np.arange(num_episodes),ep_rewards)
    plt.plot(np.arange(num_episodes),ep_performances, c='r')
    plt.show()
    # plot our final Q function


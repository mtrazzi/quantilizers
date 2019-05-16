import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import seaborn as sns
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

from collections import Counter
from joblib import dump, load
from datetime import datetime
import os, time, argparse
import itertools

from dataset import Dataset
from utils.wrappers import RobustRewardEnv
from quantilizer import test

OPTIMISER_VALUES = {
    'MountainCar-v0':               [-1.8016, -0.07979],
    'Hopper-v2':                    [37.4, 0.603],
    'VideoPinballNoFrameskip-v4':   [7200, 7200.089]
}

OPTIMISER_NAMES = {
    'MountainCar-v0':               'SARSA',
    'Hopper-v2':                    'PPO',
    'VideoPinballNoFrameskip-v4':   'Rainbow',
}

PROXY_NAMES = {
    'MountainCar-v0':               'Explicit reward (U)',
    'Hopper-v2':                    'Explicit reward (U)',
    'VideoPinballNoFrameskip-v4':   'Implicit loss (-I/λ)',
}

COLORS = {
    'MountainCar-v0':               {'true': 'g', 'proxy': 'r'},
    'Hopper-v2':                    {'true': 'g', 'proxy': 'r'},
    'VideoPinballNoFrameskip-v4':   {'true': 'y', 'proxy': 'r'},
}

def load_ep_rets(filename):
    data = np.load(filename)['ep_rets']
    return data

def plot_rets(file_list, labels=None):
    for filename in file_list:
        plt.plot(load_ep_rets(filename), label=filename)
    plt.legend(file_list, labels) if labels else plt.legend(file_list, file_list)
    plt.show()

def just_plot(array_list):
    for array in array_list:
        plt.plot(array)
    plt.show()

def smoothed_rets(filename, factor=0.99):
    data = load_ep_rets(filename)
    smoothed_data = []
    weighted_average = 0
    for ret in data:
        weighted_average = factor * weighted_average + (1-factor) * ret
        smoothed_data.append(weighted_average)
    return np.array(smoothed_data)

def smoothed_plt_plot(data, label="", factor=0.9):
    smoothed_data = []
    #weighted_average = np.mean(data)
    weighted_average = data[0]
    for ret in data:
        weighted_average = factor * weighted_average + (1-factor) * ret
        smoothed_data.append(weighted_average)
    plt.plot(np.array(smoothed_data), label=label)

def plot_smoothed_rets(file_list, factor=0.99, labels=None):
    if not labels:
        labels = file_list
    for i in range(len(file_list)):
        plt.plot(smoothed_rets(file_list[i]), label=labels[i])
    plt.legend()
    plt.show()

def plot_scatter_plot(path_list, label_list, proxy='mean', alpha=0.2):
    for index, filename in enumerate(path_list):
        data = np.load(filename)
        true_rews = data['ep_rets']
        if proxy == 'mean':
            proxy_rews = [np.mean(traj[:,4]) for traj in data['obs']]
        elif proxy == 'mean_padding':
            proxy_rews = [np.sum(traj[:,4]) / np.count_nonzero(traj[:,4]) for traj in data['obs']]
        elif proxy == 'sum':
            proxy_rews = [np.sum(traj[:,4]) for traj in data['obs']]
        plt.xlabel("average forward learn / ankle angle (explicit reward)")
        plt.ylabel("moving to the right (true reward)")
        plt.scatter(proxy_rews, true_rews, alpha=alpha)

        # cf. https://stackoverflow.com/a/50199100
        gradient, intercept, r_value, _, _ = stats.linregress(proxy_rews,true_rews)
        x = np.linspace(np.min(proxy_rews),np.max(proxy_rews),500)
        y = gradient * x + intercept
        plt.plot(x,y,'-r')

        # log and plots
        plt.title(label_list[index] + ' (r={})'.format(str(r_value)[:4]))
        plt.savefig('log/fig/{}'.format(label_list[index]))
        plt.show()

def average_performance(path_list, label_list):
    for index, filename in enumerate(path_list):
        data = np.load(filename)
        true_rews = data['ep_rets']
        proxy_rews = [np.mean(traj[:,4]) for traj in data['obs']]
        average_true = np.mean(true_rews)
        average_proxy = np.mean(proxy_rews)
        print("for {} the mean of true rewards over all trajectories is {} but for proxy reward it's actually {}".format(label_list[index], average_true, average_proxy))

def average_performance_quantile(dataset_name_list, label_list,
                        env_name='Hopper-v2', quantiles = [1.0, .5, .25, .125]):

    for index, dataset_name in enumerate(dataset_name_list):
        proxy_rews_list = np.load('log/rewards/{}_{}_true.npy'.format(dataset_name, env_name))
        true_rews_list = np.load('log/rewards/{}_{}_proxy.npy'.format(dataset_name, env_name))
        for i_q, q in enumerate(quantiles):
            tr_iq, pr_iq = true_rews_list[i_q], proxy_rews_list[i_q]
            average_true = np.mean([sum(traj) for traj in tr_iq])
            average_proxy = np.mean([sum(traj) for traj in pr_iq])
            print("for q={} dataset={} true rewards mean={} and proxy reward mean = {}".format(q, label_list[index], average_true, average_proxy))

def transform_labels(labels):
    result = np.zeros(labels.shape[0])
    for index, prediction in enumerate(labels):
        pos_pred = prediction + 1
        result[index] = pos_pred[0] * (3 ** 2) + pos_pred[1] * 3 + pos_pred[2]
    return result

def predicted_labels(dataset_name='ryan', env_name='Hopper-v2', seed=0, q=1.0, test_fraction = 0.01):
    # load the dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    #print("confusion matrix for data: [{}]".format(filename))
    dataset = Dataset(filename, quantile=q)

    # load the model
    from quantilizer import ClassificationModel
    model = ClassificationModel(dataset_name=dataset_name,
                                env_name=env_name,
                                q=q,
                                seed=seed)
    model.load_weights()

    # compute the necessary vectors
    _, x_test, _, y_test = train_test_split(dataset.obs, dataset.acs, test_size=test_fraction)
    y_pred = np.array([model.predict(ob) for ob in x_test]).astype(int)
    y_true = y_test.astype(int)

    return transform_labels(y_true), transform_labels(y_pred)

def print_confusion_matrix(dataset_name='michael', env_name='Hopper-v2', seed=0, q=1.0, test_fraction = 0.01):

    # print the confusion matrix after transforming labels to integers
    proc_true, proc_pred = predicted_labels(dataset_name=dataset_name, env_name=env_name, seed=seed, q=q, test_fraction=test_fraction)
    mat = confusion_matrix(proc_true, proc_pred, labels=range(27))
    
    index_list = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    labels = ["[{}, {}, {}]".format(i, j, k) for (i, j, k) in itertools.product(*index_list)]

    # delete the correct labels
    for i in range(len(mat)):
        mat[i][i] = 0
    
    mat = mat / mat.sum()

    df_cm = pd.DataFrame(mat, index = labels, columns = labels)

    print("confusion matrix is: \n", mat)
    plt.figure()
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 5})
    plt.show()

def format_sequence(labels):

    # start your color sequence with only black dots
    color_sequence = ['k'] * len(labels)
    marker_sequence = ['.'] * len(labels)

    # color the most important actions accordingly
    d = {12:'r', 13:'b', 14:'g', 16:'y', 22:'c'}
    m = {12:'>', 13:'x', 14:'<', 16:'3', 22:'4'}
    for index,value in enumerate(labels):
        for key in d:
            if key == value:
                color_sequence[index] = d[key]
                marker_sequence[index] = m[key]
    return color_sequence, marker_sequence

def format_sequence_per_classifier(labels, axis=0):
    # start your color sequence with only black dots
    color_sequence = ['k'] * len(labels)

    # color the most important actions accordingly
    d = {0:'r', 1:'k', 2:'b'}
    for index,value in enumerate(labels.astype(int)):
        color_sequence[index] = d[(value // (3 ** axis)) % 3]
    print("colors for axis={} is: {}".format(axis, Counter(color_sequence)))
    return color_sequence

def pad_with_zeros(obs, acs):
    n_eps = len(obs)
    ep_length = max([len(i) for i in obs])
    padded_obs = np.zeros((n_eps, ep_length, obs[0][0].shape[0]))
    for i, traj in enumerate(obs):
        padded_obs[i,:len(traj),:] = traj
    
    def pad3d(arr, shape, val=0.):
        #pad variable second dimension
        out = np.full(shape, val)
        for i, mat in enumerate(arr):
            out[i,:len(mat),:] = np.array(mat)
        return out
    
    padded_acs = pad3d(acs, (n_eps, ep_length, len(acs[0][0])))
    return padded_obs, padded_acs

def plot_dataset(dataset_name='ryan', env_name='Hopper-v2', quantile=1.0, n_components=3, alpha=0.02):
    
    # load dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    dataset = Dataset(filename, quantile=quantile)
    
    # load / fit pca
    path = 'log/models/{}.pca'.format(dataset_name)
    if not os.path.exists(path):
        pca = PCA(n_components=n_components)
        start = time.time()
        print("fitting pca for data: [{}]".format(filename))
        pca.fit(dataset.obs)
        print("fitting pca took {}s!".format(time.time()-start))
        print("In original dataset, distribution over actions is: {}".format(Counter(transform_labels(dataset.acs))))
        dump(pca, path)
    else:
        pca = load(path)
        
    # plot dataset
    fig = plt.figure(figsize=(100,100))
    for axis in range(3):
        print("starting to plot axis={} of dataset with alpha={} at: [{}]".format(axis, alpha, time.ctime(time.time())))
        start = time.time()
        colors_dataset = format_sequence_per_classifier(transform_labels(dataset.acs), axis=axis)
        ax2 = fig.add_subplot(111, projection='3d')
        ax2.scatter(dataset.obs[:, 0], dataset.obs[:, 1], dataset.obs[:, 2], c=colors_dataset, linewidth=0, alpha=alpha)
        dataset_save_path = 'log/fig/{}_{}_pca_{}_{}d_alpha{}_classif#{}'.format(env_name, dataset_name, int(1000 * quantile), n_components, str(alpha)[2:], axis)
        plt.savefig(dataset_save_path)
        print("plotting time was {}s".format(int(time.time() - start)))

def plot_rollouts(dataset_name='ryan', env_name='Hopper-v2', quantile=1.0, n_components=3, nb_trajectories=100, seed=3, alpha=0.2):
    
    # load pca components already fitted to dataset
    pca = load('log/models/{}.pca'.format(dataset_name))
    
    # transform observations from rollouts according to fitted pca
    results_list = test(env_name=env_name, dataset_name=dataset_name, n_trajectories=nb_trajectories, quantiles=[quantile], seed_min=seed)
    obs, acs = pad_with_zeros(*results_list[0])
    obs = obs.reshape(-1, obs.shape[-1])
    acs = acs.reshape(-1, acs.shape[-1])
    obs = pca.transform(obs)

    # load dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    print("loading dataset: [{}]".format(filename))
    dataset = Dataset(filename, quantile=quantile)
    
    # plot pca for each "classifier axis"
    for axis in range(3):
        colors_rollouts = format_sequence_per_classifier(transform_labels(acs), axis=axis)

        if n_components == 2:
            plt.scatter(obs[:, 0], obs[:, 1], c=colors_rollouts, alpha=alpha)
        elif n_components == 3:
            fig = plt.figure(figsize=(100,100))
            plt.title("pca for top quantile q={} of {}'s dataset".format(quantile, dataset_name))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c=colors_rollouts, linewidth=0, alpha=alpha)
            rollouts_save_path = 'log/fig/seed#{}_axis#{}'.format(seed,axis)
            plt.savefig(rollouts_save_path)

def predict_dataset(dataset_name='ryan', env_name='Hopper-v2', quantile=1.0, n_components=3, nb_trajectories=100, seed=3, alpha=0.2, path="", nb_clf=3):
    
    # load model
    env = RobustRewardEnv(env_name)
    from quantilizer import ClassificationModel
    model = ClassificationModel(dataset_name, env_name, q=quantile, seed=seed, path=path)
    model.load_weights()

    # load dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    print("loading dataset: [{}]".format(filename))
    dataset = Dataset(filename, quantile=quantile)

    # we dump the predicted dataset because it's very long to predict
    path = 'log/{}/{}.pred'.format(env_name, dataset_name)
    if not os.path.exists(path):
        # predict dataset
        print("Starting to predict the dataset:")
        acs = []
        for index, ob in enumerate(dataset.obs):
            print("prediction observation #{}/{}".format(index, len(dataset.obs)))
            acs.append(model.predict(ob))
        acs = np.array(acs)
    else:
        acs = load(path)

    # fit pca components to dataset
    pca = load('log/models/{}.pca'.format(dataset_name))
    obs = pca.transform(dataset.obs)

    obs = dataset.obs.reshape(-1, dataset.obs.shape[-1])
    acs = acs.reshape(-1, acs.shape[-1])

    # plot pca for each "classifier axis"
    for axis in range(3):
        colors = format_sequence_per_classifier(transform_labels(acs), axis=axis)

        if n_components == 2:
            plt.scatter(obs[:, 0], obs[:, 1], c=colors, alpha=alpha)
        elif n_components == 3:
            fig = plt.figure(figsize=(100,100))
            plt.title("pca for top quantile q={} of {}'s dataset".format(quantile, dataset_name))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c=colors, linewidth=0, alpha=alpha)
            rollouts_save_path = 'log/fig/predict_dataset_{}_{}_{}_pca_{}_{}d_classif#{}'.format(env_name, dataset_name, int(1000 * quantile), n_components, axis)
            plt.savefig(rollouts_save_path)

def plot_seeds(tr, pr, quantiles, env_name, dataset_name, m=MaxNLocator, width=.35, title=""):
    """ 
    takes as input a list [true rewards for seed i, true rewards for seed (i+1), ...] and a proxy reward list, 
    and plots all the results for all seeds as a scattered plot
    """

    plt.title("Using same hyperparams as in the paper")
    n_quantiles = len(quantiles)
    n_seeds = len(tr)
    tr_sum, opt_sum = np.zeros(n_quantiles + 1), np.zeros(n_quantiles + 1)
    plt.figure(figsize=(4.3, 3.2))
    seed_ticks = np.arange(n_quantiles + 1)+width/2

    # Initialize the two axes
    ax1 = plt.subplot(111)
    ax1.set_ylabel("True Reward (V)")
    ax2 = ax1.twinx()
    ax2.set_ylabel(PROXY_NAMES[env_name])

    ### Plot black datapoints for each seeds
    for i in range(n_seeds):
        ax1.plot(seed_ticks - width/2, tr[i], 'o', color='black')
        optimisation_values =  (np.array(pr[i]) - np.array(tr[i])) if env_name == 'VideoPinballNoFrameskip-v4' else pr[i]
        ax2.plot(seed_ticks + width/2, optimisation_values, 'o', color='black')
        tr_sum += tr[i]
        opt_sum += optimisation_values
    
    ### Plot the average of true reward per seeds
    bar1 = ax1.bar(np.arange(len(tr_sum)), tr_sum / n_seeds, width, label="True reward", color=COLORS[env_name]['true'])
    bar2 = ax2.bar(np.arange(len(opt_sum)) + width, opt_sum / n_seeds, width, label=PROXY_NAMES[env_name], color=COLORS[env_name]['proxy'])

    xticks = ["imitation"] + [str(i) for i in quantiles[1:]] + [OPTIMISER_NAMES[env_name]]
    #xticks = [str(i) for i in quantiles] + [optimiser]
    
    plt.xticks(np.arange(len(tr_sum))+width/2, xticks)
    plt.xlabel("q values")
    lines = (bar1, bar2)
    plt.legend(loc='upper left')
    labels = [l.get_label() for l in lines]
    ax1.yaxis.set_major_locator(m(nbins=n_quantiles))
    ax2.yaxis.set_major_locator(m(nbins=n_quantiles))
    filename = 'log/fig/multiseed_{}_{}_{}'.format(dataset_name, env_name, datetime.now().strftime("%m%d-%H%M%S"))
    print("saving results in {}.png".format(filename))
    if not os.path.exists('log/fig'):
        os.makedirs('log/fig')
    plt.savefig(filename)
    # plt.show()
    plt.close()

def plot_distribution(tr_list, pr_list, env_name, dataset_name, quantile, seed_min, seed_nb):
    from dataset import Dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    dataset = Dataset(filename, quantile=quantile)
    sns.distplot(dataset.ep_rets, label='true reward in dataset')
    for i in range(seed_nb):
        sns.distplot([sum(traj) for traj in tr_list[i]], label='seed {}'.format(seed_min + i))
    plt.legend(loc='upper left')
    plt.savefig('log/fig/tr_distribution_{}'.format(datetime.now().strftime("%m%d-%H%M%S")))
    plt.close()
    sns.distplot(dataset.proxy, label='proxy reward in dataset')
    for i in range(seed_nb):
        sns.distplot([sum(traj) for traj in pr_list[i]], label='seed {}'.format(seed_min + i))
    plt.legend(loc='upper left')
    plt.savefig('log/fig/pr_distribution_{}'.format(datetime.now().strftime("%m%d-%H%M%S")))
    plt.close()

def boxplot(tr, pr, quantiles, dataset_name):
    """ 
    takes as input a list [true rewards for seed i, true rewards for seed (i+1), ...] and a proxy reward list, 
    and plots all the results for all seeds as a boxplot
    """

    def plot_boxplot(arr, dataset_name, title, quantiles):
        n_quantile = len(arr[0])
        quantile_list = [[arr_seed[i] for arr_seed in arr] for i in range(n_quantile)]
        plt.title(title)
        plt.boxplot(quantile_list, patch_artist=True, sym="", whis=[5, 95], labels=[str(q) for q in quantiles] + ['PPO'])
        filename = 'log/fig/boxplot_{}_{}_{}'.format(dataset_name, title, datetime.now().strftime("%m%d-%H%M%S"))
        plt.savefig(filename)
        plt.close()

    # Proxy reward
    plot_boxplot(tr, dataset_name, 'true_reward', quantiles)
    plot_boxplot(pr, dataset_name, 'proxy_reward', quantiles)

def plot(env_name, dataset_name, seed_min=0, seed_nb=1, quantiles=[1.0, .5, .25, .125], plotstyle=None, path=''):
    tr_list, pr_list = [], []
    opt_val = OPTIMISER_VALUES[env_name]

    
    for seed in range(seed_min, seed_min + seed_nb):
        # loading rewards from testing
        proxy_filename = 'log/rewards/{}{}_{}_{}_proxy.npy'.format(path, dataset_name, env_name, seed)
        true_filename = 'log/rewards/{}{}_{}_{}_true.npy'.format(path, dataset_name, env_name, seed)
        proxy_rews_list, true_rews_list = np.load(proxy_filename, allow_pickle=True), np.load(true_filename, allow_pickle=True)

        # printing specific stats
        tr_imit = [sum(traj) for traj in true_rews_list[0]]
        pr_imit = [sum(traj) for traj in proxy_rews_list[0]]
        print('[n={} seed={}]: tr={}+/-{} (med={}) and pr={}+/-{} (med={})'.format(len(tr_imit), seed, np.mean(tr_imit), np.std(tr_imit), np.median(tr_imit), np.mean(pr_imit), np.std(pr_imit), np.median(pr_imit)))
        
        def apply_function(f, arr_list, opt_value):
            return [f([np.mean(t) for t in arr]) for arr in arr_list] + [opt_value]

        def append_results(f, tr, pr, tr_arr, pr_arr):
            tr.append(apply_function(f,tr_arr, opt_val[0]))
            pr.append(apply_function(f,pr_arr, opt_val[1]))

        # book-keeping for the multiple seeds plot
        if plotstyle in ['mean_seeds', 'boxplot']:
            append_results(np.mean, tr_list, pr_list, true_rews_list, proxy_rews_list)
        if plotstyle == 'distribution':
            tr_list.append(tr_imit)
            pr_list.append(pr_imit)

    if plotstyle == 'distribution':
        plot_distribution(tr_list, pr_list, env_name, dataset_name, quantiles[0], seed_min, seed_nb)
    elif plotstyle == 'mean_seeds':
        plot_seeds(tr_list, pr_list, quantiles, env_name, dataset_name)
    elif plotstyle == 'boxplot':
        boxplot(tr_list, pr_list, quantiles, dataset_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path_list', nargs='+', help='<Required> Set flag')
    parser.add_argument('-D','--dataset_list', nargs='+')
    parser.add_argument('-l','--label_list', nargs='+', default=None)
    parser.add_argument('--discount', default=0.99)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset_name', default='ryan')
    parser.add_argument('--quantile', default=1.0, type=float)
    parser.add_argument('--nb_trajectories', default=100, type=int)
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--alpha', default=0.02, type=float)
    parser.add_argument('--path', default='', type=str)
    args = parser.parse_args()
    if (args.mode == 'smooth'):
        plot_smoothed_rets(args.path_list, args.discount, args.label_list)
    elif (args.mode == 'scatter'):
        plot_scatter_plot(args.path_list, args.label_list)
    elif (args.mode == 'average'):
        average_performance(args.path_list, args.label_list)
    elif (args.mode == 'quantiles'):
        average_performance_quantile(args.dataset_list, args.label_list)
    elif (args.mode == 'confusion_matrix'):
        print_confusion_matrix()
    elif (args.mode == 'pca_dataset'):
        plot_dataset(dataset_name=args.dataset_name, alpha=args.alpha)
    elif (args.mode == 'pca_rollouts'):
        plot_rollouts(quantile=args.quantile, dataset_name=args.dataset_name, nb_trajectories=args.nb_trajectories, seed=args.seed)
    elif (args.mode == 'predict_dataset'):
        predict_dataset(quantile=args.quantile, dataset_name=args.dataset_name, nb_trajectories=args.nb_trajectories, seed=args.seed, path=args.path)


if __name__ == '__main__':
    main()
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse 
from scipy import stats
from quantilizer import ClassificationModel
from wrappers import RobustRewardEnv
from dataset import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

def load_ep_rets(filename, debug=False):
    data = np.load(filename)['ep_rets']
    #print("shape of loaded ep_rets from {} is {}".format(filename, data.shape))
    if debug:
        import ipdb; ipdb.set_trace()
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

def plot_smoothed_rets(file_list, factor=0.99, labels=None):
    if not labels:
        labels = file_list
    for i in range(len(file_list)):
        plt.plot(smoothed_rets(file_list[i]), label=labels[i])
    plt.legend()
    plt.show()

def plot_scatter_plot(path_list, label_list):
    for index, filename in enumerate(path_list):
        data = np.load(filename)
        true_rews = data['ep_rets']
        proxy_rews = [np.mean(traj[:,4]) for traj in data['obs']]
        plt.xlabel("average forward learn / ankle angle (explicit reward)")
        plt.ylabel("moving to the right (true reward)")
        plt.scatter(proxy_rews, true_rews, 1)

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
        result[index] = pos_pred[0] + pos_pred[1] * 3 + pos_pred[2] * (3 ** 2)
    return result

def print_confusion_matrix(aggregate_method='argmax', nb_clf=3, dataset_name='michael', env_name='Hopper-v2', framework='sklearn', seed=0, q=1.0, test_fraction = 0.2):

    # load the dataset
    filename = 'log/{}/{}.npz'.format(env_name, dataset_name)
    print("confusion matrix for data: [{}]".format(filename))
    dataset = Dataset(filename, quantile=q)

    # load the model
    model = ClassificationModel(nb_clf=nb_clf,
                                input_dim=dataset.obs.shape[-1],
                                dataset_name=dataset_name,
                                env_name=env_name,
                                q=q,
                                framework=framework,
                                aggregate_method=aggregate_method,
                                seed=seed)
    model.load_weights()

    # compute the necessary vectors
    _, x_test, _, y_test = train_test_split(dataset.obs, dataset.acs, test_size=test_fraction)
    y_pred = np.array([model.predict(ob) for ob in x_test]).astype(int)
    y_true = y_test.astype(int)

    # print the confusion matrix after transforming labels to integers
    proc_true, proc_pred = transform_labels(y_true), transform_labels(y_pred)
    mat = confusion_matrix(proc_true, proc_pred)
    
    print("confusion matrix is: \n", mat)
    
    import ipdb; ipdb.set_trace()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path_list', nargs='+', help='<Required> Set flag')
    parser.add_argument('-D','--dataset_list', nargs='+')
    parser.add_argument('-l','--label_list', nargs='+', default=None)
    parser.add_argument('-d', '--debug', default=False)
    parser.add_argument('--discount', default=0.99)
    parser.add_argument('--mode', required=True)
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

if __name__ == '__main__':
    main()
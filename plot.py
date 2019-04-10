import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path_list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-l','--label_list', nargs='+', default=None)
    parser.add_argument('-d', '--debug', default=False)
    parser.add_argument('--discount', default=0.99)
    args = parser.parse_args()
    plot_smoothed_rets(args.path_list, args.discount, args.label_list)

if __name__ == '__main__':
    main()
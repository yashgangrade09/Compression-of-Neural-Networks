import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.misc import logsumexp

def flatten_1(list_arr):
    temp = np.concatenate([l.flatten() for l in list_arr])
    return temp.reshape(len(temp), 1)

def invert_flatten_1(input_arr, shaped_arr):
    input_arr = list(input_arr)
    temp = np.copy(shaped_arr)

    for i, arr in enumerate(shaped_arr):
        sample_count = arr.size
        temp2 = input_arr[:sample_count]
        del input_arr[:sample_count]
        temp[i] = np.asarray(temp2).reshape(arr.shape)

    return temp

## Method to compare and merge the components 

def cmp_merge(inputs):
    """Comparing and merging components."""
    for iterator in range(3):
        arr_list = []
        for it in inputs:
            for i in it:
                temp = 1
                for k in arr_list:
                    if i in k:
                        for j in it:
                            k.append(j)
                        temp = 0
                if temp is 1:
                    arr_list.append(list(it))
        arr_list = [np.unique(k) for k in arr_list]
        inputs = arr_list

    return arr_list

def KL_Divergence(means, log_precision):
    # since the input is logarithm of precision, we take exponential of that to compensate the log
    precision = np.exp(log_precision)
    temp = ((log_precision[0] - log_precision[1])/2)  + \
           precision[1]/2. * (1. / precision[0] + (means[0] - means[1])**2) - (1/2)
    return temp


def responsibilities(x_vec, mean_vec, log_precision, pi_vec):
    x_vec = x_vec.flatten()
    result = np.zeros((len(pi_vec), len(x_vec)))
    for k in range(len(pi_vec)):
        result[k] = pi_vec[k] * np.exp(0.5 * log_precision[k]) * np.exp(-np.exp(log_precision[k]) / 2 * (x_vec - mean_vec[k]) ** 2)

    return np.argmax(result, axis=0)


def discretesize(W, pi_zero=0.999):
    # flattening hte weights
    weights = flatten_1(W[:-3])

    means = np.concatenate([np.zeros(1), W[-3]])
    log_precision = W[-2]
    log_pi_vec = np.concatenate([np.log(pi_zero) * np.ones(1), W[-1]])

    # classes K
    J = len(log_precision)
    # compute KL-divergence
    K = np.zeros((J, J))
    L = np.zeros((J, J))

    for i, (m1, pr1, pi1) in enumerate(zip(means, log_precision, log_pi_vec)):
        for j, (m2, pr2, pi2) in enumerate(zip(means, log_precision, log_pi_vec)):
            K[i, j] = KL_Divergence([m1, m2], [pr1, pr2])
            L[i, j] = np.exp(pi1) * (pi1 - pi2 + K[i, j])

    # merge
    index_x, index_y = np.where(K < 1e-10)
    lists = merger(np.asarray(zip(index_x, index_y)))
    # compute merged components
    # print lists
    new_means, new_log_precision, new_log_pi_vec = [], [], []

    for l in lists:
        new_log_pi_vec.append(logsumexp(log_pi_vec[l]))
        new_means.append(
            np.sum(means[l] * np.exp(log_pi_vec[l] - np.min(log_pi_vec[l]))) / np.sum(np.exp(log_pi_vec[l] - np.min(log_pi_vec[l]))))
        new_log_precision.append(np.log(
            np.sum(np.exp(log_precision[l]) * np.exp(log_pi_vec[l] - np.min(log_pi_vec[l]))) / np.sum(
                np.exp(log_pi_vec[l] - np.min(log_pi_vec[l])))))

    new_means[np.argmin(np.abs(new_means))] = 0.0

    # compute responsibilities
    argmax_responsibilities = responsibilities(weights, new_means, new_log_precision, np.exp(new_log_pi_vec))
    result = [new_means[i] for i in argmax_responsibilities]

    return invert_flatten_1(result, shaped_array=W[:-3])

def save_histogram(W_T,save, upper_bound=200):
        w = np.squeeze(special_flatten(W_T[:-3]))
        plt.figure(figsize=(10, 7))
        sns.set(color_codes=True)
        plt.xlim(-1,1)
        plt.ylim(0,upper_bound)
        sns.distplot(w, kde=False, color="g",bins=200,norm_hist=True)
        plt.savefig("./"+save+".png", bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 7))
        plt.yscale("log")
        sns.set(color_codes=True)
        plt.xlim(-1,1)
        plt.ylim(0.001,upper_bound*5)
        sns.distplot(w, kde=False, color="g",bins=200,norm_hist=True)
        plt.savefig("./"+save+"_log.png", bbox_inches='tight')
        plt.close()

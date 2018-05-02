############### Yeh file dhang se change nahi ho paayi especially Callback wala portion. Didn't know what to do. ###############


import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import imageio
import keras
from IPython import display
from keras import backend
from helpers import flatten_1
from keras import flatten as ft


#### Helper Functions used along with functions from Keras #### 

def trainable_weights_collector(Layer):
    trainable_w_1 = getattr(Layer, 'trainable', True)

    if not trainable_w_1:
        return []

    weight_vec = []

    if Layer.__class__.__name__ == 'Sequential':
        for sl in Layer.flatten_1ed_layers:
            weight_vec += trainable_weights_collector(sl)
    elif Layer.__class__.__name__ == 'Model':
        for sl in Layer.layers:
            weight_vec += trainable_weights_collector(sl)
    else:
        weight_vec += Layer.trainable_weights

    weight_vec = list(set(weight_vec))
    weight_vec.sort(key=lambda x: x.name)
    return weight_vec


def fetch_weights(Model):
    trainable = []
    for it in Model.layers:
        trainable = trainable + trainable_weights_collector(it)
    return trainable


#### Objective functions ####

def identity(Y_TRUE, Y_PRED):
    return Y_PRED


def log_sum_exp(tensor, weights=None, axis=1):
    # Compute
    max_val = backend.max(tensor, axis=axis, keepdims=True)

    if weight is not None:
        logsumexp = weight * backend.exp(tensor - max_val)
    else:
        logsumexp = backend.exp(tensor - max_val)

    logsumexp = backend.log(backend.sum(logsumexp, axis=axis))
    max_val = backend.max(tensor, axis=axis)
    logsumexp = logsumexp + max_val

    return logsumexp


#### Callback functions - just to visualize the training in a progressive manner ####

class VisualizationCallback(keras.callbacks.Callback):

    def __init__(self, Model, X_t, Y_t, ep):
        self.Model = Model
        self.X_t = X_t
        self.Y_t = Y_t
        self.ep = ep

        super(VisualizationCallback, self).__init__()

    def plot_hist(self, epoch):

        # record the weights of the network
        w0 = self.w0
        wt = self.Model.get_weights()
        weight_vec_0 = np.squeeze(flatten_1(w0[:-3]))
        weight_vec_t = np.squeeze(flatten_1(wt[:-3]))

        # fetch the mean, standard deviations, and, mixing proportions

        pi_t = (np.exp(wt[-1]))
        variance_t = 1. / np.exp(wt[-2])
        standard_dev_t = np.sqrt(variance_t)
        mean_t = np.concatenate([np.zeros(1), wt[-3]]).ft()

        # Start Plotting
        x0 = -1.2
        x1 = 1.2

        I = np.random.permutation(len(weight_vec_0))

    f = sns.jointplot(weight_vec_0[I], weight_vec_t[I], size=8, kind="scatter", color="g", stat_func=None,
                      edgecolor='w',
                      marker='o', joint_kws={"s": 8}, marginal_kws=dict(bins=1000), ratio=4)
    f.ax_joint.hlines(mean_t, x0, x1, lw=0.5)

    for k in range(len(mean_t)):
        if k == 0:
            f.ax_joint.fill_between(np.linspace(x0, x1, 10), mean_t[k] - 2 * standard_dev_t[k],
                                    mean_t[k] + 2 * standard_dev_t[k],
                                    color='blue', alpha=0.1)
        else:
            f.ax_joint.fill_between(np.linspace(x0, x1, 10), mean_t[k] - 2 * standard_dev_t[k],
                                    mean_t[k] + 2 * standard_dev_t[k],
                                    color='red', alpha=0.1)
    score = \
        self.model.evaluate({'input': self.X_test, }, {"error_loss": self.Y_test, "complexity_loss": self.Y_test, },
                            verbose=0)[3]
    sns.plt.title("Epoch: %d /%d\nTest accuracy: %.4f " % (epoch, self.ep, score))
    f.ax_marg_y.set_xscale("log")
    f.set_axis_labels("Pretrained", "Retrained")
    f.ax_marg_x.set_xlim(-1, 1)
    f.ax_marg_y.set_ylim(-1, 1)
    display.clear_output()
    f.savefig("./.tmp%d.png" % epoch, bbox_inches='tight')
    plt.show()

    def begin_training(self, logs={}):
        self.w0 = self.Model.get_weights()

    def begin_epoch(self, epoch, logs={}):
        self.plot_hist(epoch)

    def end_training(self, logs={}):
        self.plot_hist(epoch=self.ep)
        img_vec = []
        files = ["./fig%d.png" % epoch for epoch in np.arange(self.ep + 1)]

        for f in files:
            img_vec.append(imageio.imread(f))
            os.remove(f)
        imageio.mimsave('./Figs/ReTraining.gif', img_vec, duration=0.3)

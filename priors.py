import tensorflow as tf 
import numpy as np
from keras_helpers import log_sum_expo
from keras import backend
from keras.engine.topology import Layer
from helper_functions import flatten_1

class GMM_Prior(Layer):

    def __init__(self, num_component, arr_net_weight, arr_pretrained_weight, pi_0, **kwargs):
        self.num_component = num_component
        self.arr_net_weight = [backend.flatten(w) for w in arr_net_weight]
        self.arr_pretrained_weight = flatten_1(arr_pretrained_weight)
        self.pi_0 = pi_0

        super(GMM_Prior, self).__init__(**kwargs)

    def build(self, input_shape):
        J = self.num_component

        mean_i = np.linspace(-0.6, 0.6, J - 1)
        self.means = backend.variable(mean_i, name='means')
    
        standard_devs_i = np.tile(0.25, J) 
        gamma_i = - np.log(np.power(standard_devs_i, 2))
        self.gammas = backend.variable(gamma_i, name='gammas')
       
        mix_prop_i = np.ones((J - 1))
        mix_prop_i *= (1. - self.pi_0) / (J - 1)
        self.rhos = backend.variable(np.log(mix_prop_i), name='rhos')
        
        self.trainable_weights = [self.means] + [self.gammas] + [self.rhos]

    def call(self, x, mask=None):
        J = self.num_component
        loss = backend.variable(0.)
        means = backend.concatenate([backend.variable([0.]), self.means],axis=0)
        precision = backend.exp(self.gammas)
        
        min_rho = backend.min(self.rhos)
        mix_prop = backend.exp(self.rhos - min_rho)
        mix_prop = (1 - self.pi_0) * mix_prop / backend.sum(mix_prop)
        mix_prop = backend.concatenate([backend.variable([self.pi_0]), mix_prop],axis=0)

        for weights in self.arr_net_weight:
            loss = loss + self.compute_loss(weights, mix_prop, means, precision)

        (alpha, beta) = (5000,2)
        negative_log_prop = (1 - alpha) * backend.gather(self.gammas, [0]) + beta * backend.gather(precision, [0])
        loss = loss + backend.sum(negative_log_prop)
        alpha, beta = (250,0.1)
        index = np.arange(1, J)
        negative_log_prop = (1 - alpha) * backend.gather(self.gammas, index) + beta * backend.gather(precision, index)

        return (loss + backend.sum(negative_log_prop))

    def compute_loss(self, weights, mix_prop, means, precision):
        diff = tf.expand_dims(weights, 1) - tf.expand_dims(means, 0)
        unnormalized_log_likelihood = - (diff ** 2) / 2 * backend.flatten(precision)
        Z = backend.sqrt(precision / (2 * np.pi))
        log_likelihood = log_sum_expo(unnormalized_log_likelihood, w=backend.flatten(mix_prop * Z), axis=1)

        return -backend.sum(log_likelihood)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

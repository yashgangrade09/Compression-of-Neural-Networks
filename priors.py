import tensorflow as tf 
import numpy as np
from keras_helpers import log_sum_exp
from keras import backend
from keras.engine.topology import Layer
from helper_functions import flatten_1


class GMMPrior(Layer):
    """Gaussian mixture prior on neural network weights"""

    def __init__(self, num_component, arr_net_weight, arr_pretrained_weight, pi_0, **kwargs):
        self.num_component = num_component
        self.arr_net_weight = [backend.flatten(w) for w in arr_net_weight]
        self.arr_pretrained_weight = flatten_1(arr_pretrained_weight)

        # Fixed mixing proportion for the gaussian component corresponding to mu=0
        self.pi_0 = pi_0

        super(GMMPrior, self).__init__(**kwargs)

    def build(self, input_shape):
        # The build method
        J = self.num_component

        # Evenly spaced means around zero
        mean_i = np.linspace(-0.6, 0.6, J - 1)
        self.means = backend.variable(mean_i, name='means')

        # Gamma = - log(\sigma^2)
        # the negative log means we are essentially dealing with precision
        # instead of variance
        gamma_i = - np.log((np.ones(J) * 0.25) ** 2)
        self.gammas = backend.variable(gamma_i, name='gammas')

        # Mixing proportions for the mixture model
        # Since the mixing proportion for zero-th component is fixed
        # we initialize the rest components uniformly
        mix_prop_i = np.ones(J - 1) * (1. - self.pi_0) / (J - 1)

        # rho is log(tau) - working in log space is stable
        self.rhos = backend.variable(np.log(mix_prop_i), name='rhos')

        # Collect all weights into trainable_weights to let Keras know
        # that this are to be optimized
        self.trainable_weights = [self.means] + [self.gammas] + [self.rhos]

        # Maybe call super() method here?
        self.built = True

    def call(self, x, mask=None):
        J = self.num_component

        # Initialize loss as zero, we'll add terms to it later
        loss = backend.variable(0.)

        # The mean vector = [mu_0, mu_{1,..., J}]
        means = backend.concatenate([backend.variable([0.]), self.means], axis=0)

        # Convert back to 1 / variance = precision
        precision = backend.exp(self.gammas)

        # Get the mixing proportions back from exp(tau)
        # Employ the log_sum_exp trick to prevent over/underflow
        # log \sum exp(x_i) = a + log \sum exp(x_i - a)
        # a = min x_i
        a = backend.min(self.rhos)
        mix_prop = backend.exp(self.rhos - a)
        mix_prop = (1 - self.pi_0) * mix_prop / backend.sum(mix_prop)
        mix_prop = backend.concatenate([backend.variable([self.pi_0]), mix_prop], axis=0)

        # Compute the negative log likelihood of the weights w.r.t to the mixture model
        loss += sum([self.nll(weights, mix_prop, means, precision) for weights in self.arr_net_weight])

        # Gamma hyper-prior parameters - on the zero mean gaussian
        (alpha_0, beta_0) = (5000.0, 2.0)
        negative_log_prob = (1 - alpha_0) * backend.gather(self.gammas, [0]) + beta_0 * backend.gather(precision, [0])
        loss += backend.sum(negative_log_prob)

        # Gamma hyper-prior parameters - on the rest of the gaussians
        alpha, beta = (250, 0.1)
        index = np.arange(1, J)
        negative_log_prob = (1 - alpha) * backend.gather(self.gammas, index) + beta * backend.gather(precision, index)

        loss += backend.sum(negative_log_prob)
        return loss

    def nll(self, weights, mix_prop, means, precision):
        # Trick to compute pairwise distances (gaussian kernel)
        diff = tf.expand_dims(weights, 1) - tf.expand_dims(means, 0)

        # The expression below is equivalent to ||x - mu||^2/(2*sigma^2)
        unnormalized_log_likelihood = - (diff ** 2) / 2 * backend.flatten(precision)

        # Normalizing constant is 1/(2*pi*sigma^2)
        Z = backend.sqrt(precision / (2 * np.pi))

        # Log Likelihood = log(\sum_over_mixture_components ||x-mu||^2/(2*sigma^2) + Z*tau)
        log_likelihood = log_sum_exp(unnormalized_log_likelihood, weights=backend.flatten(mix_prop * Z), axis=1)

        # return negative log likelihood = nll
        return -backend.sum(log_likelihood)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], 1

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
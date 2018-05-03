
# coding: utf-8

# # Compressing neural networks with Gaussian mixture priors

# ---
# ## Introduction
# In this project we have implemented the paper titled [Soft weight-sharing for Neural Network compression](https://arxiv.org/abs/1702.04008) by Ullrich, Meeds and Welling. The main idea of the paper is to introduce a prior on the weights of a pre-trained network that encourages a lot of weights to go to zero and clusters the remaining points around a small number of discrete value. 
# 
# This is done by using a Gaussian Mixture prior over the weights such that the most of the weights map to a gaussian with zero mean and the rest of the weights are quantized to their closest cluster centers.
# From the paper:
# > By fitting the mixture components alongside the weights, the weights tend to concentrate very tightly around a number of cluster components, while the cluster centers optimize themselves to give the network high predictive accuracy. Compression is achieved because we only need to encode K cluster means (in full precision) in addition to the assignment of each weight to one of these J values (using log(J) bits per weight)

# ## Outline of method
# Following are the steps to achieve compression using the methods described in the given paper:
# 1. Retraining a pre trained network with gaussian mixture prior on the weights
# 2. Clustering the weights, merging redundant components and retrain
# 3. Quantize the weights by mapping them to nearest cluster mean

# In[1]:


import numpy as np
import keras
from data import get_mnist

from keras.models import Model
from keras.layers import Input, Dense,  Activation, Flatten, Conv2D
from keras import optimizers


# In[2]:


# Get the training data, this loads the mnist dataset if not already present
X_train, X_test, Y_train, Y_test, img_rows, img_cols, num_classes = get_mnist()

# Create a data input layer
InputLayer = Input(shape=(img_rows, img_cols,1), name="input")

# First convolution layer
conv_1 = Conv2D(25, (5, 5), strides = (2,2), activation = "relu")(InputLayer)
# Second convolution layer
conv_2 = Conv2D(50, (3, 3), strides = (2,2), activation = "relu")(conv_1)

# 2 fully connected layers with RELU activations
conv_output = Flatten()(conv_2)
fc1 = Dense(500)(conv_output)
fc1 = Activation("relu")(fc1)
fc2 = Dense(num_classes)(fc1)
PredictionLayer = Activation("softmax", name ="error_loss")(fc2)

# Fianlly, we create a model object:
reference_model = Model(inputs=[InputLayer], outputs=[PredictionLayer])
reference_model.summary()


# In[3]:


epochs = 10
batch_size = 256

# Adam optimizer
optimizer = optimizers.Adam(lr=0.001)

reference_model.compile(optimizer, loss = {"error_loss": "categorical_crossentropy"}, metrics=["accuracy"])

reference_model.fit(x=X_train, y=Y_train, 
          epochs= epochs, batch_size = batch_size,
          verbose = 1, validation_data=(X_test, Y_test))

score = reference_model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])


# In[4]:


keras.models.save_model(reference_model, "./ref_model")

pre_trained_model = keras.models.load_model("./ref_model")


# In[5]:


from priors import GMMPrior
from keras_helpers import fetch_weights

pi_zero = 0.99

reg_layer = GMMPrior(16, fetch_weights(reference_model), 
                     pre_trained_model.get_weights(), pi_zero, name="complexity_loss")(fc2)

compressed_model = Model(inputs=[InputLayer], outputs=[PredictionLayer, reg_layer])

compressed_model.summary()


# In[ ]:


import optimizers 
from keras_helpers import identity

tau = 0.003
N = X_train.shape[0] 

opt = optimizers.Adam(lr = [5e-4,1e-4,3e-3,3e-3], param_types_dict = ['means','gammas','rhos'])

compressed_model.compile(optimizer = opt,
              loss = {"error_loss": "categorical_crossentropy", "complexity_loss": identity},
              loss_weights = {"error_loss": 1. , "complexity_loss": tau/N},
              metrics = ['accuracy'])


# In[ ]:


epochs = 30
batch_size = 256
compressed_model.fit({"input": X_train,},
          {"error_loss" : Y_train, "complexity_loss": np.zeros((N,1))},
          epochs = epochs,
          batch_size = batch_size,
          verbose = 1)


# ## Part 3 - Post Processing Steps

# In[ ]:


import helper_functions


# In[ ]:


weights_retrain = np.copy(compressed_model.get_weights())
weights_compressed = np.copy(compressed_model.get_weights())
weights_compressed[:-3] = helper_functions.discretesize(np.copy(weights_compressed), pi_zero = pi_zero)


# Next step is to compare the accuracy of the pre-trained network with the network obtained post-processing. The procedure to do that is as follows. 

# In[ ]:


print("The accuracy of model is: \n")

acc = pre_trained_model.evaluate({'input':X_test,}, {"error_loss": Y_test,}, verbose=0)[1]
print("Reference Network: %.4f \n" % acc)

acc2 = compressed_model.evaluate({'input': X_test,}, {"error_loss": Y_test, "complexity_loss": Y_test,}, verbose=0)[3]
print("Re-trained Network: %.4f \n" % acc2)

compressed_model.set_weights(weights_compressed)

acc3 = compressed_model.evaluate({'input': X_test,}, {"error_loss": Y_test, "complexity_loss": Y_test,}, verbose=0)[3]
print("Post Processed Network: %.4f \n" % acc3)


# Now to check the number of weights that were pruned, we do the following procedures. 

# In[ ]:


from helper_functions import special_flatten as flatten_1
weight_vec = flatten_1(weights_compressed[:-3]).flatten()
print("Percentage of Non-Zero Weights: %.3f %%" % (100.* (1 - np.count_nonzero(weight_vec)/ weight_vec.size)))


# In[ ]:


from helper_functions import save_histogram
save_histogram(pre_trained_model.get_weights(), save="Figures/reference")
save_histogram(weights_retrain, save="Figures/retrain")
save_histogram(weights_compressed, save="Figures/Post-Processing")


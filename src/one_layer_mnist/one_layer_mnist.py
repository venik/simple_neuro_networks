#!/usr/bin/env python

# 1 layer perceptron neuro network against MNIST
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from lib.transferfunction import HardLim
from lib.neuron import NeuronLayer

# import data set
data_set = mnist('../../data_set/mnist')

layer = NeuronLayer.generate_weights(
    num_of_neurons = 10,
    num_of_inputs = 784,
    hasBias=False,
    activation_function = HardLim)

# learning
for k in range(0, 59999):
    (label, data) = data_set.get_next_frame()
    print("%s: label:%s" % (str(k), str(label)))

    data_for_neuro = data.reshape(784, 1)
    layer.calculate_local_field(data_for_neuro)

    desired_output = np.zeros(10).reshape(10, 1)
    desired_output[label] = 1
    layer.training(desired_output)

print("Layer: " + str(layer))

# save weights and biases
fd_neuro = open("weights", "wb")

np.savez(fd_neuro, layer.get_weights(), layer.get_biases())

# tear down
data_set.tear_down()
fd_neuro.close()

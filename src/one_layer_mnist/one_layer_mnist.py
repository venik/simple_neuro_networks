#!/usr/bin/env python

# 1 layer perseptron neuronetwork against MNIST
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from lib import HardLim
from lib.neuron import NeuronLayer

# import data set
data_set = mnist('../../data_set/mnist')


layer = NeuronLayer(
    num_of_neurons = 10,
    num_of_inputs = 784,
    hasBias=False,
    activation_function = HardLim)

# learning
for k in range(0, 9997):
    (label, data) = data_set.get_next_train_frame()
    print("%s: label:%s" % (str(k), str(label)))
    # print("Layer: " + str(layer))

    data_for_neuro = data.reshape(784, 1)
    layer.calculate_local_field(data_for_neuro)


    desired_output = np.zeros(10).reshape(10, 1)
    desired_output[label] = 1
    layer.trainining(desired_output)


print("Layer: " + str(layer))

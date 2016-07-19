#!/usr/bin/env python

# 1 layer perceptron neuronetwork against MNIST
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from lib import HardLim
from lib.neuron import NeuronLayer

# import data set
data_set = mnist('../../data_set/mnist', isTrainMode = False)

fd_neuro = open("weights", "r")
npzfile = np.load(fd_neuro)
weights = npzfile['arr_0']
biases = npzfile['arr_1']

layer = NeuronLayer(
    num_of_neurons = 10,
    num_of_inputs = 784,
    hasBias=False,
    activation_function = HardLim,
    weights = weights,
    biases = biases)

correct = 0.0

# check performance
for k in range(0, data_set.get_num_of_frames_in_dataset()):
    (label, data) = data_set.get_next_frame()

    data_for_neuro = data.reshape(784, 1)
    layer.calculate_local_field(data_for_neuro)

    desired_output = np.zeros(10).reshape(10, 1)
    desired_output[label] = 1

    if np.equal(desired_output, layer.get_output()).all():
        correct = correct + 1
        # print("True")
        # print("d:%s\n o:%s" % (str(desired_output), str(layer.get_output())) )
    # else:
        # print("False")
        # print("d:%s\n o:%s" % (str(desired_output), str(layer.get_output())) )

percentage = correct / k * 100
print("correct percentage: %s" % str(percentage))

# tear down
data_set.tear_down()
fd_neuro.close()

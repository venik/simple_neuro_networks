__author__ = 'Sasha Nikiforov nikiforov.al[a]gmail.com'
__copyright__ = "Copyright 2016, Sasha Nikiforov"

__license__ = "GPLv2"
__maintainer__ = "Sasha Nikiforov"
__email__ = "nikiforov.al[a]gmail.com"
__status__ = "Development"

import numpy as np

class NeuronLayer(object):
    _v = 0.0       # local field
    _y = 0.0       # output
    _w = None       # row vector of weights
    _b = None
    _hasBias = False
    _inputs = None

    _target = None

    _alpha = 0.1        # learning rate
    _gamma = 0.0

    _num_of_inputs = 0  # number of the inputs of every neuron
    _num_of_neurons = 0 # number of neurons in the Layer

    _activation_function = None     # transfer function

    def __init__(self, num_of_neurons, num_of_inputs, hasBias, activation_function, weights, biases):
        self._hasBias = hasBias
        self._num_of_inputs = num_of_inputs
        self._num_of_neurons = num_of_neurons

        self._activation_function = activation_function()

        # allocate input vector
        self._inputs = np.zeros((self._num_of_inputs, 1))

        self._w = weights
        self._b = biases

    @classmethod
    def generate_weights(cls, num_of_neurons, num_of_inputs, hasBias, activation_function):
        # [Cristiani-SVM] 2.1.1 Rosenblatt's Perceptron - weights can be initialized with 0
        weights = np.array(np.random.uniform(low = 0,
                                             high = 0,
                                             size = num_of_neurons * num_of_inputs)).reshape(num_of_neurons, num_of_inputs)

        # [Cristiani-SVM] 2.1.1 Rosenblatt's Perceptron - biases can be initialized with 0
        biases = np.array(np.random.uniform(low = 0,
                                             high = 0,
                                             size = num_of_neurons)).reshape(num_of_neurons, 1)

        return cls(num_of_neurons, num_of_inputs, hasBias, activation_function, weights, biases)

    # inputs is numpy.array column vector
    def calculate_local_field(self, inputs):
        # assert(inputs.__len__() == self._w.__len__(), "Input and number of inputs do not match")
        self._inputs = inputs
        self._v = self._w.dot(self._inputs) + self._hasBias * self._b
        self._y = self._activation_function.get_phi(self._v)

    # target - row
    def training(self, target):
        # if np.equal(target, self._y).all():
        #     return

        self._target = target

        # [NND] equation (4.33) for HardLim (TODO: make it part of the transfer function)
        # [NND] equation (7,45) for the Widrow-Hoff learning rule
        e = (self._target - self.get_output()).reshape(self._num_of_neurons, 1)
        self._w = (1 - self._gamma) * self._w + self._alpha * e * self._inputs.transpose()
        self._b = (1 - self._gamma) * self._b + self._alpha * e

    def get_local_field(self):
        return self._v

    def get_output(self):
        return self._y.reshape(self._num_of_neurons, 1)

    def get_weights(self):
        return self._w

    def get_biases(self):
        return self._b

    def __str__(self):
        return "Weights (w): \n%s\n Bias (b):\n%s\nInputs: %s\nLocal field(v):\n%s\nOutput(o):\n%s\nTarget\n:%s" %\
               (str(self._w), str(self._b), "empty", str(self._v), str(self._y), str(self._target))

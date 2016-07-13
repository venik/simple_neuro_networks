# GPLv2 License
# (c) Sasha Nikiforov

import numpy as np

class NeuronLayer(object):
    _v = 0.0       # local field
    _y = 0.0       # output
    _w = None       # row vector of weights
    _zero_input = np.array(([1]))
    _inputs = None

    _num_of_inputs = 0  # number of the inputs of every neuron
    _num_of_neurons = 0 # number of neurons in the Layer

    _activation_function = None     # transfer function

    def __init__(self, num_of_neurons, num_of_inputs, activation_function):
        # every neuron has b (zero index) input
        self._num_of_inputs = num_of_inputs + 1
        self._num_of_neurons = num_of_neurons

        self._activation_function = activation_function()

        # row i - neuron i
        # column j - weight of a neuron in a corresponding row
        self._w = np.ones((self._num_of_neurons, self._num_of_inputs)) / np.sqrt(self._num_of_inputs)

        # allocate input vector
        self._inputs = np.zeros((self._num_of_inputs, 1))

    # inputs is numpy.array column vector
    def calculate_local_field(self, inputs):
        # assert(inputs.__len__() == self._w.__len__(), "Input and number of inputs do not match")
        # add zero input => [1, inputs]
        self._inputs = np.append(self._zero_input, inputs)
        self._v = self._w.dot(self._inputs)

        self._y = self._activation_function.get_phi(self._v)

    def get_local_field(self):
        return self._v

    def get_output(self):
        return self._y

    def __str__(self):
        return "Weights (w): %s\nInputs: %s\nLocal field(v): %s\nOutput(o): %s" %\
               (str(self._w), str(self._inputs), str(self._v), str(self._y))

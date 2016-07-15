# GPLv2 License
# (c) Sasha Nikiforov

import numpy as np

class NeuronLayer(object):
    _v = 0.0       # local field
    _y = 0.0       # output
    _w = None       # row vector of weights
    _b = None
    _hasBias = False
    _inputs = None

    _num_of_inputs = 0  # number of the inputs of every neuron
    _num_of_neurons = 0 # number of neurons in the Layer

    _activation_function = None     # transfer function

    def __init__(self, num_of_neurons, num_of_inputs, hasBias, activation_function):
        self._hasBias = hasBias
        self._num_of_inputs = num_of_inputs
        self._num_of_neurons = num_of_neurons

        self._activation_function = activation_function()

        # row i - neuron i
        # column j - weight of a neuron in a corresponding row
        # self._w = np.ones((self._num_of_neurons, self._num_of_inputs)) / np.sqrt(self._num_of_inputs)
        # self._b = np.array(([0.5, -0.5])).reshape(2, 1)

        # TODO: how to choose proper weights here ??
        self._w = np.array(np.random.uniform(low=-2.0,
                                             high=2.0,
                                             size=self._num_of_neurons*self._num_of_inputs)).reshape(self._num_of_neurons,
                                                                                                     self._num_of_inputs)
        # TODO: how to choose proper weights ????
        self._b = np.array(np.random.uniform(low=-2.0,
                                             high=2.0,
                                             size=self._num_of_neurons)).reshape(self._num_of_neurons, 1)

        # allocate input vector
        self._inputs = np.zeros((self._num_of_inputs, 1))

    # inputs is numpy.array column vector
    def calculate_local_field(self, inputs):
        # assert(inputs.__len__() == self._w.__len__(), "Input and number of inputs do not match")
        self._inputs = inputs
        self._v = self._w.dot(self._inputs) + self._hasBias * self._b
        self._y = self._activation_function.get_phi(self._v)

    # target - row
    def trainining(self, target):
        # if np.equal(target, self._y).all():
        #     return

        # [NND] equation (4.33) for HardLim (TODO: make it part of the transfer function)
        e = (target - self.get_output()).reshape(self._num_of_neurons, 1)
        self._w = self._w + e * self._inputs.transpose()
        self._b = self._b + e

    def get_local_field(self):
        return self._v

    def get_output(self):
        return self._y.reshape(self._num_of_neurons, 1)

    def __str__(self):
        return "Weights (w): \n%s\n Bias (b):\n%s\nInputs: %s\nLocal field(v): %s\nOutput(o): %s" %\
               (str(self._w), str(self._b), str(self._inputs), str(self._v), str(self._y))

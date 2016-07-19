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

    _target = None

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
        # TODO: how to choose proper weights here ??
        weights = np.array(np.random.uniform(low = -255,
                                             high = 255,
                                             size = num_of_neurons * num_of_inputs)).reshape(num_of_neurons, num_of_inputs)

        # TODO: how to choose proper weights ????
        biases = np.array(np.random.uniform(low = -255,
                                             high = 255,
                                             size = num_of_neurons)).reshape(num_of_neurons, 1)

        return cls(num_of_neurons, num_of_inputs, hasBias, activation_function, weights, biases)

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

        self._target = target

        # [NND] equation (4.33) for HardLim (TODO: make it part of the transfer function)
        e = (self._target - self.get_output()).reshape(self._num_of_neurons, 1)
        self._w = self._w + e * self._inputs.transpose()
        self._b = self._b + e

    def get_local_field(self):
        return self._v

    def get_output(self):
        return self._y.reshape(self._num_of_neurons, 1)

    def __str__(self):
        return "Weights (w): \n%s\n Bias (b):\n%s\nInputs: %s\nLocal field(v):\n%s\nOutput(o):\n%s\nTarget\n:%s" %\
               (str(self._w), str(self._b), "empty", str(self._v), str(self._y), str(self._target))

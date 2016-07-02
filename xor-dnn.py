#!/usr/bin/env python
from abc import abstractmethod

import numpy as np

class ActivationFunction(object):
    _threshold = 0

    def __init__(self, threshold):
        self._threshold = threshold

    def get_threshold(self):
        return self._threshold

    @abstractmethod
    def get_phi(self, number_of_neurons, number_of_neurons_next_layer):
        raise NotImplementedError, "ActivationFunction class, get_phi() has to be implemented"

    @abstractmethod
    def get_phi_derivative(self, local_field):
        raise NotImplementedError, "ActivationFunction class, get_phi_derivative() has to be implemented"

class HyperbolicTangent(ActivationFunction):
    # Haykin page 145 - "3. Activation function"
    _a =  1.7159
    _b = 0.6666
    _ab = _a / _b

    def __init__(self, threshold):
        super(HyperbolicTangent, self).__init__(threshold)

    # Haykin page 136 - "2. Hyperbolic tangent function"
    def get_phi(self, local_field):
        return self._a * np.tanh(self._b * local_field)

    def get_phi_derivative(self, local_field):
        return self._ab * (self._a - local_field) * (self._a + local_field)

class Neuron(object):
    _v = None       # local field
    _y = None       # output
    _w = None       # row vector of weights

    # Haykin "Activation function" page 135
    _activation_function = None

    def __init__(self, num_of_inputs, activation_function = HyperbolicTangent):
        self._v = 0.0
        self._y = 0.0
        self._activation_function = activation_function(threshold = 0.0)

        # Haykin, initial weights page 148
        w = [None] * num_of_inputs
        for i in range(w.__len__()):
            w[i] = 1 / np.sqrt(num_of_inputs)

        self._w = np.array((w)).reshape((1, num_of_inputs))

    # inputs is numpy.array column vector
    def calculate_local_field(self, inputs):
        # assert(inputs.__len__() == self._w.__len__(), "Input and number of inputs do not match")
        self._v = self._w.dot(inputs)

        self._y = self._activation_function.get_phi(self._v)

    def get_local_field(self):
        return self._v

    def get_output(self):
        return self._y

    def __str__(self):
        return "Weights (w): %s\nLocal field(v): %s\nOutput(o): %s" % (str(self._w), str(self._v), str(self._y))

class NeuronLayer(object):
    _layer = [None]

    # number_of_neurons - number of neurons in the current layer
    # number_of_inputs - number of th input connection, for every connection we have it's own weight (w)
    def __init__(self, number_of_neurons, number_of_inputs):
        self._layer = [None] * number_of_neurons
        for i in range(number_of_neurons):
            self._layer[i] = Neuron(number_of_inputs)

    def forward(self, input_layer):
        for neuron in self._layer:
            neuron.calculate_local_field(input_layer)

    def __str__(self):
        str =  "\n"
        for neuron in self._layer:
            str = str + neuron.__str__() + "\n"

        return str

    @abstractmethod
    def backward(self):
        raise NotImplementedError, "NeuronLayer class, backward() has to be implemented"

class OutputLayer(NeuronLayer):
    def __init__(self):
        raise NotImplementedError, "OutputLayer class, not implemeted"

class HiddenLayer(NeuronLayer):
    _isOutput = False

    def __init__(self, number_of_neurons, number_of_inputs, number_of_neurons_next_layer, isOutput = False):
        self._isOutput = isOutput


input_vector = np.array((1, 2, 3)).reshape(3, 1)

nl = NeuronLayer(number_of_neurons = 2, number_of_inputs = 3)
nl.forward(input_vector)
print("Neural layer + " + str(nl))


# HiddenLayer(number_of_neurons = 3, number_of_inputs = 2, number_of_neurons_next_layer = 1)

# # input vector
# x = ((1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
#
# # desired vector
# y = (1, 0, 0, 1)

# # first layer x -> w1
# w1_x = np.array(([1], [1], [1]))
# w2_x = np.array(([1], [1], [1]))
# print("==>" + str(w2_x))

# # second layer
# w_w12 = np.array(([1], [1], [1]))
#
# iteration = 0
# # Forward propagation x -> o
# # first layer computation
# v1_1 = np.array(x[iteration]).dot(w1_x)
# v2_1 = np.array(x[iteration]).dot(w2_x)
# print ("1: v1_1:%s v2_1:%s" % (str(v1_1), str(v2_1)))
#
# a = 1.7159
# b = 0.6666
# phi_v1_1 = a * np.tanh(b*v1_1)
# phi_v2_1 = a * np.tanh(b*v2_1)
# print ("1: phi_v1:%s phi_v2:%s" % (str(phi_v1_1), str(phi_v2_1)))
#
# y1_1 = np.array([1, phi_v1_1, phi_v2_1])
# print ("1: y1_1:%s" % str(y1_1))
#
# # second layer
# v1_2 = y1_1.dot(w_w12)
# print ("2: v1_2:%s" % str(v1_2))
#
# phi_v1_2 = a * np.tanh(b * v1_2)
# print ("2: phi_v1_2:%s" % str(phi_v1_2))
#
# # calculate error
# e = y[iteration] - phi_v1_2
# print ("e:%s" % str(e))
#
# # Back propagation x <- e
# delta = e * b/a * (a - phi_v1_2) * (a + phi_v1_2)
# print ("delta:%s" % str(delta))
#
# w_w3_2 =  w_w12 + 0.5 * delta * y1_1.reshape(3, 1)
# print ("w_w3_2:%s" % str(w_w3_2))
#
# delta_0_2 = b/a * (a - 1) * (a + 1) * delta * w_w12[0]
# delta_1_2 = b/a * (a - phi_v1_1) * (a + phi_v1_1) * delta * w_w12[1]
# delta_2_2 = b/a * (a - phi_v2_1) * (a + phi_v2_1) * delta * w_w12[2]
#
# print ("delta_0_2:%s delta_1_2:%s delta_2_2:%s" % (str(delta_0_2), str(delta_1_2), str(delta_2_2)))
#
# w_w1_1 =  w1_x + 0.5 * delta * x[iteration].reshape(3, 1)
# print ("w_w1_1:%s" % str(w_w1_1))
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

    def get_phi_derivative(self):
        return self._activation_function.get_phi_derivative(self._v)

    def __str__(self):
        return "Weights (w): %s\nLocal field(v): %s\nOutput(o): %s" % (str(self._w), str(self._v), str(self._y))

class NeuronLayer(object):
    _layer = [None]
    _layer_output = [None]
    _layer_output_len = 0

    _delta = None    # local gradient

    # number_of_neurons - number of neurons in the current layer
    # number_of_inputs - number of th input connection, for every connection we have it's own weight (w)
    def __init__(self, number_of_neurons, number_of_inputs):
        self._layer_output = [None] * number_of_neurons
        self._layer_output_len = number_of_neurons

        self._layer = [None] * number_of_neurons
        for i in range(number_of_neurons):
            self._layer[i] = Neuron(number_of_inputs)

    def forward(self, input_layer):
        for i in range(self._layer.__len__()):
            self._layer[i].calculate_local_field(input_layer)
            self._layer_output[i] = self._layer[i].get_output()

    def get_layer_output(self):
        return np.array((self._layer_output)).reshape(self._layer_output_len, 1)

    def __str__(self):
        str =  "\n"
        for neuron in self._layer:
            str = str + neuron.__str__() + "\n"

        return str

    @abstractmethod
    def backward(self):
        raise NotImplementedError, "NeuronLayer class, backward() has to be specific to layer type (output/hidden)"

class OutputLayer(NeuronLayer):
    _e = None        # output error
    _d = None        # desired output

    _phi_derivative_array = None    # Storage for of phi derivatives for the back propagation process

    def __init__(self, number_of_neurons, number_of_inputs):
        super(OutputLayer, self).__init__(number_of_neurons, number_of_inputs)

        self._phi_derivative_array = [None] * self._layer_output_len

    def backward(self, desired_output):
        self._d = desired_output
        self._e = self._d - self.get_layer_output()
        print("OutputLayer: " + str(self._e))

        for i in range(self._layer_output_len):
            self._phi_derivative_array[i] = self._layer[i].get_phi_derivative()

        print("phi_array: " + str(self._phi_derivative_array))

        self._delta = np.multiply(self._e, np.array((self._phi_derivative_array)).reshape(2, 1) )

        print("_delta: " + str(self._delta))


class HiddenLayer(NeuronLayer):
    def backward(self, local_gradient):
        pass


input_vector = np.array((1, 2)).reshape(2, 1)

nl_1 = HiddenLayer(number_of_neurons = 2, number_of_inputs = 2)
nl_1.forward(input_vector)
print("Neural layer 1:" + str(nl_1))

nl_2 = OutputLayer(number_of_neurons = 2, number_of_inputs = 2)
nl_2.forward(nl_1.get_layer_output())
print("Neural layer 2:" + str(nl_2))

desired = np.array((1, 2)).reshape(2, 1)
nl_2.backward(desired)
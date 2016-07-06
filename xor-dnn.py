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

    def get_phi_derivative(self, output_signal):
        return self._ab * (self._a - output_signal) * (self._a + output_signal)

class Neuron(object):
    _v = 0.0       # local field
    _y = 0.0       # output
    _w = None       # row vector of weights
    _w_minus_one = None
    _w_minus_two = None
    _num_of_inputs = 0  # num of the inputs
    _zero_input = np.array(([1]))
    _inputs = None

    # Haykin "Activation function" page 135
    _activation_function = None

    def __init__(self, num_of_inputs, activation_function = HyperbolicTangent):
        self._num_of_inputs = num_of_inputs + 1
        self._activation_function = activation_function(threshold = 0.0)

        # Haykin, initial weights page 148, default weight could be 1/sqrt(num_of_inputs)
        self._w = np.ones((1, self._num_of_inputs)) / np.sqrt(self._num_of_inputs)

        self._w_minus_one = np.zeros((1, self._num_of_inputs))
        self._w_minus_two = np.zeros((1, self._num_of_inputs))

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

    def get_phi_derivative(self):
        return self._activation_function.get_phi_derivative(output_signal = self._y)

    def __str__(self):
        return "Weights (w): %s\nInputs: %s\nLocal field(v): %s\nOutput(o): %s" %\
               (str(self._w), str(self._inputs), str(self._v), str(self._y))

class NeuronLayer(object):
    _layer = [None]
    _layer_output = None
    _layer_output_len = 0
    _previous_layer = None          # Previous layer

    _delta = None                   # local gradient
    _phi_derivative_array = None    # Storage for of phi derivatives for the back propagation process
    _learning_rate = 0.1            # TODO: learning rate should be adaptive

    # number_of_neurons - number of neurons in the current layer
    # number_of_inputs - number of th input connection, for every connection we have it's own weight (w)
    def __init__(self, number_of_neurons, number_of_inputs, previous_layer = None):
        self._layer_output_len = number_of_neurons
        self._layer_output = np.zeros((self._layer_output_len, 1))
        self._previous_layer = previous_layer

        self._layer = [None] * number_of_neurons
        for i in range(number_of_neurons):
            self._layer[i] = Neuron(number_of_inputs)

        # back propagation part
        self._phi_derivative_array = np.zeros((self._layer_output_len, 1))

    def forward(self, input_signals):
        for i in range(self._layer_output_len):
            self._layer[i].calculate_local_field(input_signals)
            self._layer_output[i] = self._layer[i].get_output()

    def get_neurons_number(self):
        return self._layer_output_len

    def get_layer_output(self):
        return self._layer_output

    def __str__(self):
        str =  "\n"
        for neuron in self._layer:
            str = str + neuron.__str__() + "\n"

        return str

    def backward(self):
        for i in range(self._layer_output_len):
            self._phi_derivative_array[i] = self._layer[i].get_phi_derivative()

        # print("phi_array: " + str(self._phi_derivative_array))

    def recalculate_weghts(self):
        # # update weights
        # # Haykin "Backward computation" page 141
        for i in range(self._layer_output_len):
            # print("delta %s val: %s" % (i, str(self._delta[i])) )
            neuron = self._layer[i]
            neuron._w_minus_one = np.copy(neuron._w)
            for k in range(1, neuron._num_of_inputs):
                # every neuron has _num_of_inputs - 1 inputs, because 0 input is 1
                neuron._w[0, k] = neuron._w_minus_one[0, k] + self._learning_rate * self._delta[i] * neuron._inputs[k]
                print("\t w(n)%s:%s input:%s" % (k, str(neuron._w_minus_one[0, k]), str(neuron._inputs[k])) )
                print("\t w(n+1)%s:%s" % (k, str(neuron._w[0, k])) )

class OutputLayer(NeuronLayer):
    _e = 0.0        # output error
    _d = 0.0        # desired output

    def __init__(self, number_of_neurons, number_of_inputs, previous_layer):
        super(OutputLayer, self).__init__(number_of_neurons, number_of_inputs, previous_layer)

    def forward(self):
        super(OutputLayer, self).forward(input_signals = self._previous_layer.get_layer_output())

    def backward(self, desired_output):
        super(OutputLayer, self).backward()

        self._d = desired_output
        self._e = self._d - self.get_layer_output()
        print("OutputLayer error: " + str(self._e))

        # Haykin "Backward computation" page 141
        self._delta = np.multiply(self._e, self._phi_derivative_array)
        print("OutputLayer _delta: " + str(self._delta))

        self.recalculate_weghts()

class InputLayer(NeuronLayer):
    def __init__(self, number_of_neurons, number_of_inputs):
        super(InputLayer, self).__init__(number_of_neurons, number_of_inputs, None)

    def forward(self, input_signals):
        super(InputLayer, self).forward(input_signals = input_signals)

    def backward(self, layer_minus_one):
        super(InputLayer, self).backward()

        self._delta = np.zeros((self._layer_output_len, 1))

        # # update weights
        # # Haykin "Backward computation" page 141
        for i in range(self.get_neurons_number()):
            # neuron_l = self._layer[i]
            for k in range(layer_minus_one.get_neurons_number()):
                neuron_l_minus_one = layer_minus_one._layer[k]
                self._delta[i, 0] = self._delta[i, 0] + neuron_l_minus_one._w_minus_one[0, k] * layer_minus_one._delta[k - 1]
                print("InputLayer w%s:%s delta(l+1):%s delta(l)" %
                      (i, str(neuron_l_minus_one._w_minus_one[0, k]), layer_minus_one._delta[k - 1]), str(self._delta[i, 0]) )

        print("Inputlayer: delta(l)" + str(self._delta))

        self.recalculate_weghts()


# neuron test
# n = Neuron(2)
# n.calculate_local_field(input_vector)
# print("neuron:\n" + str(n))

nl_1 = InputLayer(number_of_neurons = 2, number_of_inputs = 2)
nl_2 = OutputLayer(number_of_neurons = 1, number_of_inputs = 2, previous_layer = nl_1)

input_vector = np.array((1, 1)).reshape(2, 1)
nl_1.forward(input_vector)
nl_2.forward()
print("Neural layer 2:" + str(nl_2))

# desired = np.array((1, 2)).reshape(2, 1)
# desired = np.array((1.60065595, 1.60065595)).reshape(2, 1)

desired = np.array((1.54257532))

nl_2.backward(desired)
nl_1.backward(layer_minus_one = nl_2)
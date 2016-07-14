#!/usr/bin/env python

import numpy as np
from lib.transferfunction import HardLim
from lib.neuron import NeuronLayer

layer = NeuronLayer(
    num_of_neurons = 1,
    num_of_inputs = 1,
    hasBias=False,
    activation_function = HardLim)

layer.calculate_local_field(np.array(([1, -1, -1])).reshape(3, 1))
print("Layer: " + str(layer))
layer.trainining(np.array(([0, 1])).reshape(2, 1))
print("Layer: " + str(layer))

# layer.calculate_local_field(np.array(([1, 1, -1])))
# print("Layer: " + str(layer))
# layer.trainining(np.array(([1])))
# print("Layer: " + str(layer))

# layer.calculate_local_field(np.array(([1, -1, -1])))
# print("Layer: " + str(layer))
# layer.trainining(np.array(([0])))
# print("Layer: " + str(layer))
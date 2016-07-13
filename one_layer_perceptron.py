#!/usr/bin/env python

import numpy as np
from lib.transferfunction import HardLim
from lib.neuron import NeuronLayer

layer = NeuronLayer(
    num_of_neurons = 2,
    num_of_inputs = 2,
    activation_function = HardLim)

input_vector = np.array((0, 0)).reshape(2, 1)

layer.calculate_local_field(input_vector)

print("Layer: " + str(layer))

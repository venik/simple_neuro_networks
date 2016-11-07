# Library with transfer functions
# GPLv2 License
# (c) Sasha Nikiforov

import numpy as np
from abc import abstractmethod


class ActivationFunction(object):
    _threshold = 0

    def __init__(self, threshold = 0):
        self._threshold = threshold

    def get_threshold(self):
        return self._threshold

    @abstractmethod
    def get_phi(self, number_of_neurons, number_of_neurons_next_layer):
        raise NotImplementedError("ActivationFunction class, get_phi() has to be implemented")

    @abstractmethod
    def get_phi_derivative(self, local_field):
        raise NotImplementedError("ActivationFunction class, get_phi_derivative() has to be implemented")


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


class HardLim(ActivationFunction):

    def __init__(self):
        super(HardLim, self).__init__()

    def get_phi(self, local_field):
        # True is 1 and False is 0, so need to multiply to get float value
        return (local_field >= 0) * 1

    def get_phi_derivative(self, local_field):
        raise NotImplementedError("HardLim.get_phi_derivative() hardlim doesnt have derivative, " \
                                  "so can not be used for example with backpropagation")
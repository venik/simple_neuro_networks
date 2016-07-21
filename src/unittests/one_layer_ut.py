#!/usr/bin/env python

# 1 layer perceptron neuronetwork against MNIST UnitTests
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import numpy as np
import unittest

import sys
sys.path.append('../../')
from lib import HardLim
from lib.neuron import NeuronLayer

class TestOneLayerPerceptron(unittest.TestCase):

    def setUp(self):
        pass

    def getLayerWithPositiveOutput(self, hasBias):
        weights = np.array((1, 2)).reshape(1, 2)
        biases = np.array((3)).reshape(1, 1)

        layer = NeuronLayer(
            num_of_neurons = 1,
            num_of_inputs = 2,
            hasBias = hasBias,
            activation_function = HardLim,
            weights = weights,
            biases = biases)

        input = np.array((4, 5)).reshape(2, 1)
        layer.calculate_local_field(input)

        # with bias
        # 1*4 + 2*5 + 3 = 17
        result_with_bias = 17

        # without bias
        # 1*4 + 2*5 = 14
        result_without_bias = 14

        output = 1

        return (layer, input, weights, biases, result_with_bias, result_without_bias, output)

    def getLayerWithNegativeOutput(self, hasBias):
        weights = np.array((1, 2)).reshape(1, 2)
        biases = np.array((3)).reshape(1, 1)

        layer = NeuronLayer(
            num_of_neurons = 1,
            num_of_inputs = 2,
            hasBias = hasBias,
            activation_function = HardLim,
            weights = weights,
            biases = biases)

        input = np.array((-4, -5)).reshape(2, 1)
        layer.calculate_local_field(input)

        # with bias
        # 1*(-4) + 2*(-5) + 3 = -11
        result_with_bias = -11

        # without bias
        # 1*(-4) + 2*(-5) = -14
        result_without_bias = -14

        output = 0

        return (layer, input, weights, biases, result_with_bias, result_without_bias, output)

    def testPositiveOutputWithBias(self):
        (layer, _, _, _, result_with_bias, _, output) = self.getLayerWithPositiveOutput(hasBias = True)

        self.assertEqual(layer._v, result_with_bias)
        self.assertEqual(layer.get_output(), output)

    def testPositiveOutputWithoutBias(self):
        (layer, _, _, _, _, result_without_bias, output) = self.getLayerWithPositiveOutput(hasBias = False)

        self.assertEqual(layer._v, result_without_bias)
        self.assertEqual(layer.get_output(), output)

    def testPositiveTrainingNoChangesWithBias(self):
        (layer, _, weights, biases, _, _, output) = self.getLayerWithPositiveOutput(hasBias = True)

        layer.trainining(target = output)

        # output 1 desired 1 - no changes in weights
        self.assertTrue(np.equal(layer.get_weights(), weights).all())
        self.assertTrue(np.equal(layer.get_biases(), biases).all())

    def testPositiveTrainingWithChangesWithBias(self):
        (layer, input, weights, biases, _, _, output) = self.getLayerWithPositiveOutput(hasBias = True)

        layer.trainining(output - 1)

        # output 1 desired 0 - substrata transpose input from weights
        self.assertTrue(np.equal(layer.get_weights(), weights - input.reshape(1, 2)).all())
        self.assertTrue(np.equal(layer.get_biases(), biases - 1).all())

    def testNegativeTrainingWithChangesWithBias(self):
        (layer, input, weights, biases, _, _, output) = self.getLayerWithNegativeOutput(hasBias = True)

        layer.trainining(output + 1)

        # output 1 desired 0 - substrata transpose input from weights
        self.assertTrue(np.equal(layer.get_weights(), weights + input.reshape(1, 2)).all())
        self.assertTrue(np.equal(layer.get_biases(), biases + 1).all())

if __name__ == '__main__':
    unittest.main()
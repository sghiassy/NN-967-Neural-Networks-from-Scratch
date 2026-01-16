# import numpy as np
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        print("Initializing Layer_Dense")
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        print(self.weights)
        self.biases = np.zeros((1, n_neurons))
        print(self.biases)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Generate Sample Data
Feature, categories = spiral_data(samples=100, classes=3)
print("Feature: ", Feature[:5], np.shape(Feature))
print("categories: ", categories)

print("")

# Create 2x3 Hidden Neural Network
dense1 = Layer_Dense(2, 3)
dense1.forward(Feature)
activation1 = Activation_ReLU()
activation1.forward(dense1.output)


# Create 3x3 Output Layer
dense2 = Layer_Dense(3, 3)
dense2.forward(activation1.output)
activation2 = Activation_Softmax()
activation2.forward(dense2.output)

print(activation2.output[:5])

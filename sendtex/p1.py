import numpy as np

inputs = [1.2, 5.1, 2.1]
inputs2 = [3.2, 3.1, 2.3]
weights = [3.1, 2.1, 8.7]
bias = 3

def calculate_result_of_neuron(inputs, weights, bias):
    assert len(inputs) == len(weights)
    total = 0

    for i in range(0, len(inputs)):
        total += (weights[i] * inputs[i])
    return total + bias

# single linear neuron, also implementation for linear regression
print(calculate_result_of_neuron(inputs2, weights, bias))
print(np.dot(inputs2, weights)+bias)
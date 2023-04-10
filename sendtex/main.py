import numpy as np
import sys
import matplotlib

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.uniform(size=n_inputs)
        self.b = np.random.uniform()
        

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    def generate_weight_matrix(self):
        matrix = [neuron.weights for neuron in self.neurons]
        return np.array(matrix)
    
    # def generate_bias_matrix(self, num_inputs=1):
    #     biases = [neuron.b for neuron in self.neurons]
    #     biases_matrix = [biases for _ in range(num_inputs)]
    #     biases_matrix = np.array(biases_matrix)
    #     biases_matrix = biases_matrix.T
        
    #     return biases_matrix
    
    def generate_biases_vector(self):
        return np.array([neuron.b for neuron in self.neurons])
    
    
    def __call__(self, inputs):
        inputs = np.array(inputs)
        function = self.generate_weight_matrix()
        result = (inputs @ function.T) + self.generate_biases_vector()
        return result
        
    
    
class MultiLayerPerceptron:
    def __init__(self, num_inputs):
        self.layers = []
        self.num_inputs = num_inputs
        
        
    def add_layer(self, num_outputs):
        layer  = Layer(self.num_inputs, num_outputs)
        self.layers.append(layer)
        self.num_inputs = num_outputs
    
    
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
        
        
mlp = MultiLayerPerceptron(3)
mlp.add_layer(4)

inputs = [[1.2, 5.1, 2.1],[3.2, 3.1, 2.3]]
print(mlp(inputs))
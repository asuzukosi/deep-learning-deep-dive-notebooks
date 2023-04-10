import numpy as np
# building a neural network library

# implementation design

# base layer class

# create dense layer

# create activation layer

# implementation of activation and loss function

class Base:
    pass

class Layer(Base):
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def backward(self, output_gradient, learning_rate):
        pass
    
    def __call__(self, input):
        pass
    
class DenseLayer(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_outputs, n_inputs)
        self.bias = np.random.randn(n_outputs)
        
    def __call__(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.bias
    
    def backward(self, output_gradients, learning_rate):
        weight_gradients = np.dot(output_gradients, self.inputs.T)
        bias_gradients = output_gradients
        input_gradients = np.dot(weight_gradients.T, output_gradients)
        self.weights  -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradients
        
        return input_gradients
    
    

class Activation(Layer):
    def __init__(self, activtion, activtion_prime):
        self.activation = activtion
        self.activation_prime = activtion_prime
            
    def __call__(self, inputs):
        self.inputs = inputs
        return self.activation(self.inputs)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.inputs))
    
    

class HyperbolicTangentActivation(Activation):
    def __init__(self):
        def hyperbolic_tangent(x):
            return np.tanh(x)

        def hyperbolic_prime(x):
            return 1 - (np.tanh(x)**2)
    
        super().__init__(hyperbolic_tangent, hyperbolic_prime)
        
        
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class Loss(Base):
    def __init__(self, function):
        self.function = function
        
    def __call__(self, y_actual, y_pred):
        self.function(y_actual, y_pred)

class MeanSquareError(Loss):
    pass

class Trainer(Base):
    pass

# solviing the xor problem, the reason we use the xor funciton
# for testing is because the problme rqquires nnon linarity to se solved
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0, 1, 1, 0])


network = [
    DenseLayer(2, 3),
    HyperbolicTangentActivation(),
    DenseLayer(3,1),
    HyperbolicTangentActivation(),
]

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    error = 0
    
    for x, y_true in zip(X, y):
        output = x
        for layer in network:
            output = layer(output)
        
        error += mse(y_true, output)
        
        grad = mse_prime(y_true, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
        
    error /= len(X)
    print("%d/%d, error: %f" % (epoch+ 1, epochs, error))
        
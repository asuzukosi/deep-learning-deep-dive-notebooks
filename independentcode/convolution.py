# implementation of a convolutional neural network
import numpy as np
from main import Layer, Activation
from scipy import signal

# quick introductiion of convolution
# convolutional layer
# reshape layer
# binary cross entropy
# sigmoid activation

# size of the output convolution is 
def size_of_output_of_kernel_valid_convolution(input_size, output_size):
    return (input_size - output_size) + 1

def size_of_output_of_kernel_full_convolution(input_size, output_size):
    return (input_size + output_size) - 1

input_size = np.array([3, 3])
kernel_size = np.array([2, 2])

print(size_of_output_of_kernel_valid_convolution(input_size, kernel_size))

# valid and full cross correlation
class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, channels):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.channels = channels
        self.input_shape = input_shape
        self.outputs_shape = (channels, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernel_shape = (channels, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.outputs_shape)
        
    def __call__(self, input):
        self.output = np.copy(self.biases)
        self.input = input
        for channel in range(self.channels):
            for depth in range(self.input_depth):
                self.output[channel] += signal.correlate2d(self.input[depth], self.kernels[channel, depth], "valid")
        #         total += np.multiply(input[depth], self.kernels[channel][depth])
        #     channels.append(total)
        # channels = np.array(channels)
        # self.output += channels
        # return self.output
        return self.output
    
    
    def backward(self, output_gradient, learning_rate):
        kernel_gradients = np.zeros(self.kernel_shape)
        input_gradients = np.zeros(self.input_shape)
        
        for i in range(self.channels):
            for j in range(self.input_depth):
                kernel_gradients[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradients[j] = signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernel_gradients
        self.biases -= learning_rate * output_gradient            
                
        return input_gradients
    
    
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def __call__(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(self.output_shape, self.input_shape)
    
    
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true)/(1 - y_pred) - y_true/y_pred) / np.size(y_true)
        

class SigmoidActivation(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
            
        super(SigmoidActivation, self).__init__(sigmoid, sigmoid_prime)
        
        
class SoftmaxActivation(Activation):
    def __init__(self):
        def softmax(x):
            return np.exp(x) / sum(np.exp(x))
        
        super(SoftmaxActivation, self).__init__(softmax, None)
        
        
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.outputs)
        tmp = np.tile(self.outputs, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
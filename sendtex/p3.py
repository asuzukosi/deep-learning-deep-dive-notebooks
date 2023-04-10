# arrays or tensors??
import numpy as np

X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0],[-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

def stepFunction(x):
    # implementation of the step activation function
    return 1 if x > 0 else 0

def sigmoidFunction(x):
    # implementation of the sigmoid activation function
    return 1/(1 - np.exp(-x))


def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    
    return X, y


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)
        
    def __call__(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases


class Activation_ReLU:
    def __call__(self, inputs):
        return np.maximum(0, inputs)
    
    
    
class Activation_Softmax:
    
    def __call__(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values/ np.sum(exp_values, axis=1, keepdims=True)
    
    
    
    
class Loss:
    def forward(self, logits, targets):
        raise NotImplementedError
    
    def __call__(self, targets, logits):
        losses = self.forward(logits, targets)
        batch_loss = np.mean(losses)
        return batch_loss
    
 
class CategoricalCrossEntropyLoss(Loss):
    def forward(self, logits, targets):
        samples = len(logits)
        y_pred_clipped = np.clip(logits, 1e-7, 1-1e-7)
        if len(targets.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), logits]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*targets, axis=1)
        return -np.exp(correct_confidences)
        
        
    def __call__(self, targets, logits):
        return self.calculate(targets, logits)
    
    
    
    
    
X, y = create_data(100, 3)

layer = Layer_Dense(2, 3)
# print(layer.weights)
relu_activation = Activation_ReLU()
layer2= Layer_Dense(3, 3)
softmax_activation = Activation_Softmax()

# layer2 = Layer_Dense(5, 3)
print(softmax_activation(layer2(relu_activation(layer(X)))))
# print(layer2(layer(X)))
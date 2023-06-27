"""
Defining the hyperparameter used in the training the model
"""


# parameters
learning_rate = 0.01
number_of_steps = 100
batch_size = 128
display_step = 1



# network parameters
n_hidden_1 = 300
n_hidden_2 = 200
n_inputs = 784
n_outputs = 10

# training parameters
checkpoints_every = 100
checkpoints_dir = '/runs/'

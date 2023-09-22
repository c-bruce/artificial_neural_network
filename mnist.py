import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from network import Network

def calculate_accuracy(network, x, y):
    # Calculate network accuracy
    correct = 0
    for i in range(0, len(x)):
        network.feedforward(x[i].flatten() / 255.0)
        if np.where(network.activations[-1] == max(network.activations[-1]))[0][0] == y[i]:
            correct += 1
    print(f"Accuracy: {correct / len(x)}")

# Prepare training data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Define n_epochs and set up batches
n_epochs = 5
n_batches = 600
batches = []
input_layer = np.array_split(train_X, n_batches)
expected_output = np.array_split(np.eye(10)[train_y], n_batches)
for i in range(0, n_batches):
    batch = []
    for j in range(0, len(input_layer[i])):
        batch.append({'input_layer' : input_layer[i][j].flatten() / 255.0, 'expected_output' : expected_output[i][j]})
    batches.append(batch)

# Setup and train network
network = Network([784,32,32,10], 0.1)
network.train_network(n_epochs, batches)

# Calculate accuracy of the network
calculate_accuracy(network, test_X, test_y)

plt.plot(network.costs)
plt.show()
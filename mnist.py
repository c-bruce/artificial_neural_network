import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from network import Network

# Prepare training data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Note:
# - To go from train_t/test_y as an int to an array we can do np.eye(10)[train_t[i]]
# - We can flattern train_X/test_X using train_X[i].flatten()

def calculate_accuracy(network, x, y):
    correct = 0
    for i in range(0, len(x)):
        network.feedforward(x[i].flatten() / 255.0)
        if np.where(network.activations[-1] == max(network.activations[-1]))[0][0] == y[i]:
            correct += 1
    print(f"Accuracy: {correct / len(x)}")

n_batches = 600
batches = []
input_layer = np.array_split(train_X, n_batches)
expected_output = np.array_split(np.eye(10)[train_y], n_batches)
for i in range(0, n_batches):
    batch = []
    for j in range(0, len(input_layer[i])):
        batch.append({'input_layer' : input_layer[i][j].flatten() / 255.0, 'expected_output' : expected_output[i][j]})
    batches.append(batch)

# Setup network
network = Network([784,32,10], 0.1)
network.train_network(10, batches)

plt.plot(network.costs)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from matplotlib import rc, rcParams

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

# Load the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Define the number of rows (m) and columns (n) for the grid
m = 6  # Number of rows
n = 15  # Number of columns

# Randomly select m x n samples and their corresponding labels
random_indices = np.random.randint(0, len(x_train), size=m * n)
random_samples = x_train[random_indices]
random_labels = y_train[random_indices]

# Create a grid for plotting
fig, axes = plt.subplots(m, n, figsize=(10, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot the selected samples on the grid with their labels as titles
for i, ax in enumerate(axes.flat):
    sample = random_samples[i]
    label = random_labels[i]
    ax.imshow(sample, cmap='gray')
    # ax.set_title(f'{label}:')
    ax.axis('off')

# Display the grid of samples
plt.show()

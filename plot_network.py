import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from network import Network

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

def plot_neural_network(layer_sizes, display_nodes, weights, activations, background_color='white', node_color='black', positive_weight_color='red', negative_weight_color='blue', figsize=(10, 4), node_size=200, plot_lables=False):
    num_layers = len(layer_sizes)
    vertical_spacing = 1.0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the background color of the plot
    ax.set_facecolor(background_color)
    
    # List of lists to store y-coordinates of nodes for each layer
    y_coords_layers = []
    
    # Draw nodes and calculate y-coordinates
    for i in range(num_layers):
        num_nodes = layer_sizes[i]
        y_positions_center_adjusted = np.linspace(0.5 + vertical_spacing * (display_nodes[i] - 1) / 2, 0.5 - vertical_spacing * (display_nodes[i] - 1) / 2, display_nodes[i] + 2)
        y_positions = np.linspace(0.5 + vertical_spacing * (num_nodes - 1) / 2, 0.5 - vertical_spacing * (num_nodes - 1) / 2, num_nodes)
        
        if display_nodes[i] != layer_sizes[i]:
            y_positions[:display_nodes[i]] = y_positions_center_adjusted[:display_nodes[i]]
            y_positions[-1] = y_positions_center_adjusted[-1]
        
        y_coords_layers.append(y_positions.tolist())  # Store y-coordinates for each layer
        
        for j, y in enumerate(y_positions):
            if j < display_nodes[i] or j == num_nodes - 1:
                x = i
                activation = activations[i][j]
                ax.scatter(x, y, color=background_color, edgecolor=node_color, linewidth=1, s=node_size, zorder=1)  # Node appearance updated
                ax.scatter(x, y, color=node_color, edgecolor=node_color, alpha=abs(activation), linewidth=1, s=node_size, zorder=1)  # Node appearance updated
                # if plot_lables: ax.text(x-0.02, y-0.1, f"$a_{j}^{{({i})}}$")
                if plot_lables: ax.text(x-0.04, y-0.05, f"$a_{j}^{{({i})}}$")
    
    # Draw connections using the calculated y-coordinates
    for i in range(num_layers - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                if j < display_nodes[i] or j == layer_sizes[i] - 1:
                    if k < display_nodes[i+1] or k == layer_sizes[i+1] - 1:
                        x = i
                        next_x = i + 1
                        y = y_coords_layers[i][j]  # Use stored y-coordinates
                        next_y = y_coords_layers[i+1][k]  # Use stored y-coordinates
                        weight = weights[i][j, k]
                        line_width = abs(weight) * 2  # Scale line width based on weight
                        line_color = positive_weight_color if weight >= 0 else negative_weight_color
                        ax.plot([x, next_x], [y, next_y], color=line_color, linewidth=line_width, zorder=0)
    
    y_min = y_coords_layers[0][-1] - vertical_spacing
    y_max = y_coords_layers[0][0] + vertical_spacing
    ax.set_ylim(y_min, y_max)  # Adjust y-axis limits to fit all nodes and connections
    
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f'Layer {i+1}' for i in range(num_layers)])
    ax.set_title('Neural Network Structure')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Nodes')

    return fig, ax

# Cover image
# layer_sizes = [10, 6, 6, 6, 2]
# display_nodes = [10, 6, 6, 6, 2]
# weights = [np.random.uniform(-0.5, 0.5, size=(layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
# activations = [np.random.uniform(0, 1, size=layer_size) for layer_size in layer_sizes]
# plot_neural_network(layer_sizes, display_nodes, weights, activations, background_color='#1F1F1F', node_color='white', positive_weight_color='springgreen', negative_weight_color='springgreen')
# plt.tight_layout()
# plt.savefig('network.png', dpi=600)
# plt.show()

# Neural network structure
# layer_sizes = [5, 3, 3, 2]
# display_nodes = [5, 3, 3, 2]
# network = Network(layer_sizes, 0.1)
# network.feedforward(np.random.uniform(0,1,(5)))
# weights = [array.T*0.5 for array in network.weights]
# activations = network.activations
# fig, ax = plot_neural_network(layer_sizes, display_nodes, weights, activations, positive_weight_color='red', negative_weight_color='blue')
# ax.set_xticklabels(['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer'])
# plt.savefig('network.png', dpi=600)
# plt.show()

# Calculating a_0^(1)
# layer_sizes = [4,3]
# display_nodes = [4,3]
# weights = [np.zeros([4,3])]
# weights[0][:,0] = 0.5
# activations = [np.zeros(4), np.zeros(3)]
# plot_neural_network(layer_sizes, display_nodes, weights, activations, positive_weight_color='k', negative_weight_color='k', node_size=600, plot_lables=True)
# plt.text(1.2,2.0,'$w_{0,0}$')
# plt.text(1.2,1.5,'$w_{0,1}$')
# plt.text(1.2,1.0,'$w_{0,2}$')
# plt.text(1.2,0.5,'$w_{0,3}$')
# plt.tight_layout()
# plt.savefig('network.png', dpi=600)
# plt.show()

# Single neuron layers
layer_sizes = [1, 1, 1]
display_nodes = [1, 1, 1]
weights = [np.zeros([1,1]) + 0.5, np.zeros([1,1]) + 0.5]
activations = [np.zeros(1), np.zeros(1), np.zeros(1)]
fig, ax = plot_neural_network(layer_sizes, display_nodes, weights, activations, positive_weight_color='k', negative_weight_color='k', node_size=1200)
plt.text(2.6,0.0,'$w^{(L)}$')
plt.text(2.6,0.2,'$w^{(L-1)}$')
plt.text(1.96,1,'$b^{(L)}$')
plt.text(0.925,1,'$b^{(L-1)}$')
plt.text(-0.075,1,'$b^{(L-2)}$')
#plt.text(2.6,0.8,'$a^{(L)}=\sigma(w^{(L)}a^{(L-1)}+b^{(L)})$')
plt.text(1.96,0.475,'$a^{(L)}$')
plt.text(0.925,0.475,'$a^{(L-1)}$')
plt.text(-0.075,0.475,'$a^{(L-2)}$')
plt.text(-0.25,0.475,'$...$')
plt.savefig('network.png', dpi=600)
plt.show()
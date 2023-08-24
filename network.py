import numpy as np
import math

class Network:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activations = self.__init_activations_zero()
        self.weights = self.__init_weights_random()
        self.biases = self.__init_biases_zero()
        self.G_weights = self.__init_G_weights()
        self.G_biases = self.__init_G_biases()
        self.dcost_dweights = self.__init_weights_zero()
        self.dcost_dbiases = self.__init_biases_zero()
        self.cost = 0
        self.costs = []
    
    def __init_activations_zero(self):
        activations = []
        for layer in self.layers:
            activations.append(np.zeros(layer))
        return activations
    
    def __init_weights_random(self):
        weights = []
        for i in range(0, len(self.layers) - 1):
            weights.append(np.random.uniform(-1, 1, (self.layers[i+1], self.layers[i])))
        return weights
    
    def __init_weights_zero(self):
        weights = []
        for i in range(0, len(self.layers) - 1):
            weights.append(np.zeros((self.layers[i+1], self.layers[i])))
        return weights
    
    def __init_biases_zero(self):
        biases = []
        for i in range(1, len(self.layers)):
            biases.append(np.zeros(self.layers[i]))
        return biases
    
    def __init_G_weights(self):
        G_weights = []
        for i in range(0, len(self.layers) - 1):
            G_weights.append(np.zeros([len(self.weights[i]), len(self.weights[i][0]), len(self.weights[i][0])]))
        return G_weights
    
    def __init_G_biases(self):
        G_biases = []
        for i in range(0, len(self.layers) - 1):
            G_biases.append(np.zeros([len(self.biases[i]), len(self.biases[i])]))
        return G_biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def calculate_cost(self, expected_output):
        cost = sum((self.activations[-1] - expected_output)**2) # Mean square error
        return cost
    
    def feedforward(self, input_layer):
        self.activations[0] = input_layer
        for i in range(0, len(self.layers) - 1):
            self.activations[i+1] = self.sigmoid(np.dot(self.weights[i], self.activations[i]) + self.biases[i])
    
    def backpropagation(self, expected_output):
        # 1) Calculate dcost_dweights and dcost_dbiases for each training example in a batch
        # 2) Add them together and average them
        # 3) Output the gradients for the gradient descent algorithm to nudge them

        # Calculate dcost_dactivations for the output layer
        dcost_dactivations = 2 * (self.activations[-1] - expected_output)

        # Loop backward through the layers to calculate dcost_dweights and dcost_dbiases
        for i in range(-1, -len(self.layers), -1):
            dactivations_dz = self.dsigmoid(np.dot(self.weights[i], self.activations[i-1]) + self.biases[i]) # Sigmoid output layer

            dz_dweights = self.activations[i-1]
            dz_dbiases = 1

            self.dcost_dweights[i] += dz_dweights[np.newaxis,:] * (dactivations_dz * dcost_dactivations)[:,np.newaxis]
            self.dcost_dbiases[i] += dz_dbiases * dactivations_dz * dcost_dactivations

            # Calculate dcost_dactivations for hidden layer
            dz_dactivations = self.weights[i]
            dcost_dactivations = sum(dz_dactivations * (dactivations_dz * dcost_dactivations)[:,np.newaxis])
    
    def average_gradients(self, n):
        # Calculate the average gradients for a batch containing n samples
        for i in range(0, len(self.layers) - 1):
            self.dcost_dweights[i] = self.dcost_dweights[i] / n
            self.dcost_dbiases[i] = self.dcost_dbiases[i] / n
    
    def reset_gradients(self):
        # Reset gradients before starting a new batch
        self.dcost_dweights = self.__init_weights_zero()
        self.dcost_dbiases = self.__init_biases_zero()
    
    def reset_cost(self):
        self.cost = 0
    
    def update_G(self):
        for i in range(0, len(self.layers) - 1):
            self.G_biases[i] += np.outer(self.dcost_dbiases[i], self.dcost_dbiases[i].T)
            for j in range(0, len(self.weights[i])):
                self.G_weights[i][j] += np.outer(self.dcost_dweights[i][j], self.dcost_dweights[i][j].T)
    
    def update_weights_and_biases(self):
        # Perform gradient descent step to update weights and biases
        # Vanilla Gradient Descent
        # for i in range(0, len(self.layers) - 1):
        #     self.weights[i] -= (self.learning_rate * self.dcost_dweights[i])
        #     self.biases[i] -= (self.learning_rate * self.dcost_dbiases[i])
        
        # AdaGrad Gradient Desecent
        self.update_G()
        for i in range(0, len(self.layers) - 1):
            self.biases[i] -= (self.learning_rate * (np.diag(self.G_biases[i]) + 0.00000001)**(-0.5)) * self.dcost_dbiases[i]
            for j in range(0, len(self.weights[i])):
                self.weights[i][j] -= (self.learning_rate * (np.diag(self.G_weights[i][j]) + 0.00000001)**(-0.5)) * self.dcost_dweights[i][j]
    
    def process_batch(self, batch):
        for sample in batch:
            self.feedforward(sample['input_layer'])
            self.backpropagation(sample['expected_output'])
            self.cost += self.calculate_cost(sample['expected_output'])
    
    def train_network(self, epochs, batches):
        for epoch in range(0, epochs):
            print(f"Epoch: {epoch}\n")
            for batch in batches:
                self.process_batch(batch)
                self.costs.append(self.cost / len(batch))
                if math.isnan(self.costs[-1]):
                    break
                self.reset_cost()
                self.average_gradients(len(batch))
                self.update_weights_and_biases()
                self.reset_gradients()
                print(f"Cost: {self.costs[-1]}")

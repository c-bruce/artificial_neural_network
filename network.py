import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.activations = self.__init_activations()
        self.weights = self.__init_weights()
        self.biases = self.__init_biases()
        self.dcost_dweights = self.__init_dcost_dweights()
        self.dcost_dbiases = self.__init_dcost_dbiases()
    
    def __init_activations(self):
        activations = []
        for layer in self.layers:
            activations.append(np.zeros(layer))
        return activations
    
    def __init_weights(self):
        weights = []
        for i in range(0, len(self.layers) - 1):
            weights.append(np.random.uniform(-1, 1, (self.layers[i+1], self.layers[i])))
        return weights
    
    def __init_biases(self):
        biases = []
        for i in range(1, len(self.layers)):
            biases.append(np.zeros(self.layers[i]))
        return biases
    
    def __init_dcost_dweights(self):
        dcost_dweights = []
        for i in range(0, len(self.layers) - 1):
            dcost_dweights.append(np.zeros((self.layers[i+1], self.layers[i])))
        return dcost_dweights
    
    def __init_dcost_dbiases(self):
        dcost_dbiases = []
        for i in range(1, len(self.layers)):
            dcost_dbiases.append(np.zeros(self.layers[i]))
        return dcost_dbiases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return np.exp(x) / (np.exp(x) + 1)
    
    def feedforward(self, input_layer):
        self.activations[0] = input_layer
        for i in range(0, len(self.layers) - 1):
            self.activations[i+1] = self.sigmoid(np.dot(self.weights[i], self.activations[i]) + self.biases[i])
    
    def cost(self, expected):
        return sum((self.activations[-1] - expected)**2)
    
    def backpropagation(self, expected):
        # 1) Calculate dcost_dweights and dcost_dbiases for each training example in a batch
        # 2) Add them together and average them
        # 3) Output the gradients for the gradient descent algorithm to nudge them

        # Calculate dcost_dactivations for the output layer
        dcost_dactivations = 2 * (self.activations[-1] - expected)

        # Loop backward through the layers to calculate dcost_dweights and dcost_dbiases
        for i in range(-1, -len(self.layers), -1):
            dactivations_dz = self.dsigmoid(np.dot(self.weights[i], self.activations[i-1]) + self.biases[i])
            dz_dweights = self.activations[i-1]
            dz_dbiases = 1

            self.dcost_dweights[i] += dz_dweights[np.newaxis,:] * (dactivations_dz * dcost_dactivations)[:,np.newaxis]
            self.dcost_dbiases[i] += dz_dbiases * dactivations_dz * dcost_dactivations

            # Calculate dcost_dactivations for hidden layer
            dz_dactivations = self.weights[i]
            dcost_dactivations = sum(dz_dactivations * (dactivations_dz * dcost_dactivations)[:,np.newaxis])
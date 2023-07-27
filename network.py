import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.neurons = self.__init_neurons()
        self.weights = self.__init_weights()
        self.biases = self.__init_biases()
    
    def __init_neurons(self):
        neurons = []
        for layer in self.layers:
            neurons.append(np.zeros(layer))
        return neurons
    
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
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def feedforward(self, input_layer):
        self.neurons[0] = input_layer
        for i in range(0, len(self.layers) - 1):
            self.neurons[i+1] = self.sigmoid(np.dot(self.weights[i], self.neurons[i]) + self.biases[i])
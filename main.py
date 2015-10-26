__author__ = 'John Bucknam & Bill Clark'

import numpy as np
# from scipy import optimize

class NeuralNetwork(object):
    def __init__(self, layerSizes):
        self.inputLayerSize = layerSizes[0]
        self.outputLayerSize = layerSizes[len(layerSizes) - 1]
        self.hiddenLayerSizes = layerSizes[1:len(layerSizes) - 1]
        self.W = []

        self.W.append(np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[0]))
        for i in range(len(self.hiddenLayerSizes) - 1):
            self.W.append(np.random.randn(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i+1]))
        self.W.append(np.random.randn(self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize))

        # self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        # self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, input_matrix):
        self.z = []
        self.a = []
        self.z.append(np.dot(input_matrix, self.W[0]))
        for i in range(len(self.W) - 1):
            self.a.append(self.sigmoid(self.z[i]))
            self.z.append(np.dot(self.a[i], self.W[i+1]))
        y_hat = self.sigmoid(self.z[len(self.z) - 1])
        return y_hat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def getParams(self):
        return np.concatenate(self.W[:].ravel())


class Trainer(object):
    def __init__(self, N):
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.Y))

    def train(self, X, y):
        self.X = X
        self.y = y

        # Costs
        self.J = []

        # Parameters of weights from the Neural Network
        params = self.N.getParams()


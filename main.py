import numpy as np
# from scipy import optimize

__author__ = 'John Bucknam & Bill Clark'

X = np.matrix([1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1])
y = np.matrix([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        self.inputLayerSize = layer_sizes[0]
        self.outputLayerSize = layer_sizes[len(layer_sizes) - 1]
        self.hiddenLayerSizes = layer_sizes[1:len(layer_sizes) - 1]

        self.W = []
        if len(self.hiddenLayerSizes) != 0:
            self.W.append(np.matrix(np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[0])))
            for i in range(len(self.hiddenLayerSizes) - 1):
                self.W.append(np.matrix(np.random.randn(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i+1])))
            self.W.append(np.matrix(np.random.randn(self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize)))
        else:
            self.W.append(np.matrix(np.random.randn(self.inputLayerSize, self.outputLayerSize)))

    def forward(self, input_matrix):
        self.z = []
        self.a = []
        self.z.append(np.dot(input_matrix, self.W[0]))
        for i in range(len(self.W) - 1):
            self.a.append(self.sigmoid(self.z[i]))
            self.z.append(np.dot(self.a[i], self.W[i+1]))
        y_hat = self.sigmoid(self.z[len(self.z) - 1])
        return y_hat

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def get_params(self):
        params = None
        for i in range(len(self.W)):
            if params is None:
                params = np.concatenate(self.W[i].ravel())
            else:
                params = np.concatenate((params.ravel(), self.W[i].ravel()), axis=1)
        return params


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

NN = NeuralNetwork((35, 15, 10))
print(NN.forward(X))

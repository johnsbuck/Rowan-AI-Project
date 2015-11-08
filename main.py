import numpy as np
# from scipy import optimize

__author__ = 'John Bucknam & Bill Clark'

# Practice Inputs
X = np.matrix([[1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 1, 0,
               0, 0, 0, 0, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 0]])

# Practice Outputs
y = np.matrix([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        # Define each layer
        self.inputLayerSize = layer_sizes[0]
        self.outputLayerSize = layer_sizes[len(layer_sizes) - 1]
        self.hiddenLayerSizes = layer_sizes[1:len(layer_sizes) - 1]

        # Set each weight depending on the number of each layer being paired
        self.W = []
        if len(self.hiddenLayerSizes) != 0:
            self.W.append(np.matrix(np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[0])))
            for i in range(len(self.hiddenLayerSizes) - 1):
                self.W.append(np.matrix(np.random.randn(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i+1])))
            self.W.append(np.matrix(np.random.randn(self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize)))
        else:
            self.W.append(np.matrix(np.random.randn(self.inputLayerSize, self.outputLayerSize)))

    def forward(self, input_matrix):
        # Z are variables taken from the matrix multiplication of inputs of a node in a layer
        self.z = []
        # A are variables taken from inserting the Z variable into the sigmoid function
        self.a = []

        # Append each Z value matrix
        self.z.append(np.dot(input_matrix, self.W[0]))
        for i in range(len(self.W) - 1):
            # Append each A value from the sigmoid(Z)
            self.a.append(self.sigmoid(self.z[i]))
            self.z.append(np.dot(self.a[i], self.W[i+1]))
        y_hat = self.sigmoid(self.z[len(self.z) - 1])
        return y_hat

    # Static Sigmoid Function
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    # Sigmoid Prime (will provide proof from reference)
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    # Sum of Squares Error Cost Function
    def cost_function(self, X, y):
        y_hat = self.forward(X)
        return .5 * np.sum(y - y_hat)**2

    def cost_function_prime(self, X, y):
        # Defines output as well as sum of weighted inputs (z) and the threshold output (a)
        y_hat = self.forward(X);
        delta = []

        delta.append(np.multiply(-(y-y_hat)), )


    # Obtains the different weights in a 1-D matrix
    def get_params(self):
        params = None
        for i in range(len(self.W)):
            if params is None:
                params = np.concatenate(self.W[i].ravel())
            else:
                params = np.concatenate((params.ravel(), self.W[i].ravel()), axis=1)
        return params

    # all the comments.
    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        hiddenStart = 0
        hiddenEnd = self.hiddenLayerSizes[0] * self.inputLayerSize
        self.W[0] = np.reshape(params[hiddenStart:hiddenEnd], (self.inputLayerSizes, self.hiddenLayerSizes[0]))

        for layer in range(1,self.hiddenLayerSizes):
            hiddenStart = hiddenEnd+1
            hiddenEnd = self.hiddenLayerSizes[layer] * self.inputLayerSizes[layer-1]
            self.W[layer] = np.reshape(params[hiddenStart:hiddenEnd],
                                       (self.hiddenLayerSizes[layer-1],self.hiddenLayerSizes[layer]))
        hiddenStart = hiddenEnd+1
        hiddenEnd = self.hiddenLayerSizes[len(self.hiddenLayerSizes)-1] * self.outputLayerSize
        self.W[len(W)-1] = np.reshape(params[hiddenStart:hiddenEnd],
                                   (self.hiddenLayerSizes[len(self.hiddenLayerSizes)-1], self.outputLayerSize))

# Unfinished Trainer Class
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
        self.N.setParams(params)

# SCRIPT
NN = NeuralNetwork((35, 15, 10))
print(NN.forward(X))
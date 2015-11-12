import numpy as np
from scipy import optimize

__author__ = 'John Bucknam & Bill Clark'

# Practice Inputs
x = np.matrix([[1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                0, 1, 1, 1, 0,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 1, 0,
                0, 0, 1, 0, 0,
                0, 1, 0, 0, 0,
                0, 1, 0, 0, 0,
                0, 1, 0, 0, 0],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [0, 0, 1, 1, 1,
                0, 1, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 0, 1, 1, 0,
                1, 1, 0, 0, 1,
                1, 1, 0, 0, 1,
                0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                1, 0, 0, 1, 1,
                1, 1, 1, 1, 0],
               [1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [0, 0, 0, 1, 1,
                0, 0, 1, 0, 1,
                0, 1, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0,
                1, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 0,
                0, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1],
               [0, 1, 1, 0, 0,
                1, 0, 0, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 1, 0, 0,
                0, 1, 0, 0, 0,
                1, 1, 1, 1, 1],
               [0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                1, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                1, 1, 1, 1, 1],
               [0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0,
                1, 1, 0, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 0, 1, 1,
                0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1]])

# Practice Outputs
y = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
'''
y = np.matrix([[0.9],[0.9], [0.8], [0.8], [0.7], [0.7], [0.6], [0.6], [0.5], [0.5],
               [0.4], [0.4], [0.3], [0.3], [0.2], [0.2], [0.1], [0.1], [0.0], [0.0]])
'''
class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        # Define each layer
        self.inputLayerSize = layer_sizes[0]
        self.outputLayerSize = layer_sizes[len(layer_sizes) - 1]
        self.hiddenLayerSizes = layer_sizes[1:len(layer_sizes) - 1]

        # Set each weight depending on the number of each layer being paired
        self.weight = []

        # If there are hidden layers
        if len(self.hiddenLayerSizes) != 0:
            # Add random weights from input layer to hidden layer 0
            self.weight.append(np.matrix(np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[0])))
            # For each hidden layer
            for i in range(len(self.hiddenLayerSizes) - 1):
                # Add random weights between each hidden layer
                self.weight.append(np.matrix(np.random.randn(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i + 1])))
            # Add random weights between the last hidden layer and output layer
            self.weight.append(
                np.matrix(np.random.randn(self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize)))
        else:
            # Add random weights between the input layer and output layer
            self.weight.append(np.matrix(np.random.randn(self.inputLayerSize, self.outputLayerSize)))

    def forward(self, input_matrix):
        # Values taken from each set of weights * input summed together
        self.inputSum = []
        # The inputSums inserted into the threshold function (sigmoid)
        self.threshold = []

        # Append each inputSum value matrix
        self.inputSum.append(np.dot(input_matrix, self.weight[0]))
        for i in range(len(self.weight) - 1):
            # Append each A value from the sigmoid(Z)
            self.threshold.append(self.sigmoid(self.inputSum[i]))
            self.inputSum.append(np.dot(self.threshold[i], self.weight[i + 1]))
        y_hat = self.sigmoid(self.inputSum[len(self.inputSum) - 1])
        return y_hat

    # Static Sigmoid Function
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Sigmoid Derivative Function
    def sigmoid_prime(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    # Static TanH Function
    @staticmethod
    def tanh(z):
        return np.tanh(z)

    # Static TanH Derivative Function
    @staticmethod
    def tanh_prime(z):
        return ((2 * np.cosh(z))/(np.cosh(2 * z) + 1)) ** 2

    # Cross Entropy Error Function
    def cost_function(self, x, y):
        y_hat = self.forward(x)
        return -1 * np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1 - y, np.log(1 - y_hat)))
        # return .5 * np.sum(abs(y - y_hat)) ** 2

    def cost_function_prime(self, x, y):
        y_hat = self.forward(x);

        delta = []
        derived = []
        delta.append(np.multiply(-(y-y_hat), self.sigmoid_prime(self.inputSum[len(self.inputSum) - 1])))
        derived.append(np.dot(self.threshold[len(self.threshold) - 1].T, delta[len(delta) - 1]))

        for i in range(2, len(self.inputSum)):
            delta.append(np.array(np.dot(delta[len(delta) - 1], self.weight[len(self.weight) - i + 1].T)) * np.array(self.sigmoid_prime(self.inputSum[len(self.inputSum) - i])))
            derived.append(np.dot(self.threshold[len(self.threshold) - i].T, delta[len(delta) - 1]))

        delta.append(np.array(np.dot(delta[len(delta) - 1], self.weight[1].T)) * np.array(self.sigmoid_prime(self.inputSum[0])))
        derived.append(np.dot(x.T, delta[len(delta) - 1]))
        return derived

    def compute_gradients(self, x, y):
        # Obtains the derived costs over each derived weight set
        derived = self.cost_function_prime(x, y)

        # Unravels the derived cost for each set of weights and concatenates them all together
        params = derived[len(derived) - 1].ravel();
        for i in range(len(derived) - 1):
            params = np.concatenate((params.ravel(), derived[len(derived) - 2 - i].ravel()), axis=1)
        return params

    # Obtains the different weights in a 1-D matrix
    def get_params(self):
        params = None
        for i in range(len(self.weight)):
            if params is None:
                params = np.concatenate(self.weight[i].ravel())
            else:
                params = np.concatenate((params.ravel(), self.weight[i].ravel()), axis=1)
        return params

    def set_params(self, params):
        # Starting position of first set of weights
        hiddenStart = 0
        # Ending position of first set of weights
        hiddenEnd = self.hiddenLayerSizes[0] * self.inputLayerSize
        # Sets the first set of weights
        self.weight[0] = np.reshape(params[hiddenStart:hiddenEnd], (self.inputLayerSize, self.hiddenLayerSizes[0]))

        # for each hidden layer
        for layer in range(0, len(self.hiddenLayerSizes) - 1):
            # new start position is the previous end position
            hiddenStart = hiddenEnd
            # set new end position
            hiddenEnd = hiddenStart + self.hiddenLayerSizes[layer] * self.hiddenLayerSizes[layer + 1]
            # Sets the set of weights to weight list
            self.weight[layer + 1] = np.reshape(params[hiddenStart:hiddenEnd],
                                       (self.hiddenLayerSizes[layer], self.hiddenLayerSizes[layer + 1]))
        # Setting the final set of weights to output
        hiddenStart = hiddenEnd
        hiddenEnd = hiddenStart + self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1] * self.outputLayerSize
        self.weight[len(self.weight) - 1] = np.reshape(params[hiddenStart:hiddenEnd],
                                        (self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize))

class Trainer(object):
    def __init__(self, N):
        self.N = N

    def set_params(self, params):
        self.N.set_params(params)

    def cost_function_wrapper(self, params, x, y):
        # Sets the parameters
        self.N.set_params(params)
        # Gets the cost
        cost = self.N.cost_function(x, y)
        # Obtain the derived cost for each derived weights
        grad = self.N.compute_gradients(x, y)
        # Conversion
        grad = np.array(grad)
        grad.shape = (grad.shape[1],)
        return cost, grad

    def train(self, x, y):
        # Parameters of weights from the Neural Network
        params = self.N.get_params()

        # Options: maximum # of iterations and show information display after minimizing
        options = {'maxiter': 200, 'disp': False}
        results = optimize.minimize(self.cost_function_wrapper, params, jac=True, method='BFGS',
                                    args=(x, y), options=options, callback=self.set_params)
        self.N.set_params(results.x)

# SCRIPT
NN = NeuralNetwork((35, 10, 10))
train = Trainer(NN)
print(NN.forward(x))
print(NN.cost_function(x, y))
train.train(x, y)
while np.isnan(NN.cost_function(x, y)) or not (y * 10 == np.round(NN.forward(x) * 10)).all():
    NN = NeuralNetwork((35, 10, 10))
    train = Trainer(NN)
    train.train(x, y)
print(NN.forward(x))
print(np.round(NN.forward(x) * 10))
print(NN.cost_function(x, y))

print(np.round(NN.forward(np.matrix([
                            [0, 0, 0, 0, 0,
                             0, 1, 1, 1, 0,
                             0, 1, 0, 1, 0,
                             0, 1, 1, 1, 0,
                             0, 0, 0, 1, 0,
                             0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0]])))*10)

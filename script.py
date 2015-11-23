__author__ = 'Bill Clark & John Bucknam'
import NeuralNetwork
import numpy as np
import sys

#Reads in file as an array of arrays.
def readFile(name):
    X = []
    f = open(name,'r')
    for line in f:
        X.append([int(y) for y in line.split(' ')])
    f.close()
    return np.array(X)


def run():
    X = readFile(sys.argv[1])
    Y = readFile(sys.argv[2])

    NN = NeuralNetwork.NeuralNetwork((35, 10, 10))
    train = NeuralNetwork.Trainer(NN)
    raw_input("Now printing an initial run on the 20 base inputs and their cost function.")
    print(NN.forward(X))
    print(NN.cost_function(X, Y))
    raw_input("Training the network, then training on a monte carlo.")
    train.train(X, Y)
    while np.isnan(NN.cost_function(X, Y)) or not (Y * 10 == np.round(NN.forward(X) * 10)).all():
        NN = NeuralNetwork.NeuralNetwork((35, 10, 10))
        #print(NN.cost_function(X, Y))
        train = NeuralNetwork.Trainer(NN)
        train.train(X, Y)
    raw_input("Now printing the final match results.")
    print(np.round(NN.forward(X) * 10))

    while 1:
        ans = raw_input("input a command, forward on file, exit: ")
        if ans.split(' ')[0] == 'forward':
                print ans.split(' ')
                input = readFile(ans.split(' ')[1])
                output = NN.forward(input);
                output = np.round(np.multiply(output, 10));
                print(output)
        elif ans.split(' ')[0] == 'exit':
                break
        else:
            print "Hashtag Nope Nope Nope."

run()
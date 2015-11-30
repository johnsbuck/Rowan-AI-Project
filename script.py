__author__ = 'Bill Clark & John Bucknam'
import NeuralNetwork
import numpy as np
import sys
import os.path

#Reads in file as an array of arrays.
def readFile(name):
    X = []
    if os.path.isfile(name):
        f = open(name,'r')
        for line in f:
            X.append([int(y) for y in line.split(' ')])
        f.close()
        return np.array(X)
    return False


def run():
    X = readFile(sys.argv[1])
    Y = readFile(sys.argv[2])

    NN = NeuralNetwork.NeuralNetwork((35, 10, 10))
    train = NeuralNetwork.Trainer(NN)
    raw_input("Now printing an initial run on the " + str(X.shape[0]) + " base inputs and their cost function.")
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
        if ans.split(' ')[0] == 'forward' and len(ans.split(' ')) > 1:
                print ans.split(' ')
                input = readFile(ans.split(' ')[1])
                output = NN.forward(input);
                output = np.round(np.multiply(output, 10));
                print(output)

                valid = raw_input("Is this the expected output? (y/n): ")
                if valid == "n":
                    actualOutput = raw_input("What is the correct output (0-9): ")
                    try:
                        actualOutput = int(actualOutput)
                        if actualOutput >= 0 and actualOutput < 10:
                            with open(sys.argv[1], "a") as trainInput:
                                with open(ans.split(' ')[1], "r") as newInput:
                                    trainInput.write(newInput.read())
                                    trainInput.close()
                                    newInput.close()
                            with open(sys.argv[2], "a") as trainOutput:
                                newData = ""
                                zeroes = 9 - actualOutput
                                while actualOutput > 0:
                                    newData += "0 "
                                    actualOutput = actualOutput - 1
                                newData += "1 "
                                while zeroes > 0:
                                    if zeroes == 1:
                                        newData += "0\n"
                                    else:
                                        newData += "0 "
                                    zeroes = zeroes - 1
                                trainOutput.write(newData)
                                trainOutput.close()
                            print("Is added to training data. Will not be implemented until restart.")
                        else:
                            print("Invalid input")
                    except ValueError:
                        print("Invalid input.")
        elif ans.split(' ')[0] == 'exit':
                break
        else:
            print "Hashtag Nope Nope Nope."

run()
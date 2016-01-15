#!/usr/bin/env python

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

def readWeights(name):
    if os.path.isfile(name):
        f = open(name, 'r')
        prevWeights = np.fromstring(f.readline(), dtype=float, sep=" ")
        f.close()
        return prevWeights
    return False

#In place of a main, as python lacks one. Call run to read in the training data and train,
#As well as run a monte carlo to find a satisfactory network. Then it will sit in a loop
#waiting for input commands.
def run():
    #reads in the test data files.
    X = readFile(sys.argv[1])
    Y = readFile(sys.argv[2])
    if len(sys.argv) >= 4:
        weights = readWeights(sys.argv[3])

    #Creates a trainer and network, created as a 35 input to 10 hidden to 10 output.
    print "This neural network will take in 35 inputs and output 10 floats (rounded)."

    numHiddenLayers = ""
    while not numHiddenLayers.isdigit():
        numHiddenLayers = raw_input("How many hidden layers do you want: ")
    numHiddenLayers = int(numHiddenLayers)

    layerNodes = (35,)

    print("Enter the number of nodes for each hidden layer")
    for i in range(1, numHiddenLayers+1):
        hiddenLayerNodes = ""
        while not hiddenLayerNodes.isdigit():
            hiddenLayerNodes = raw_input("Hidden Layer " + str(i) + ": ")
        layerNodes = layerNodes + (int(hiddenLayerNodes),)
        print layerNodes

    layerNodes = layerNodes + (10,)
    print layerNodes

    NN = NeuralNetwork.NeuralNetwork(layerNodes)
    if len(sys.argv) >= 4:
        NN.set_params(weights)

    train = NeuralNetwork.Trainer(NN)

    # As the print states, runs a forward operation on the network with it's randomly generated weights.
    raw_input("Now printing an initial run on the " + str(X.shape[0]) + " base inputs and their cost function.")
    print(NN.forward(X))
    print(NN.cost_function(X, Y))

    #Trains the network using the trainer and test data.
    max_count = ""
    while not max_count.isdigit():
        max_count = raw_input("Training the network, then training on a monte carlo.\n# of Cycles: ")
    max_count = int(max_count)

    #This is our monte carlo. Continually trains networks until one statisfies our conditions.
    if max_count > 0:
        count = 0

        while np.isnan(NN.cost_function(X, Y)) or count < max_count:
            train = NeuralNetwork.Trainer(NN)
            train.train(X, Y)
            if count == 0:
                bestNN = NeuralNetwork.NeuralNetwork(layerNodes)
                bestNN.set_params(NN.get_params())
                print("Cost: " + str(bestNN.cost_function(X, Y)))
            elif bestNN.cost_function(X, Y) > NN.cost_function(X, Y):
                bestNN = NeuralNetwork.NeuralNetwork(layerNodes)
                bestNN.set_params(NN.get_params())
                print("New cost: " + str(bestNN.cost_function(X, Y)))
            count += 1
            NN = NeuralNetwork.NeuralNetwork(layerNodes)
            print("Current cycle: " + str(count))

        NN.set_params(bestNN.get_params())
    #Print the results of the training and monte carlo.
    raw_input("Now printing the final match results.")
    print(np.round(NN.forward(X) * 10))
    print("Cost function: " + str(NN.cost_function(X, Y)))

    #Input control loop.
    while 1:
        ans = raw_input("input a command: forward <file>, save <file>, or exit: ")

        # When a user inputs forward and a file, read in the file and run forward using it.
        if ans.split(' ')[0] == 'forward' and len(ans.split(' ')) > 1:
                print ans.split(' ')
                input = readFile(ans.split(' ')[1])
                output = NN.forward(input);
                output = np.round(np.multiply(output, 10));
                print(output)

                #Additional checker tool, allows for a forwarded file to be added to test data.
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
                                newData += "1"
                                if zeroes == 0:
                                    newData += "\n"
                                else:
                                    newData += " "
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
        elif ans.split(' ')[0] == 'save' and len(ans.split(' ')) > 1:
            print ans.split(' ')
            with open(ans.split(' ')[1], "w") as weights:
                print NN.get_params()
                weights.write(str(NN.get_params()).replace("[", "").replace("]","")
                              .replace("\n", "").replace("   "," ").replace("  ", " "))
                weights.close()
        # Exit.
        elif ans.split(' ')[0] == 'exit':
                break
        # Completely invalid input.
        else:
            print "#NopeNopeNope."

#If this script is being run, as opposed to imported, run the run function.
if __name__ == '__main__':
    if len(sys.argv) >= 2:
        run()
    else:
        print "Invalid number of inputs. (Requires expected input & expected output files)"

__author__ = 'Bill Clark & John Bucknam'

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
    weights = readWeights(sys.argv[3])

    #Creates a trainer and network, created as a 35 input to 10 hidden to 10 output.
    NN = NeuralNetwork.NeuralNetwork((35, 10, 10))
    NN.set_params(weights)

    train = NeuralNetwork.Trainer(NN)

    # As the print states, runs a forward operation on the network with it's randomly generated weights.
    raw_input("Now printing an initial run on the " + str(X.shape[0]) + " base inputs and their cost function.")
    print(NN.forward(X))
    print(NN.cost_function(X, Y))

    #Trains the network using the trainer and test data.
    raw_input("Training the network, then training on a monte carlo.")
    train.train(X, Y)

    #This is our monte carlo. Continually trains networks until one statisfies our conditions.
    while np.isnan(NN.cost_function(X, Y)) or not (Y == np.round(NN.forward(X))).all():
        NN = NeuralNetwork.NeuralNetwork((35, 10, 10))
        train = NeuralNetwork.Trainer(NN)
        train.train(X, Y)

    #Print the results of the training and monte carlo.
    raw_input("Now printing the final match results.")
    print(np.round(NN.forward(X) * 10))

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
            print "Hashtag Nope Nope Nope."

#If this script is being run, as opposed to imported, run the run function.
if __name__ == '__main__':
    run()
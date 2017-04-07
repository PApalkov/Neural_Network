import numpy as np
import scipy.special
#import matplotlib.pyplot
import scipy.misc
import pandas as pd


class NeuralNetwork:
    """Class with neural network"""
    def __init__(self, inputnodes = None, hiddennodes = None, outputnodes = None, learningrate = None):

        #set number of nodes in each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #set learning rate
        self.lr = learningrate

        #reading weights from csv
        #it is supposed that we have already trained the network
        self.who = pd.read_csv("who.csv", usecols = list(range(1, self.hnodes + 1))).as_matrix()
        self.win = pd.read_csv("win.csv", usecols = list(range(1, self.inodes + 1))).as_matrix()


        #activation fuction is sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass


    def weights_init(self):
        """Initilize weights with random values"""
        # weights initialization by random values [-0.5; 0.5)
        self.win = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))


    def train(self, inputs_list, targets_list):
        """Training method"""
        #convert inputs and targets to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.win, inputs)

        #calculate the signals emerging the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #calculate the signals emerging the final output layer
        final_outputs = self.activation_function(final_inputs)

        #error is the (target - actual)
        output_errors = targets - final_outputs

        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.win += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        """Making prediction method"""

        #convert inputs to 2d aray
        inputs = np.array(inputs_list, ndmin = 2).T


        hidden_inputs = np.dot(self.win, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save(self):
        """Saving veights in csv file"""
        df = pd.DataFrame(self.win)
        df.to_csv("win.csv")

        df = pd.DataFrame(self.who)
        df.to_csv("who.csv")

    def recognize_image(self, filename):
        """Method wich recognizes images"""

        #opening image and converting it into array
        img_array = scipy.misc.imread(filename, flatten=True)

        #making matrix 28x28
        img_data = np.asfarray(255.0 - img_array.reshape(28*28))

        #matrix's values normalization
        #making values in (0; 1)
        img_data = (img_data / 255 * 0.99) + 0.01

        #getting the list with network ansvers
        answer_list = self.query(img_data)

        #choosing the max value and return its index
        index = 0
        max_value = answer_list[0]
        for i in range(1,10):
            if answer_list[i] > max_value:
                max_value = answer_list[i]
                index = i


        return index, max_value


def train(n):
    """Function which train network by mnist data"""

    training_data_file = open("mnist_dataset/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    #making new random weighs
    n.weights_init()


    epochs = 3
    for _ in range(0, epochs):
        for record in training_data_list:
             all_values = record.split(",")

             #make normalization
             inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

             targets = np.zeros(n.onodes) + 0.1
             targets[int(all_values[0])] = 0.99
             n.train(inputs, targets)


    #savinf weights in file
    n.save()

def test(n):
    "Tests the existing network by mnist data"
    test_data_file = open("mnist_dataset/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    answers = []

    for record in test_data_list:

        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        correct_answer = int(all_values[0])
        answer = np.argmax(n.query(inputs))

        if (correct_answer == answer):
            answers.append(1)
        else:
            answers.append(0)


    scores_array = np.asarray(answers)
    quality = scores_array.sum() / scores_array.size

    print("Quality:", quality)



def main():

    input_nodes = 28 * 28
    hidden_nodes = 300
    output_nodes = 10

    learning_rate = 0.008

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    #train network and save weights before using program
    #train(n)
    #test(n)


if __name__ == '__main__':
    main()

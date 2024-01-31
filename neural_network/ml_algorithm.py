#!/usr/bin/python
"""
============================================================================
Name: Neural Network
Group: 4
Author(s): Pierre Visconti, Reece Reklai, Rob Frohne
Course: CPTR330
Assignment: Lab 4
Description: Implementation of the ML algorithm neural network.
============================================================================
"""

import numpy as np
from scipy import optimize


class MLAlgorithm:
    """
    Implement the Neural Network algorithm
    """

    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.
    def __init__(self, parameters):
        """
        Initial all algorithm variables
        """
        # initialize input_size variable
        try:
            self.input_size = int(parameters["input_size"])
        except KeyError:  # no parameter passed
            print("No 'input_size' value given, set to 1 by default")
            self.input_size = 1
        except ValueError:  # empty value passed
            print("No 'input_size' value given, set to 1 by default")
            self.input_size = 1
        # initialize hidden_layers variable
        try:
            self.hidden_layers = [int(parameters["hidden_layers"])]
        except KeyError:  # no parameter passed
            print("No 'hidden_layers' value given, set to 1 by default")
            self.hidden_layers = [2, 4]
        except ValueError:  # empty value passed
            print("No 'hidden_layers' value given, set to 1 by default")
            self.hidden_layers = [2, 4]
        # initialize output_size variable
        try:
            self.output_size = int(parameters["output_size"])
        except KeyError:  # no parameter passed
            print("No 'output_size' value given, set to 1 by default")
            self.output_size = 1
        except ValueError:  # empty value passed
            print("No 'output_size' value given, set to 1 by default")
            self.output_size = 1
        print(self.input_size, self.hidden_layers, self.output_size)

        # initialize other variables
        self.algorithm = "Neural Network"
        self.data = None
        self.labels = None
        self.net = None
        self.costs = []

    def get_algorithm(self):
        """
        Return the name of the algorithm.
        """
        return self.algorithm

    def train(self, dataset, labels):
        """
        Train the model using the input dataset and labels.
        The dataset and labels are two dimensional lists.
        """
        self.data = dataset
        self.labels = labels

    def get_predictions(self, test_set):
        """
        Return predictions for the test_set.
        The dataset and labels are two dimensional lists.
        """

        # get max value of labels before normalization
        max_value = self.get_max(self.labels)
        print(max_value)
        # combining train and test sets together before normalizing
        dataset = self.data + test_set
        dataset = self.normalize(dataset)
        # Normalize the entire dataset
        dataset = self.normalize(dataset)
        self.labels = self.normalize(self.labels)
        # splitting dataset back into train and test sets
        self.data = dataset[0 : len(self.data)]
        test_set = dataset[len(self.data) :]

        # converting data to numpy array
        test_set = np.array(test_set, dtype=np.float64)
        self.data = np.array(self.data, dtype=np.float64)
        self.labels = np.array(self.labels, dtype=np.float64)

        # train the model
        self.build_model()

        # predict test set
        output = self.net.forward(test_set)
        output = [i * max_value for i in output]

        return output

    def build_model(self):
        """
        Builds the neural network model
        """
        # define net object with inputted values
        self.net = NeuralNetwork(
            self.input_size, self.output_size, self.hidden_layers[0]
        )

        params0 = self.net.get_params()

        # specifies net object options
        options = {"maxiter": 200, "disp": True}
        _res = optimize.minimize(
            self.cost_function_wrapper,
            params0,
            jac=True,
            method="BFGS",
            args=(self.data, self.labels),
            options=options,
            callback=self.call_back_f,
        )

        self.net.set_params(
            _res.x, self.input_size, self.hidden_layers[0], self.output_size
        )

    def call_back_f(self, params):
        """
        Calls back our function f
        """
        self.net.set_params(
            params, self.input_size, self.hidden_layers[0], self.output_size
        )
        self.costs.append(self.net.cost_function(self.data, self.labels))

    def cost_function_wrapper(self, params, data, labels):
        """
        Wrapper for the cost function
        """
        self.net.set_params(
            params, self.input_size, self.hidden_layers[0], self.output_size
        )
        cost = self.net.cost_function(data, labels)
        grad = self.net.compute_gradients(data, labels)
        return cost, grad

    @staticmethod
    def get_max(labels):
        """
        Finds the max value for the labels
        """
        max_value = 0
        for row in labels:
            if max_value < row[0]:
                max_value = row[0]
        return max_value

    @staticmethod
    def normalize(dataset):
        """
        Normalize the data

        Code provided by Jared Sexton and Jahri Harris
        """
        # Find min and max values
        minimum_values = dataset[0].copy()
        maximum_values = dataset[0].copy()
        for row in dataset:
            for col, cell in enumerate(row):
                if cell < minimum_values[col]:
                    minimum_values[col] = cell
                if cell > maximum_values[col]:
                    maximum_values[col] = cell
        # Convert dataset
        for row in dataset:
            for col, cell in enumerate(row):
                if maximum_values[col] - minimum_values[col] != 0:
                    row[col] = (cell - minimum_values[col]) / (
                        maximum_values[col] - minimum_values[col]
                    )
                else:
                    row[col] = 0

        return dataset


class NeuralNetwork:
    """
    Neural Network object
    """

    def __init__(self, input_size, output_size, hidden):
        """
        Initialize all Neural Network variables
        """
        # Weights (parameters)
        self.weight_1 = np.random.randn(input_size, hidden)
        self.weight_2 = np.random.randn(hidden, output_size)
        self.a_2 = None
        self.z_2 = None
        self.z_3 = None

        self.y_hat = None

    def forward(self, data):
        """
        Propagate inputs though network
        """
        self.z_2 = np.dot(data, self.weight_1)
        self.a_2 = self.sigmoid(self.z_2)
        self.z_3 = np.dot(self.a_2, self.weight_2)
        y_hat = self.sigmoid(self.z_3)
        return y_hat

    def cost_function(self, data, labels):
        """
        Compute cost for given X,y, use weights already stored in class.
        """
        self.y_hat = self.forward(data)
        cost = 0.5 * sum((labels - self.y_hat) ** 2)
        return cost

    def cost_function_prime(self, data, labels):
        """
        Compute derivative with respect to weight_1 and weight_2 for a given data and labels:
        """
        self.y_hat = self.forward(data)

        delta_3 = np.multiply(-(labels - self.y_hat), self.sigmoid_prime(self.z_3))
        d_weight_2 = np.dot(self.a_2.T, delta_3)

        delta_2 = np.dot(delta_3, self.weight_2.T) * self.sigmoid_prime(self.z_2)
        d_weight_1 = np.dot(data.T, delta_2)

        return d_weight_1, d_weight_2

    def get_params(self):
        """
        Get weight_1 and weight_2 unrolled into vector:
        """
        params = np.concatenate((self.weight_1.ravel(), self.weight_2.ravel()))
        return params

    def set_params(
        self, params, input_layer_size, hidden_layer_size, output_layer_size
    ):
        """
        Set weight_1 and weight_2 using single parameter vector.
        """
        weight_1_start = 0
        weight_1_end = hidden_layer_size * input_layer_size
        self.weight_1 = np.reshape(
            params[weight_1_start:weight_1_end],
            (input_layer_size, hidden_layer_size),
        )
        weight_2_end = weight_1_end + hidden_layer_size * output_layer_size
        self.weight_2 = np.reshape(
            params[weight_1_end:weight_2_end],
            (hidden_layer_size, output_layer_size),
        )

    def compute_gradients(self, data, labels):
        """
        Computes the gradients
        """
        d_weight_1, d_weight_2 = self.cost_function_prime(data, labels)
        return np.concatenate((d_weight_1.ravel(), d_weight_2.ravel()))

    @staticmethod
    def sigmoid(input_z):
        """
        Apply sigmoid activation function to scalar, vector, or matrix
        """
        return 1.0 / (1.0 + np.exp(-input_z))

    @staticmethod
    def sigmoid_prime(input_z):
        """
        Gradient of sigmoid
        """
        return np.exp(-input_z) / ((1.0 + np.exp(-input_z)) ** 2)

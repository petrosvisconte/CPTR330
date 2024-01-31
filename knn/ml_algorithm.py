#!/usr/bin/python
"""============================================================================
# Name: kNN
# Username: smitta
# Username: viscpi
# Course: CPTR330
# Assignment: Lab 1
# Description: Implementation of the ML algorithm kNN.
#============================================================================
"""

# pylint: disable = invalid-name, consider-using-enumerate

from math import sqrt


class MLAlgorithm:
    """
    Implement k Nearest Neighbors
    """

    def __init__(self, parameters):
        """
        Initializes all the variable for the algorithm.
        Parameters will hold key:value pairs for defining details in the
        algorithm.
        """
        self.k = int(parameters["k"])

        try:
            self.n = bool(parameters["n"])
        except KeyError:
            print("No 'n' value given, n set to false")
            self.n = False
        self.dataset = None
        self.labels = None

    def get_algorithm(self):
        """
        Returns the name of the algorithm.
        """
        return "k Nearest Neighbors"

    def train(self, dataset, labels):
        """
        Trains the model based on the dataset and labels.
        """
        self.dataset = [d + l for d, l in zip(dataset, labels)]
        self.labels = labels

    def get_predictions(self, test_set):
        """
        Return the predictions for testSet using the algorithm model.
        """
        distances = []
        neighbors = []
        predictions = []

        self.min_max(self.dataset, test_set)

        # Normalize the entire dataset
        if self.n:
            test_set = self.normalize(test_set, False)
            self.dataset = self.normalize(self.dataset, True)

        for test_row in test_set:
            # Determine Distances
            for dataset_row in self.dataset:
                distance = self.distance(test_row, dataset_row)
                distances.append((distance, dataset_row[-1]))

            distances.sort()

            # Determine Neighbors
            for j in range(self.k):
                neighbors.append(distances[j][1])

            # Return Label
            predictions.append([(max(set(neighbors), key=neighbors.count))])
            distances = []
            neighbors = []

        # Return Predictions
        return predictions

    def distance(self, a, b):
        """
        Calculates the euclidean distance between two points
        """
        distance = 0.0
        for i in range(len(a)):
            distance += (a[i] - b[i]) ** 2
        return sqrt(distance)

    def normalize(self, dataset, train):
        """ "
        Function to normalize an array
        """
        k = 0

        if train:
            col = len(dataset[0]) - 1
        else:
            col = len(dataset[0])

        while k < col:
            j = 0
            while j < len(dataset):
                dataset[j][k] = (dataset[j][k] - self.min[k]) / (
                    self.max[k] - self.min[k]
                )
                j += 1
            k += 1
        return dataset

    def min_max(self, train_set, test_set):
        """
        Finds the min and max values for each column in the entire dataset
        """

        k = 0
        self.max = []
        self.min = []

        dataset = train_set + test_set

        col = len(train_set[0]) - 1
        z = 0
        while z < col:
            self.max.append([0])
            self.min.append([100000])
            z += 1

        while k < col:
            # Find min and max of columns
            self.min[k] = float(dataset[0][k])
            self.max[k] = float(dataset[0][k])
            i = 0
            while i < len(dataset):
                if self.min[k] > dataset[i][k]:
                    self.min[k] = float(dataset[i][k])
                if self.max[k] < dataset[i][k]:
                    self.max[k] = float(dataset[i][k])
                i += 1
            k += 1

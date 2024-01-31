#!/usr/bin/python
"""
============================================================================
Name: Naive Bayes
Group: 4
Author(s): Pierre Visconti, Adam Taylor
Course: CPTR330
Assignment: Lab 2
Description: Implementation of Naive Bayes Algorithm
============================================================================
"""

# Your code goes here.


class MLAlgorithm:
    """
    Implementation of the ML algorithm Naive Bayes.
    Building on a student submission.
    """

    def __init__(self, parameters):
        """
        Initializes all the variable for the algorithm.
        Parameters will hold key:value pairs for defining details in the
        algorithm.
        """
        try:
            self.laplace = str(parameters["laplace"])
            if self.laplace == "":
                print("No 'laplace' parameter given, set to false by default")
                self.laplace = 0
            elif self.laplace.lower() in ["true", "True", "t", "T"]:
                print("'laplace 'set to True")
                self.laplace = 1
            else:
                print(
                    "Not a valid input for 'laplace' parameter, set to false by default"
                )
                self.laplace = 0
        except KeyError:
            print("No 'laplace' parameter given, set to false by default")
            self.laplace = 0

        # initializes all the variables
        self.training_set = []
        self.labels_set = []
        self.freq = []
        self.like = []
        self.prob_classes = {}
        self.algorithm = "Naive Bayes"

    def get_algorithm(self):
        """
        Returns the name of the algorithm.
        """
        return self.algorithm

    def train(self, dataset, labels):
        """
        Train the model based on the dataset and labels.
        """
        # Merge dataset and labels.
        self.training_set = [d + l for d, l in zip(dataset, labels)]
        self.labels_set = labels
        # build frequency and likelihood tables
        self.build_freq_table()
        self.build_like_table()
        # calculates probabilities for each class
        self.build_classes_prob()

    def get_predictions(self, test_set):
        """
        Return the predictions for test_set using the algorithm model.
        """
        result = []
        # for each row in the test set, calculate a prediction
        for row in test_set:
            # make a copy to preserve original values
            prob = self.prob_classes.copy()
            # calculate conditional probability
            for index, col in enumerate(row):
                self.prob_update(index, col, prob)
            # normalize probability
            max_value = max(prob.values())
            # collect the key of the max probability
            for pair in prob.items():
                if pair[1] == max_value:
                    max_key = pair[0]
            result.append([max_key])
        return result

    def build_freq_table(self):
        """
        Builds the frequency table for the training set
        """
        # loops over each feature in the train set
        for col in range(len(self.training_set[0])):
            freq_dict = {}
            possible_values = set()
            # loops over each row for a feature
            for row in self.training_set:
                possible_values.add(row[col])
                label = row[len(self.training_set[0]) - 1]
                # increases frequency by 1 if the category already exists, otherwise sets it to 1
                if label in freq_dict:
                    if row[col] in freq_dict[label]:
                        freq_dict[label][row[col]] += 1
                    else:
                        freq_dict[label][row[col]] = 1
                else:
                    freq_dict[label] = {}
                    if row[col] in freq_dict[label]:
                        freq_dict[label][row[col]] += 1
                    else:
                        freq_dict[label][row[col]] = 1
            for i in possible_values:
                for pair in freq_dict.items():
                    if i not in pair[1]:
                        pair[1][i] = 0
            # add the subtable created above to the frequency table
            self.freq.append(freq_dict)

    def build_like_table(self):
        """
        Builds the likelihood table for the training set
        """
        self.like = self.freq[:]
        # loops over each category in the frequency table
        for cat in self.like:
            # loops over each feature we are trying to predict
            for key in cat.keys():
                count = 0.0
                # increment over each predictive feature and collect frequency
                for key2 in cat[key].keys():
                    if key2 == "?":
                        continue
                    count += cat[key][key2]
                # calculate the conditional probabilities
                denominator = self.laplace * (len(self.like) - 1) + count
                for key2 in cat[key].keys():
                    if key2 == "?":
                        continue
                    cat[key][key2] = [
                        (cat[key][key2] + self.laplace),
                        denominator,
                    ]
                cat[key]["?"] = [self.laplace, denominator]

    def build_classes_prob(self):
        """
        Calculates the probabilities of each possible class
        """
        # loops over each row in the labels set
        for row in self.labels_set:
            if row[0] in self.prob_classes:
                self.prob_classes[row[0]] += 1.0
            else:
                self.prob_classes[row[0]] = 1.0
        # calculates the probability for each class
        for key in self.prob_classes:
            self.prob_classes[key] = self.prob_classes[key] / len(self.labels_set)

    def prob_update(self, index, col, prob):
        """
        Update prob with conditional probability
        """
        # identify the probability we want to multiply
        for key in self.like[index].keys():
            if col in self.like[index][key].keys():
                prob[key] *= (
                    self.like[index][key][col][0] / self.like[index][key][col][1]
                )
            else:
                for i in self.like[index][key].keys():
                    if i != col:
                        prob[key] *= self.laplace / self.like[index][key][i][1]
                        break

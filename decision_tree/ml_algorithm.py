#!/usr/bin/python
"""
Name: Decision Trees
Group: 4
Author(s): Pierre Visconti, Eddie Coberly
Course: CPTR330
Assignment: Lab 3
Description: Implementation of the ML algorithm decision trees.
"""

# used for getting most frequent element in a list
from statistics import mode


class MLAlgorithm:
    """
    Implement the decision tree algorithm.
    """

    def __init__(self, parameters):
        """
        Initial all algorithm variables
        """
        # initialize max_depth variable (max depth of tree)
        try:
            self.max_depth = int(parameters["max_depth"])
            if self.max_depth < 0:
                self.max_depth = 0
                print("max_depth cannot be negative, no max_depth value set")
        except KeyError:  # no parameter passed
            print("No 'max_depth' value given, this may result in a large tree")
            self.max_depth = 0
        except ValueError:  # empty value passed
            print("No 'max_depth' value given, this may result in a large tree")
            self.max_depth = 0

        # initialize min_size variable (minimum number of samples for a leaf node)
        try:
            self.min_size = int(parameters["min_size"])
            if self.min_size < 0:
                self.min_size = 0
                print("min_size cannot be negative, set to 0 by default")
        except KeyError:  # no parameter passed
            print("No 'min_size' value given, set to 0 by default")
            self.min_size = 0
        except ValueError:  # empty value passed
            print("No 'min_size' value given, set to 0 by default")
            self.min_size = 0
        # initialize variables
        self.algorithm = "Decision Tree"
        self.tree = None

    def get_algorithm(self):
        """
        Return the name of the algorithm.
        """
        return self.algorithm

    def train(self, dataset, labels):  # implement min_size
        """
        Train the model using the input dataset and labels.
        The dataset and labels are two dimensional lists.
        """
        # initialize labels as 1D list
        labels = self.column(labels, 0)
        # Create a variable that keeps track of all the features
        features = list(range(0, len(dataset[0])))
        # build the tree
        self.tree = self.make_tree(dataset, labels, features, 1)

    def get_predictions(self, test_set):
        """
        Return predictions for the test_set.
        The dataset and labels are two dimensional lists.
        """
        labels = []
        # append the label for each row to labels
        for row in test_set:
            labels.append([self.iterate_tree(row, self.tree)])

        return labels

    def iterate_tree(self, data, node):
        """
        Classifies a single object by recursively iterating through the decision tree
        """
        # stopping condition: if node is a leaf, return node value
        if node.lower_child is None and node.upper_child is None:
            # print(node.val[0])
            return node.val

        # if value of data at feature is less than node value, go to node.lower_child
        # otherwise go to node.upper_child
        if data[node.val[0]] < node.val[1]:
            return self.iterate_tree(data, node.lower_child)

        return self.iterate_tree(data, node.upper_child)

    def make_tree(self, data, labels, features, depth):
        """
        Builds the decision tree recursively.
        """
        # create a new node object
        node = Node()
        # Stopping condition: if labels contains only 1 class, set node val to that class
        if len(list(dict.fromkeys(labels))) == 1:
            node.val = labels[0]
            return node
        # Stopping condition: if there are no more features left, set node val to most common class
        if len(features) == 1:
            node.val = mode(labels)
            return node
        # Stopping condition: if current depth == max_depth, set node val to most common class
        if depth == self.max_depth:
            node.val = mode(labels)
            return node
        # Stopping condition: if current size <= min_size, do not split the node further
        if len(labels) <= self.min_size:
            node.val = mode(labels)
            return node

        # get best feature to split on, and the best split, save it as node value
        # node.val = [feature_num, split_point]
        node.val = self.get_best_feature(data, labels, features)
        if node.val[0] is None:
            node.val = mode(labels)
            return node
        # split the data based on node.val and create two subsets
        lower_data = []
        upper_data = []
        lower_labels = []
        upper_labels = []
        row_num = 0
        for row in data:
            # append to lower subsets if less than split point
            if row[node.val[0]] < node.val[1]:
                lower_data.append(row)
                lower_labels.append(labels[row_num])
            # append to upper subsets if greater than split point
            elif row[node.val[0]] > node.val[1]:
                upper_data.append(row)
                upper_labels.append(labels[row_num])
            row_num += 1

        # Build the child nodes with the subsets
        node.lower_child = self.make_tree(
            lower_data,
            lower_labels,
            features[: node.val[0]] + features[node.val[0] + 1 :],
            depth + 1,
        )
        node.upper_child = self.make_tree(
            upper_data,
            upper_labels,
            features[: node.val[0]] + features[node.val[0] + 1 :],
            depth + 1,
        )

        return node

    def get_best_feature(self, data, labels, features):
        """
        Returns a list with best feature to split on and it's split point,
        determined by the gini impurity.

        Index 0: feature number
        Index 1: split point
        """
        # initialize best_feature
        best_feature = [2, None, None]  # highest possible gini impurity is 1
        # iterate over all features
        for feature in features:
            feature_data = self.column(data, feature)
            # get the gini impurity and split point for feature
            gini = self.get_gini_impurity_feature(feature_data, labels, feature)
            # if gini impurity for feature is lower than previous low then update best feature
            if gini[0] < best_feature[0]:
                best_feature = gini
        return [best_feature[1], best_feature[2]]

    def get_gini_impurity_feature(self, feature_data, labels, feature):
        """
        Returns a list with the weighted gini impurity for a specific feature,
        the best split point for that feature, and the feature number.

        Index 0: weighted gini impurity
        Index 1: feature number
        Index 2: best split point
        """
        # initialize gini_impurity based on the indexes from above
        gini_impurity = [2, feature, None]  # highest possible gini impurity is 1

        # sorts the feature data and labels accordingly
        order = sorted(range(len(feature_data)), key=lambda i: feature_data[i])
        feature_data_sorted = [feature_data[i] for i in order]
        labels_sorted = [labels[i] for i in order]
        # all unique classes in labels
        classes = list(dict.fromkeys(labels))

        # self.get_split_points(feature_data_sorted):
        # creates a list of possible split points, using averages from sorted feature data
        # iterate over all possible split points
        for point in self.get_split_points(feature_data_sorted):
            split_index = 0
            # iterate over data until point is reached
            for element in feature_data_sorted:
                # determine where to split data
                if element > point:
                    split_index = feature_data_sorted.index(element)
                    break
            # split sorted labels in two by the value of the split point
            labels_lower = labels_sorted[:split_index]
            labels_upper = labels_sorted[split_index:]

            # get the gini impurity for the current split
            gini = self.get_gini_impurity(labels_upper, labels_lower, classes)
            # tests for min_size parameter
            if gini is False:
                continue
            # test if gini impurity for current split is better than previous splits
            if gini < gini_impurity[0]:
                gini_impurity[2] = point
                gini_impurity[0] = gini

        return gini_impurity

    def get_gini_impurity(self, label_lower, label_upper, classes):
        """
        Returns the weighted gini impurity for a specific split.
        """
        # test if split will result in a node less than min_size
        if len(label_lower) <= self.min_size or len(label_upper) <= self.min_size:
            return False

        # initialize empty lists to be the same size as number of classes
        count_classes_lower = [0.0 for x in range(len(classes))]
        count_classes_upper = [0.0 for x in range(len(classes))]

        index = 0
        # find number of each class in each split
        for cla in classes:
            count_classes_lower[index] = sum(1.0 for i in label_lower if i == cla)
            count_classes_upper[index] = sum(1.0 for i in label_upper if i == cla)
            index += 1

        # calculate gini impurity for each split: 1 - (c_1 / total)^2 - ... - (c_n / total)^2
        # initializing variables
        total_class_lower = float(sum(count_classes_lower))
        total_class_upper = float(sum(count_classes_upper))
        total_class = total_class_lower + total_class_upper
        gini_lower = 1.0  # set to 1 since we are subtracting, see formula above
        gini_upper = 1.0
        # iterate over each class
        for i in range(len(classes)):
            factor = (count_classes_lower[i] / total_class_lower) ** 2.0
            gini_lower -= factor
            factor = (count_classes_upper[i] / total_class_upper) ** 2.0
            gini_upper -= factor

        # calculate and return weighted gini impurity
        return (total_class_lower / total_class) * gini_lower + (
            total_class_upper / total_class
        ) * gini_upper

    @staticmethod
    def get_split_points(data):
        """
        Returns all possible split points for a feature.
        """
        # remove duplicates from data
        data = [*set(data)]
        # convert to float values
        data = [float(i) for i in data]
        # data = sorted(data)
        # creates a list of averages from sorted feature data, to use as possible split points
        averages = []
        # calculates averages to use as split points
        for i in range(len(data) - 1):
            average = (data[i] + data[i + 1]) / 2.0
            averages.append(average)

        return averages

    @staticmethod
    def column(data, feature):
        """
        Returns a feature from data as a 1D list.
        """
        return [row[feature] for row in data]


# pylint: disable=too-few-public-methods
class Node:
    """
    Defines a Node datastructure.
    """

    def __init__(self):
        """
        Initialize all Node variables.
        """
        self.val = None
        self.lower_child = None
        self.upper_child = None

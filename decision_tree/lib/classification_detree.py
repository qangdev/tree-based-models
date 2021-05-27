from abc import ABC, abstractmethod
from dataclasses import dataclass
from decision_tree.lib.utils import label_count

# TODO: [X] Question to Feature
# TODO: [X] Introducing branch_depth (max number of nodes a branch should have; to avoid overfeeding)
# TODO: [X] Build random tree by remove impurity part
# TODO: [X] Build random forest by combine many random trees
# TODO: [] Random Forest solve dup trees

#########################################################################
@dataclass
class Feature:
    column: any
    value: any

    def match(self, row):
        try:
            self.value = float(self.value) or int(self.value)
            return float(row[self.column]) >= self.value
        except ValueError:
            return row[self.column].lower() == str(self.value).lower()
        except Exception as e:
            raise e

    def print(self, headers=None):
        try:
            self.value = float(self.value)
            if len(headers) >= self.column:
                return "%s >= %s " % (headers[self.column], self.value)
            else:
                return "%s >= %s " % (self.column, self.value)
        except ValueError:
            if len(headers) >= self.column:
                return "%s == %s " % (headers[self.column], self.value)
            else:
                return "%s == %s " % (self.column, self.value)


@dataclass
class Node:
    question: Feature
    left_branch: object
    right_branch: object


@dataclass
class Leaf:
    predictions: dict

    def predict(self):
        # return the label with highest outcome percentage
        best_label = max(self.predictions, key=self.predictions.get)
        return best_label, self.predictions[best_label]


@dataclass
class MedianLeaf:
    predictions: dict

    def predict(self):
        # return the label in the middle
        labels = list(self.predictions.keys())
        values = list(self.predictions.values())
        values.sort()
        # get the middle point (round down)
        mid_point = round(len(values) - 1)
        return labels[mid_point], values[mid_point]


#########################################################################
class Tree(ABC):
    def __init__(self, information_gain: any, max_depth: int = 3, leaf_style = None):
        self.information_gain = information_gain  # Choices: Gini | Entropy
        self.max_depth = max_depth
        self.leaf_style = leaf_style

    def classify(self, row, node):
        if isinstance(node, self.leaf_style) or hasattr(node, "question") is False:  # Stop and predict outcome the recursion when a Leaf is found
            return node.predict()

        if node.question.match(row):
            return self.classify(row, node.left_branch)
        else:
            return self.classify(row, node.right_branch)

    @abstractmethod
    def build_tree(self, rows):
        raise NotImplementedError("NotImplementedError")

    @abstractmethod
    def split(self, rows):
        raise NotImplementedError("NotImplementedError")

    def partition(self, rows, question):
        left_branch, right_branch = [], []
        for row in rows:
            if question.match(row):
                left_branch.append(row)
            else:
                right_branch.append(row)
        return left_branch, right_branch

    def unique_values(self, rows):
        # return a list of unique values for each columns
        u_values = []
        row = rows[0]
        for col in range(len(row) - 1):
            values = set([row[col] for row in rows])
            u_values.append(values)
        return list(zip(range(len(rows)), u_values))

    def label_percentage(self, rows):
        # return the outcome percentage for each possible values
        perc_label = {}
        count_label = label_count(rows)
        for label, count in count_label.items():
            perc_label[label] = count / sum([i for i in count_label.values()])
        return perc_label

    def print_tree(self, node, spacing="", headers=None):
        if isinstance(node, self.leaf_style) or hasattr(node, "question") is False:
            print(spacing + "↓ Leaf", node.predictions)
            return

        # Print the question at this node
        print(spacing + node.question.print(headers))

        # Call this function recursively on the true branch
        print(spacing + '→ Left:')
        self.print_tree(node.left_branch, spacing + "  ", headers)

        # Call this function recursively on the false branch
        print(spacing + '→ Right:')
        self.print_tree(node.right_branch, spacing + "  ", headers)


class DecisionTree(Tree):

    def __init__(self, information_gain: any, max_depth: int = 3, leaf_style=None):
        super(DecisionTree, self).__init__(information_gain, max_depth, leaf_style)

    def build_tree(self, rows, level=0):
        gain, question = self.split(rows)
        if gain == 0 or level == self.max_depth:
            return Leaf(predictions=self.label_percentage(rows))

        # left branch contains rows satisfy the question, and right branch is opposite
        left_branch, right_branch = self.partition(rows, question)

        left_branch = self.build_tree(left_branch, level=level + 1)
        right_branch = self.build_tree(right_branch, level=level + 1)

        return Node(question, left_branch, right_branch)

    def split(self, rows):
        best_gain = 0.0
        best_question = None
        for feature in self.unique_values(rows):  # get unique values for each feature (column)
            col_idx, col_values = feature
            for col_val in col_values:
                question = Feature(col_idx, col_val)  # make a question bases on the current value
                # left branch contains rows satisfy the question, and right branch is opposite
                left_branch, right_branch = self.partition(rows, question)

                # if the question can't split data into left or right branches then skip
                if not left_branch or not right_branch:
                    continue

                # calculate information gain
                gain = self.information_gain(rows, left_branch, right_branch)

                if gain > best_gain:  # This decides which question will be selected
                    best_gain = gain
                    best_question = question

        return best_gain, best_question


class RandomTree(Tree):
    def __init__(self, information_gain: any, max_depth: int = 3, leaf_style=None):
        super(RandomTree, self).__init__(information_gain, max_depth, leaf_style)

    def build_tree(self, rows, level=0):
        question = self.split(rows)
        if question is None or level == self.max_depth:
            return self.leaf_style(predictions=self.label_percentage(rows))

        # left branch contains rows satisfy the question, and right branch is opposite
        left_branch, right_branch = self.partition(rows, question)

        if left_branch:
            left_branch = self.build_tree(left_branch, level=level+1)
        if right_branch:
            right_branch = self.build_tree(right_branch, level=level+1)

        return Node(question, left_branch, right_branch)

    def split(self, rows):
        question = None
        for feature in self.unique_values(rows):  # get unique values for each feature (column)
            col_idx, col_values = feature
            for col_val in col_values:
                question = Feature(col_idx, col_val)  # make a question bases on the current value
                # left branch contains rows satisfy the question, and right branch is opposite
                left_branch, right_branch = self.partition(rows, question)

                # if the question can't split data into left or right branches then skip
                if not left_branch or not right_branch:
                    continue
                # calculate information gain
                question = question

        return question


class RandomForest:
    def __init__(self, information_gain, max_tree=10, max_depth=3, tree_style: Tree = None, leaf_style=None):
        self.information_gain = information_gain
        self.max_tree = max_tree
        self.max_depth = max_depth
        self.tree_style = tree_style
        self.leaf_style = leaf_style
        self.bunch_of_tree = list()

    def build_forest(self, rows):
        for _ in range(0, self.max_tree):
            # make the tree instance and then build the real tree with given data
            inst_tree = self.tree_style(self.information_gain, self.max_depth, self.leaf_style)
            real_tree = inst_tree.build_tree(rows)
            self.bunch_of_tree.append((inst_tree, real_tree))

    def process(self, a_piece_of_data):
        result = {}
        for item in self.bunch_of_tree:
            inst_tree, real_tree = item
            label, perc = inst_tree.classify(a_piece_of_data, real_tree)
            result[label] = perc
        best_label = max(result, key=result.get)
        return best_label, result[best_label]
        # return result

import csv
from dataclasses import dataclass

#########################################################################
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):
        try:
            self.value = float(self.value)
            return float(row[self.column]) >= self.value
        except ValueError:
            return row[self.column].lower() == self.value.lower()
        return False

    def __repr__(self):
        try:
            self.value = float(self.value)
            return "%s >= %s " % (self.column, self.value)
        except ValueError:
            return "%s == %s " % (self.column, self.value)


@dataclass
class Node:
    question: Question
    left_branch: object
    right_branch: object


@dataclass
class Leaf:
    predictions: object

    def predict(self):
        print(self.predictions)
        best_label = max(self.predictions, key=self.predictions.get)
        return best_label, self.predictions[best_label]
#########################################################################

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predict()

    if node.question.match(row):
        return classify(row, node.left_branch)
    else:
        return classify(row, node.right_branch)


def build_tree(rows):
    gain, question = best_split(rows)
    if gain == 0:
        return Leaf(predictions=label_percentage(rows))
    
    left_branch, right_branch = partition(rows, question)

    left_branch = build_tree(left_branch)
    right_branch = build_tree(right_branch)

    return Node(question, left_branch, right_branch)


def best_split(rows):
    best_gain = 0.0
    best_question = None
    for feature in set_values(rows):
        col_idx, col_values = feature
        for col_val in col_values:
            question = Question(col_idx, col_val)
            left_branch, right_branch = partition(rows, question)

            if not left_branch or not right_branch:
                continue

            gain = gini_info_gain(rows, left_branch, right_branch)
            if gain >= best_gain:
                best_gain = gain
                best_question = question
    return best_gain, best_question


def partition(rows, question):
    left_branch, right_branch = [], []
    for row in rows:
        if question.match(row):
            left_branch.append(row)
        else:
            right_branch.append(row)
    return left_branch, right_branch


def set_values(rows):
    u_values = []
    row = rows[0]
    for col in range(len(row) - 1):
        values = set([row[col] for row in rows])
        u_values.append(values)
    return list(zip(range(len(rows)), u_values))
            

def gini_info_gain(parent_branch, left_branch, right_branch):
    prob_left = len(left_branch) / (len(left_branch) + len(right_branch))
    prob_right = len(right_branch) / (len(left_branch) + len(right_branch))

    gini_left = gini(left_branch)
    gini_right = gini(right_branch)

    # Information Gain = Entropy(parent) - ( (Probability(left) * Entropy(Left)) + Probability(right) * Entropy(Right) )
    gain = gini(parent_branch) - ( (prob_left*gini_left) + prob_right*gini_right )

    return round(gain, 3)


def gini(rows):
    counts = label_count(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / len(rows)
        impurity -= prob_of_label**2
    return round(impurity, 3)


def label_count(rows):
    count = {}
    for row in rows:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count


def label_percentage(rows):
    perc_label = {}
    count_label = label_count(rows)
    for label, count in count_label.items():
        perc_label[label] = count/len(count_label.keys())
    return perc_label




def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.left_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.right_branch, spacing + "  ")


if __name__ == "__main__":
    with open("decision_tree/fruits/data/fruits.csv", "r") as src:
        dataset = csv.reader(src, delimiter=",")
        dataset = list(dataset)[1:]

        detree = build_tree(dataset)
        result = classify(['Yellow', 3, 'Lemon'], detree)
        print(result)
        # 'Apple': 0.5, 'Lemon': 0.5} -> A tie!  Which one should pick
        # Which columns should be added to make more accuracy

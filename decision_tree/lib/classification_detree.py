from dataclasses import dataclass
#########################################################################
@dataclass
class Question:

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
    question: Question
    left_branch: object
    right_branch: object


@dataclass
class Leaf:
    predictions: dict

    def predict(self):
        # return the label with highest percentage
        best_label = max(self.predictions, key=self.predictions.get)
        return best_label, self.predictions[best_label]
#########################################################################

class ClassifyDeTree:

    def __init__(self):
        pass

    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.predict()

        if node.question.match(row):
            return self.classify(row, node.left_branch)
        else:
            return self.classify(row, node.right_branch)

    def build_tree(self, rows):
        gain, question = self.best_split(rows)
        if gain == 0:
            return Leaf(predictions=self.label_percentage(rows))

        left_branch, right_branch = self.partition(rows, question)

        left_branch = self.build_tree(left_branch)
        right_branch = self.build_tree(right_branch)

        return Node(question, left_branch, right_branch)

    def best_split(self, rows):
        best_gain = 0.0
        best_question = None
        for feature in self.unique_values(rows):
            col_idx, col_values = feature
            for col_val in col_values:
                question = Question(col_idx, col_val)
                left_branch, right_branch = self.partition(rows, question)

                if not left_branch or not right_branch:
                    continue

                gain = self.gini_info_gain(rows, left_branch, right_branch)

                if gain > best_gain:  # This decide which question will be selected
                    best_gain = gain
                    best_question = question

        return best_gain, best_question

    def partition(self, rows, question):
        left_branch, right_branch = [], []
        for row in rows:
            if question.match(row):
                left_branch.append(row)
            else:
                right_branch.append(row)
        return left_branch, right_branch

    def unique_values(self, rows):
        u_values = []
        row = rows[0]
        for col in range(len(row) - 1):
            values = set([row[col] for row in rows])
            u_values.append(values)
        return list(zip(range(len(rows)), u_values))

    def gini_info_gain(self, parent_branch, left_branch, right_branch):
        prob_left = len(left_branch) / (len(left_branch) + len(right_branch))
        prob_right = len(right_branch) / (len(left_branch) + len(right_branch))

        gini_left = self.gini_impurity(left_branch)
        gini_right = self.gini_impurity(right_branch)

        # Information Gain = Entropy(parent) - ( (Probability(left) * Entropy(Left)) + Probability(right) * Entropy(Right) )
        gain = self.gini_impurity(parent_branch) - ((prob_left * gini_left) + prob_right * gini_right)
        return round(gain, 3)

    def gini_impurity(self, rows):
        counts = self.label_count(rows)
        impurity = 1
        for label in counts:
            prob_of_label = counts[label] / len(rows)
            impurity -= prob_of_label ** 2
        return round(impurity, 3)

    def label_count(self, rows):
        count = {}
        for row in rows:
            label = row[-1]
            if label not in count:
                count[label] = 0
            count[label] += 1
        return count

    def label_percentage(self, rows):
        perc_label = {}
        count_label = self.label_count(rows)
        for label, count in count_label.items():
            perc_label[label] = count / sum([i for i in count_label.values()])
        return perc_label

    def print_tree(self, node, spacing="", headers=None):
        """World's most elegant tree printing function."""
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
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

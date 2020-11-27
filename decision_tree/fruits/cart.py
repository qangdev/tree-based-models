import csv
from decision_tree.lib.utils import gini_gain
from decision_tree.lib.classification_detree import ClassifyDecisionTree


if __name__ == "__main__":
    with open("./data/fruits.csv", "r") as src:
        print("STARTED")
        dataset = csv.reader(src, delimiter=",")
        dataset = list(dataset)[1:]

        classifier = ClassifyDecisionTree(information_gain=gini_gain)
        detree = classifier.build_tree(dataset)
        result = classifier.classify(['Yellow', 3], detree)
        print(result)
        print("DONE")

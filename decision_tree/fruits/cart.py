import csv
from decision_tree.lib.classification_detree import ClassifyDeTree


if __name__ == "__main__":
    with open("decision_tree/fruits/data/fruits.csv", "r") as src:
        print("STARTED")
        dataset = csv.reader(src, delimiter=",")
        dataset = list(dataset)[1:]

        classifier = ClassifyDeTree()
        detree = classifier.build_tree(dataset)
        result = classifier.classify(['Yellow', 3], detree)
        print(result)
        print("DONE")

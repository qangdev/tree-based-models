import csv
import pandas as pd
from dataclasses import dataclass
from decision_tree.lib.utils import gini_gain, entropy_gain
from decision_tree.lib.classification_detree import ClassifyDecisionTree


@dataclass()
class CarRecord:
    buying: str
    maint: str
    doors: int
    persons: int
    lug_boot: str
    safety: str
    klass: str = None

    def clean_data(self):
        if self.doors == "5more":
            self.doors = 5
        else:
            self.doors = int(self.doors)

        if self.persons == "more":
            self.persons = 6
        else:
            self.persons = int(self.persons)
        return self

    def to_list(self):
        return [self.buying, self.maint, self.doors, self.persons, self.lug_boot, self.safety, self.klass]



if __name__ == '__main__':
    print("S T A R T E D")
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

    known_label = []
    predicted_label = []
    with open("./data/car.data", "r+") as src:
        rows = []
        dataset = list(csv.reader(src, delimiter=","))
        for i in dataset:
            known_label.append(str(i[6]))
            rows.append(CarRecord(i[0], i[1], i[2], i[3], i[4], i[5], i[6]).clean_data().to_list())

        # make a classifier instance with information gain = gini | entropy
        classifier = ClassifyDecisionTree(information_gain=entropy_gain)
        tree = classifier.build_tree(rows)
        # classifier.print_tree(tree, headers=headers)

        for i in dataset:
            r = CarRecord(i[0], i[1], i[2], i[3], i[4], i[5]).clean_data().to_list()
            label, perc = classifier.classify(r, tree)
            predicted_label.append(label)

    df_prediction = pd.Series(predicted_label, name="Predicted")
    df_actual = pd.Series(known_label, name="Actual")
    df_confusion = pd.crosstab(df_prediction, df_actual)

    print("-" * 35)
    print(df_confusion)
    print("-"*35)

    print("D O N E")
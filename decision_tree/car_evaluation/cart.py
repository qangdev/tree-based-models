import csv
from dataclasses import dataclass
from decision_tree.lib.classification_detree import ClassifyDeTree


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

    def to_list(self):
        return [self.buying, self.maint, self.doors, self.persons, self.lug_boot, self.safety, self.klass]

if __name__ == '__main__':
    print("S T A R T E D")
    dataset = []
    with open("./data/car.data", "r+") as src:
        dataset = list(csv.reader(src, delimiter=","))
        rows = []
        for item in dataset:
            car_record = CarRecord(item[0],
                                   item[1],
                                   item[2],
                                   item[3],
                                   item[4],
                                   item[5],
                                   item[6])
            car_record.clean_data()
            rows.append(car_record.to_list())

    classifier = ClassifyDeTree()
    tree = classifier.build_tree(rows)
    # vhigh,vhigh,2,2,big,med,unacc
    # vhigh,low,2,2,small,low,unacc
    # low, med, 4, more, big, med, good
    # low, med, 4, more, big, high, vgood
    exam_car = CarRecord("low", "med", "4", "more", "big", "high")
    exam_car.clean_data()
    a_prediction = classifier.classify(exam_car.to_list(), tree)
    print(a_prediction)
    print("D O N E")
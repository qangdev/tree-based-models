import csv
from dataclasses import dataclass


@dataclass
class TennisRecord:
    day: int
    outlook: str
    humidity: str
    wind: str
    play: str = None


if __name__ == '__main__':
    print("[S T A R T E D]")

    def build_tree():
        YES = "Yes"
        NO = "No"

        l0 = {"v": YES}
        l1 = {"v": YES}
        l2 = {"v": NO}
        l3 = {"v": YES}
        l4 = {"v": NO}

        n2 = {"question": lambda val: val.wind == "Weak",
              "l": l3,
              "r": l4}

        n1 = {"question": lambda val: val.humidity == "Normal",
              "l": l1,
              "r": l2}

        n0 = {"question": lambda val: val.outlook == "Sunny",
              "l": n1,
              "r": n2}

        tree = {"question": lambda val: val.outlook == "Overcast",
                "l": l0,
                "r": n0}

        return tree

    def predict(node, val):
        if "v" in node:
            return node["v"]
        next_node = node["l"] if node["question"](val) else node["r"]
        return predict(next_node, val)

    tree = build_tree()
    with open("./data/tennis.csv", "r") as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        data = [r for i, r in enumerate(data) if i != 0]  # remove header

        known_target = [TennisRecord(int(r[0]), r[1], r[2], r[3], r[4]) for i, r in enumerate(data)]
        predict_target = []

        for r in data:
            record = TennisRecord(int(r[0]), r[1], r[2], r[3])
            record.play = predict(tree, record)
            predict_target.append(record)

        assert known_target == predict_target, "Ops, Something wrong"
    print("[D O N E]")
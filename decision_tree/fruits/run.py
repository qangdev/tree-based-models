import csv



if __name__ == "__main__":
    with open("./data/fruits.csv", "r") as src:
        dataset = csv.reader(src, delimiter=",")
        for row in dataset:
            print(row)

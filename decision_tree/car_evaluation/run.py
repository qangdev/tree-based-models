import math

# How Gini impurity works
chest_pain = {(105, 39), (34, 125)}


def gini(data):
    impurity = 1 - (data[0]/sum(data))**2 - (data[1]/sum(data))**2
    return round(impurity, 3)


def sum(a_list):
    return a_list.pop() if len(a_list) == 1 else a_list.pop() + sum(a_list)



if __name__ == '__main__':
    # impurity = gini(chest_pain["YES"])
    # assert impurity == 0.395, "Ops something went wrong"
    a_list = [1, 2, 3, 4, 5]
    total = sum(a_list)
    print(total)
    assert total == 15, "What the hell"

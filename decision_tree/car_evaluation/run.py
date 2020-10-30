import csv



def features_impurity(features, i=0):
    return [feature_gini(features[i])] if (i+1) == len(features) else [feature_gini(features[i])] + features_impurity(features, i+1)


def feature_gini(feature):
    # TODO: X * x + Y * y What is X, x, Y, y
    x = gini(feature[0])
    y = gini(feature[1])
    X = sum(feature[0])/sum(feature[0]+feature[1])
    Y = sum(feature[1])/sum(feature[0]+feature[1])
    return round((X*x) + (Y*y), 3)


def gini(data):
    return round(1 - (data[0]/sum(data))**2 - (data[1]/sum(data))**2, 3)


def get_best_feature(features, impurites):
    # Return the index of the minimum value in the impurity list
    bindex = impurites.index(min(impurites))
    return features[bindex], impurites[bindex]


def counting_orabettername(col_index, target_label, data, i=0):
    print(col_index, target_label, data, i)
    val = 1 if data[i][] == target_label and i <= len(data) else 0
    return val if (i+1) == len(data) else val + counting_orabettername(col_index, target_label, data, i+1)



if __name__ == '__main__':
    # S T E P  0: Import dataset
    dataset = []
    with open("./data/car.data", "r+") as src:
        dataset = list(csv.reader(src, delimiter=","))
    print(dataset[:2])

    # S T E P  1: Look at data and make question
    


    # S T E P  2: Massage the data to be ready
    label = "safety"
    label_values = ["acc", "unacc"]
    features = ["buying", "maint", "doors", "persons", "lug_boot"]
    features_values = []

    # features_values.append(counting_orabettername(0, "acc", dataset[:2]))
    features_values.append(counting_orabettername(0, "unacc", dataset[:2]))
    print(features_values)


    # S T E P  3: Calculating Gini Impurity


    # chest_pain = [[105, 39], [34, 125]]
    # good_bcir = [[37, 127], [100, 33]]
    # blocked_ar = [[92, 31], [45, 129]]

    # features = ["check_pain", "good_blood_circulation", "blocked_arterise"]
    # f_data = [chest_pain, good_bcir, blocked_ar]

    # imputiry = features_impurity(f_data)
    # print(imputiry, f_data)
    # assert imputiry == [0.364, 0.360, 0.381], "Impurity wrong"
    # b_feature, b_index = get_best_feature(features, imputiry)
    # print("Best Feature:> %s with impurity is %s" % (b_feature, b_index))

import math


def gini_gain(parent_branch, left_branch, right_branch):
    prob_left = len(left_branch) / (len(left_branch) + len(right_branch))
    prob_right = len(right_branch) / (len(left_branch) + len(right_branch))

    gini_left = gini_index(left_branch)
    gini_right = gini_index(right_branch)

    # Information Gain = Entropy(parent) - ( (Probability(left) * Entropy(Left)) + Probability(right) * Entropy(Right) )
    gain = gini_index(parent_branch) - ((prob_left * gini_left) + prob_right * gini_right)
    return round(gain, 3)


def gini_index(rows):
    counts = label_count(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / len(rows)
        impurity -= prob_of_label ** 2
    return round(impurity, 3)


def entropy_gain(parent_branch, left_branch, right_branch):
    prob_left = len(left_branch) / (len(left_branch) + len(right_branch))
    prob_right = len(right_branch) / (len(left_branch) + len(right_branch))

    entropy_left = entropy_impurity(left_branch)
    entropy_right = entropy_impurity(right_branch)

    # Information Gain = Entropy(parent) - ( (Probability(left) * Entropy(Left)) + Probability(right) * Entropy(Right) )
    gain = entropy_impurity(parent_branch) - ((prob_left * entropy_left) + prob_right * entropy_right)
    return round(gain, 3)


def entropy_impurity(rows):
    entropy = 0
    counts = label_count(rows)
    for label in counts:
        prob_of_label = counts[label] / len(rows)
        entropy += -(prob_of_label) * math.log2(prob_of_label)
    return round(entropy, 3)


def label_count(rows):
    count = {}
    for row in rows:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count

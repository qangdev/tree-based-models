
# Step 1: Gini index
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Step 2: Spliting

# Step 3: Build a Tree

# Step 4: Make a Prediction

# Step 5: Make multiple Prediction

if __name__ == '__main__':
    print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
    print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
    # Load banknote data set.

    # Clean data

    # Run steps

    pass
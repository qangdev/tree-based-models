import unittest
from decision_tree.lib.utils import entropy_impurity, gini_index

class ImpurityTestCase(unittest.TestCase):


    def test_entropy_calculating(self):
        rows = ["red", "red", "red", "red", "red", "blue", "blue", "blue"]
        entropy = entropy_impurity(rows)
        self.assertEqual(0.954, entropy)


    def test_gini_index_calculating(self):
        rows = ["red", "red", "red", "red", "red", "blue", "blue", "blue"]
        gini_impurity = gini_index(rows)
        self.assertEqual(0.469, gini_impurity)

if __name__ == '__main__':
    unittest.main()

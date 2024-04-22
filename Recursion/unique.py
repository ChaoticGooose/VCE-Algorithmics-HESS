import unittest

"""
list -> int
For a given list of elems, return the number of unique permutations of the list.

def unique_permutations(arr: list) -> int:
    return 0
"""


def unique_permutations(arr: list) -> int:
    def factorial(m: int) -> int:
        if m == 0: # Base case
            return 1 # 0! = 1
        return m * factorial(m - 1) # Recurse down to the largest known value (0) then use results to calculate the factorial of m

    length = len(arr) # Get the length of the list
    return factorial(length) # Return the factorial of the length of the list

class TestUniquePermutations(unittest.TestCase):
    def test_1(self):
        self.assertEqual(unique_permutations([1, 2, 3]), 6)

    def test_2(self):
        self.assertEqual(unique_permutations([True, False]), 2)

    def test_3(self):
        self.assertEqual(unique_permutations(["a", "b", "c"]), 6)

def main():
    unittest.main()

main()

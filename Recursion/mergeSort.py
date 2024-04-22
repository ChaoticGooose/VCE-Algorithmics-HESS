import unittest
import random

random.seed(0)

"""
List -> List
Return a sorted version of the input list

def merge_sort(arr):
    return []
"""
def merge_sort(arr: list) -> list:

    def merge(left: list, right: list) -> list:
        result = list()
        left_len = len(left)
        right_len = len(right)
        left_sel = 0
        right_sel = 0

        # Merge the two lists, selecting the smallest element at each step
        while (left_sel < left_len) and (right_sel < right_len):
            if left[left_sel] < right[right_sel]:
                result.append(left[left_sel])
                left_sel += 1
            else:
                result.append(right[right_sel])
                right_sel += 1

        # Add the remaining elements
        while (left_sel < left_len):
            result.append(left[left_sel])
            left_sel += 1
        while (right_sel < right_len):
            result.append(right[right_sel])
            right_sel += 1
        
        return result

    if len(arr) <= 1:
        return arr
    
    # Recursively split the list in half, then merge the results
    split = len(arr) // 2
    left_arr = merge_sort(arr[:split])
    right_arr = merge_sort(arr[split:])

    return merge(left_arr, right_arr)
    

# Unit tests
# Dies of cringe
class TestQuickSort(unittest.TestCase):

    def test_short(self):
        # Short list
        self.assertEqual(merge_sort([1, 3, 2, 4, 5, 6]), [1, 2, 3, 4, 5, 6])

    def test_long(self):
        # Long list
        arr = [random.randint(0, 1000) for _ in range(1000)]
        self.assertEqual(merge_sort(arr), sorted(arr))

    def test_edge(self):
        # Sorted list
        self.assertEqual(merge_sort([1, 2, 3, 4, 5, 6, 7]), [1, 2, 3, 4, 5, 6, 7])
        
    def test_empty(self):
        # Empty list
        self.assertEqual(merge_sort([]), [])


def main():
    unittest.main()
main()

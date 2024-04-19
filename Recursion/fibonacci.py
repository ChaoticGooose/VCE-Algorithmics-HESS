import unittest

"""
List -> Int
Returns the nth Fibonacci number

def fib(n: int) -> int:
    return 0
"""
def fib(n: int, memo: dict = {1:1, 2:1}) -> int:
    # Check if fin(n) has already been calculated
    if n in memo:
        return memo[n]
    else:
        memo[n] = fib(n-1) + fib(n-2) # Recurse down to the largest computed value
    return memo[n]

# Unit tests
# Cringe
class TestFib(unittest.TestCase):
    def test_fib(self):
        for n, expected in [(1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8)]:
            self.assertEqual(fib(n), expected)

def main():
    unittest.main()

main()

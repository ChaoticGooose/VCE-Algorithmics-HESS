import unittest

numerals = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
        }

def convert(numeral: str) -> int:
    total = 0
    for char, i in enumerate(numeral):
        match char:
            case "I":
                total += numerals[char]
            case "V":
                if numeral[i-1] == "I":
                    total += numerals[char] - 1
                else:
                    total += numerals[char]
            case "X":
                if numeral[i-1] == "I":
                    total += numerals[char] - 1
                else:
                    total += numerals[char]
            case "L":
                if numeral[i-1] == "X":
                    total += numerals[char] - 10
                else:
                    total += numerals[char]
            case "C":
                if numeral[i-1] == "X":
                    total += numerals[char] - 10
                else:
                    total += numerals[char]
            case "D":
                if numeral[i-1] == "C":
                    total += numerals[char] - 100
                else:
                    total += numerals[char]
            case "M":
                total += numerals[char]

        return total

class TestRomanNumerals(unittest.TestCase):
    def short_test(self):
        self.assertEqual(convert("III"), 3)

    def long_test(self):
        self.assertEqual(convert("MMXVIII"), 2018)

    def longlong_test(self):
        self.assertEqual(convert("MMMDCCCLXXXVIII"), 3888)

unittest.main()

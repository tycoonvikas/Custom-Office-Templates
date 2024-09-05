import unittest
from parameterized import parameterized

def multiply(x, y):
    return x * y

class TestMultiplyFunction(unittest.TestCase):
    @parameterized.expand([
        ("positive_numbers", 2, 3, 6),
        ("negative_numbers", -2, 3, -6),
        ("positive_and_negative", 2, -3, -6),
        ("with_zero", 0, 3, 0),
    ])
    def test_multiply(self, name, x, y, expected):
        self.assertEqual(multiply(x, y), expected)

if __name__ == "__main__":
    unittest.main()

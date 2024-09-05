import inc_dec
import unittest
import string    # The code to test  # The test framework

class Test_TestIncrementDecrement(unittest.TestCase):
    def test_increment(self):
        self.assertEqual(inc_dec.increment(3), 4)

    # This test is designed to fail for demonstration purposes.
    def test_decrement(self):
        self.assertEqual(inc_dec.decrement(3), 2)
    
    def test_concatenate_strings():
        self.assertEqual(concatenate_strings("hello", "world"), "hello world")
        # Test concatenation of two regular words
        assert concatenate_strings("hello", "world") == "hello world"
    
    # Test concatenation of two other words
        assert concatenate_strings("foo", "bar") == "foo bar"
    
    # Test concatenation where the second string is empty
        assert concatenate_strings("nonempty", "") == "nonempty "
    
    # Test concatenation where both strings are non-empty and identical
        assert concatenate_strings("empty", "empty") == "empty empty"
    
    # Test concatenation where both strings are empty
        assert concatenate_strings("", "") == " "
    
    def test_frequency_of_string():
        assert frequency_of_string(["apple", "banana", "apple"], "apple") == 2
        assert frequency_of_string(["dog", "cat", "dog", "dog"], "dog") == 3
    
    # Test frequency of "a" in a list containing "a" once
        assert frequency_of_string(["a", "b", "c", "d"], "a") == 1
    
    # Test frequency of "test" in a list containing "test" once
        assert frequency_of_string(["test"], "test") == 1
    
    # Test frequency of a string in an empty list
        assert frequency_of_string([], "nothing") == 0
        

if __name__ == '__main__':
    unittest.main()
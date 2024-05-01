import collections
import itertools
from typing import Generator, List


class Solution:
    # 6. Zigzag Conversion
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        rows = [[] for _ in range(numRows)]

        direction = 1  # 1: down, -1: up
        curr_row = 0
        for c in s:
            rows[curr_row].append(c)
            curr_row += direction
            if curr_row == 0 or curr_row == numRows - 1:
                direction *= -1

        return "".join(itertools.chain.from_iterable(rows))

    # 8. String to Integer (atoi)
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if not s:
            return 0

        int_min = -(2**31)
        int_max = 2**31 - 1

        sign = -1 if s[0] == "-" else 1
        if s[0] in {"-", "+"}:
            s = s[1:]

        result = 0
        for c in s:
            if not c.isdigit():
                break
            result = result * 10 + ord(c) - ord("0")
            if sign * result < int_min:
                return int_min
            if sign * result > int_max:
                return int_max

        return sign * result

    # 58. Length of Last Word
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.strip().split(" ")[-1])

    # 214. Shortest Palindrome
    def shortestPalindrome(self, s: str) -> str:
        # reversed string for checking
        r = s[::-1]

        # find the longest palindromic substring
        for i in range(len(s)):
            if s.startswith(r[i:]):
                return r[:i] + s

        # worst case: add the reverse with original
        # -> automatically a palindrome
        return r + s

    # 796. Rotate String
    def rotateString(self, s: str, goal: str) -> bool:
        return len(s) == len(goal) and goal in s + s

    # 1422. Maximum Score After Splitting a String
    def maxScore(self, s: str) -> int:
        left = 0
        right = s.count("1")
        max_score = 0
        for c in s:
            if c == "0":
                left += 1
            else:
                right -= 1
            max_score = max(max_score, left + right)
        return max_score

    # 1614. Maximum Nesting Depth of the Parentheses
    def maxDepth(self, s: str) -> int:
        result = 0

        curr_depth = 0
        for c in s:
            if c == "(":
                curr_depth += 1
            elif c == ")":
                curr_depth -= 1
            result = max(result, curr_depth)

        return result

    # 1662. Check If Two String Arrays are Equivalent
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        def char_generator(word: List[str]) -> Generator:
            for w in word:
                for c in w:
                    yield c
            yield None

        return all(
            c1 == c2 for c1, c2 in zip(char_generator(word1), char_generator(word2))
        )

    # 1704. Determine if String Halves Are Alike
    def halvesAreAlike(self, s: str) -> bool:
        # # Straightforward counting (HashMap)
        # vowels = 'aeiouAEIOU'
        # half = len(s) // 2
        #
        # left_half = collections.Counter(s[:half])
        # right_half = collections.Counter(s[half:])
        #
        # vowels_left = sum(
        #     count for count in (left_half[vowel] for vowel in vowels)
        # )
        # vowels_right = sum(
        #     count for count in (right_half[vowel] for vowel in vowels)
        # )
        #
        # return vowels_left == vowels_right

        # Two pointers
        length = len(s)
        left, right = 0, length - 1
        vowels_left = vowels_right = 0

        # Clever vowel check: (0x208222 >> (char & 0x1f)) & 1
        # - (char & 0x1f) 'isolates' the first 5 bits of the character.
        #   Regardless of whether the character is uppercase or not,
        #   the resulting number will be in the range 1-26, which is the
        #   position of the character in the alphabet.
        # - (0x208222) is a 24-bit number that holds a pattern of 1's in
        #   specific positions that correspond to vowel positions when shifted
        #   right. Binary representation: 001000001000001000100010
        # - (0x208222 >> (char & 0x1f)) shifts the special number right by
        #   the position of the character computed earlier.
        # - (...) & 1 is the final vowel check, 'isolating' the rightmost bit.
        #   If the result is 1, this is a vowel, otherwise it is not.
        #
        # For example, for letter 'e', which is a vowel, 0x208222 will be
        # shifted right by 5 places, which results in a number with its
        # rightmost bit being 1 -> Vowel check successful
        while left < right:
            if (0x208222 >> (ord(s[left]) & 0x1F)) & 1:
                vowels_left += 1
            if (0x208222 >> (ord(s[right]) & 0x1F)) & 1:
                vowels_right += 1
            left += 1
            right -= 1

        return vowels_left == vowels_right

    # 1758. Minimum Changes To Make Alternating Binary String
    def minOperations(self, s: str) -> int:
        # In an alternating binary string that starts with '0' ('010101'),
        # numbers at even indices must be '0', and odd indices must be '1'.
        # We count how many times this rule got violated to get the number
        # of changes needed to construct the string.
        #
        # For alternate binary string that starts with '1' ('101010'),
        # the number of changes needed can be calculated using the same
        # method but with reversed rules. As such, the result is
        # len(s) - count_start_zero

        # count_start_zero = list(
        #     int(c) != i % 2 for i, c in enumerate(s)
        # ).count(True)
        count_start_zero = sum(int(c) != i % 2 for i, c in enumerate(s))
        count_start_one = len(s) - count_start_zero
        return min(count_start_zero, count_start_one)

    # 2000. Reverse Prefix of Word
    def reversePrefix(self, word: str, ch: str) -> str:
        index = word.find(ch)
        return (
            word[: index + 1][::-1] + word[index + 1 :]
            if 0 <= index < len(word)
            else word
        )

    # 2108. Find First Palindromic String in the Array
    def firstPalindrome(self, words: List[str]) -> str:
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]

        for word in words:
            if is_palindrome(word):
                return word
        return ""

    # 2264. Largest 3-Same-Digit Number in String
    def largestGoodInteger(self, num: str) -> str:
        return max(
            num[i - 2 : i + 1] if num[i - 2] == num[i - 1] == num[i] else ""
            for i in range(2, len(num))
        )

    # 2864. Maximum Odd Binary Number
    def maximumOddBinaryNumber(self, s: str) -> str:
        count = collections.Counter(s)
        return (
            "0" * count["0"]
            if not count["1"]
            else "1" * (count["1"] - 1) + "0" * count["0"] + "1"
        )

import collections
import itertools
import math
from typing import List


def ceil_div(x, y):
    return -(x // -y)


class ListNode:
    def __init__(self, val=0, n=None):
        self.val = val
        self.next = n


class Solution:
    # 9. Palindrome Number
    def isPalindrome(self, x: int) -> bool:
        # # Without converting to string
        # if x < 0:
        #     return False
        #
        # reverse = 0
        # y = x
        # while y:
        #     reverse = reverse * 10 + y % 10
        #     y //= 10
        #
        # return reverse == x

        # Converting to string
        return str(x)[::-1] == str(x)

    # 661. Image Smoother
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        h, w = len(img), len(img[0])
        smoothed_img = [[0] * w for _ in range(h)]

        for y, i in enumerate(img):
            for x, j in enumerate(i):
                summ = 0
                count = 0
                for k in range(max(0, y - 1), min(h, y + 2)):
                    for l in range(max(0, x - 1), min(w, x + 2)):
                        summ += img[k][l]
                        count += 1
                smoothed_img[y][x] = summ // count

        return smoothed_img

    # 796. Rotate String
    def rotateString(self, s: str, goal: str) -> bool:
        return len(s) == len(goal) and goal in s + s

    # 1422. Maximum Score After Splitting a String
    def maxScore(self, s: str) -> int:
        left = 0
        right = s.count('1')
        max_score = 0
        for c in s:
            if c == '0':
                left += 1
            else:
                right -= 1
            max_score = max(max_score, left + right)
        return max_score

    # 1496. Path Crossing
    def isPathCrossing(self, path: str) -> bool:
        x, y = 0, 0
        visited = {(x, y)}

        for p in path:
            if p == 'N':
                y += 1
            elif p == 'E':
                x += 1
            elif p == 'S':
                y -= 1
            elif p == 'W':
                x -= 1

            if (x, y) in visited:
                return True
            else:
                visited.add((x, y))

        return False

    # 1582. Special Positions in a Binary Matrix
    def num_special(self, mat: List[List[int]]) -> int:
        result = 0
        special_rows = []

        for i in range(len(mat)):
            curr_row = mat[i]
            one_count = curr_row.count(1)
            if one_count == 1:
                j = curr_row.index(1)
                special_rows.append([i, j])

        for x, y in special_rows:
            is_special_column = True
            for i in range(len(mat[0])):
                if i != x and mat[i][y] == 1:
                    is_special_column = False
                    break

            if is_special_column:
                result += 1

        return result

    # 1716. Calculate Money in Leetcode Bank
    def totalMoney(self, n: int) -> int:
        # # Simulation
        # day = day_amount = 1
        # total = 0
        # for _ in range(n):
        #     total += day_amount
        #     day += 1
        #     day_amount += 1
        #     if day % 7 == 0:
        #         day_amount -= 6  # -7 + 1
        # return total

        # Math
        weeks_num = n // 7
        days_left = n % 7
        return 28 * weeks_num + (7 * weeks_num * (weeks_num - 1)) // 2 + \
            (days_left * weeks_num + (days_left * (days_left + 1)) // 2)

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

    # 2706. Buy Two Chocolates
    def buyChoco(self, prices: List[int], money: int) -> int:
        min1, min2 = math.inf, math.inf

        for price in prices:
            if price < min1:
                min2 = min1
                min1 = price
            elif price < min2:
                min2 = price

        min_cost = min1 + min2
        return money - min_cost if min_cost <= money else money

    # 1897. Redistribute Characters to Make All Strings Equal
    def makeEqual(self, words: List[str]) -> bool:
        length = len(words)
        return all(
            count % length == 0
            for count in collections.Counter(
                itertools.chain.from_iterable(words)
            ).values()
        )

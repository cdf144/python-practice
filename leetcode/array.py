import math
from itertools import starmap, pairwise
from operator import sub
from typing import List


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

    # 238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # # Extra space
        # length = len(nums)
        # left_products = [1] * length
        # right_products = [1] * length
        #
        # for i in range(1, length):
        #     left_products[i] = left_products[i-1] * nums[i-1]
        # for i in range(length - 2, -1, -1):
        #     right_products[i] = right_products[i+1] * nums[i+1]
        #
        # result = [0] * length
        # for i in range(length):
        #     result[i] = left_products[i] * right_products[i]
        #
        # return result

        # Constant space
        length = len(nums)
        result = [1] * length

        for i in range(1, length):
            result[i] = result[i - 1] * nums[i - 1]

        curr_right_product = 1
        for i in range(length - 2, -1, -1):
            curr_right_product *= nums[i + 1]
            result[i] *= curr_right_product

        return result

    # 274. H-Index
    def hIndex(self, citations: List[int]) -> int:
        max_h = len(citations)
        count_citation_num = [0] * (max_h + 1)

        for citation in citations:
            count_citation_num[min(citation, max_h)] += 1

        accumulative_citation = 0
        for i, citation_count in reversed(
                list(enumerate(count_citation_num))):
            accumulative_citation += citation_count
            if accumulative_citation >= i:
                return i

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

    # 1637. Widest Vertical Area Between Two Points Containing No Points
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        # # Sorted set
        # x_set = SortedSet()
        # for point in points:
        #     x_set.add(point[0])
        #
        # max_diff = 0
        # prev = x_set[0]
        # for x in x_set:
        #     max_diff = max(max_diff, x - prev)
        #     prev = x
        #
        # return max_diff

        # Starmap with tuple of adjacent pairs in sorted list
        return -min(
            starmap(
                sub,
                pairwise(
                    sorted(x for x, y in points)
                )
            ),
            default=0
        )

    # 2125. Number of Laser Beams in a Bank
    def numberOfBeams(self, bank: List[str]) -> int:
        above, below = -1, -1
        result = 0

        for row in bank:
            if above == -1:
                above = row.count('1')
                continue

            below = row.count('1')
            if below == 0:
                continue

            result += above * below
            above = below

        return result

    # 2482. Difference Between Ones and Zeros in Row and Column
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])

        ones_row = [row.count(1) for row in grid]
        ones_col = [col.count(1) for col in zip(*grid)]

        diff = [[0 for x in range(w)] for y in range(h)]
        for i in range(h):
            for j in range(w):
                diff[i][j] = 2 * ones_row[i] + 2 * ones_col[j] - w - h

        return diff

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

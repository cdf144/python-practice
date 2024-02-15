import collections
import itertools
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

    # 1074. Number of Submatrices That Sum to Target
    def numSubmatrixSumTarget(self, matrix: List[List[int]],
                              target: int) -> int:
        # Basically a 2-D version of this problem:
        # https://www.geeksforgeeks.org/find-subarray-with-given-sum-in-array-of-integers/
        m = len(matrix)
        n = len(matrix[0])

        # Since we are working with a matrix, we have to "convert" the problem
        # into a 1-D array with prefix-sum technique
        for row in matrix:
            for col in range(1, n):
                row[col] += row[col - 1]

        # Now we choose 2 columns from n columns. For each combination of 2
        # columns (y1 and y2), we sum up the elements in the sub-matrices
        # (0, y1, x2, y2) where x2 goes in range(0, m). Now we can use the
        # prefix sum HashMap technique to find out the matrices that sum to
        # `target` the same way we solve the 1-D problem.
        result = 0
        for start_col in range(n):
            for end_col in range(start_col, n):
                prefix_count = collections.Counter({0: 1})
                sub_matrix_sum = 0
                for row in range(m):
                    if start_col > 0:
                        sub_matrix_sum -= matrix[row][start_col - 1]
                    sub_matrix_sum += matrix[row][end_col]
                    result += prefix_count[sub_matrix_sum - target]
                    prefix_count[sub_matrix_sum] += 1

        return result

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

    # 1630. Arithmetic Subarrays
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int],
                                 r: List[int]) -> List[bool]:
        def is_arithmetic(sequence: List[int]) -> bool:
            n = len(sequence)
            if n <= 2:
                return True

            sequence.sort()
            diff = sequence[1] - sequence[0]
            for i in range(2, n):
                if sequence[i] - sequence[i - 1] != diff:
                    return False
            return True

        result = []
        for query, left in enumerate(l):
            right = r[query]
            result.append(is_arithmetic(nums[left: right + 1]))
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

    # 1887. Reduction Operations to Make the Array Elements Equal
    def reductionOperations(self, nums: List[int]) -> int:
        # # Counting sort, O(n) time, O(n) space
        # buckets = [0] * 50001  # lazy counting

        # for num in nums:
        #     buckets[num] += 1

        # count = []
        # for c in buckets:
        #     if c != 0:
        #         count.append(c)

        # result = 0
        # for i in range(len(count) - 1, 0, -1):
        #     result += count[i]
        #     count[i - 1] += count[i]

        # return result

        # Sort, O(n*log(n)) time, O(1) space
        nums.sort()
        result = 0

        for i in range(len(nums) - 2, -1, -1):
            if nums[i] != nums[i + 1]:
                result += len(nums) - 1 - i

        return result

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

    # 2391. Minimum Amount of Time to Collect Garbage
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        # Prefix sum of total travel cost to index i
        prefix = [0] + list(itertools.accumulate(travel))
        total_garbage = len(''.join(garbage))

        m_idx, p_idx, g_idx = 0, 0, 0
        for i, house in enumerate(garbage):
            m_idx = m_idx if 'M' not in house else i
            p_idx = p_idx if 'P' not in house else i
            g_idx = g_idx if 'G' not in house else i

        return total_garbage + prefix[m_idx] + prefix[p_idx] + prefix[g_idx]

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

    # 2966. Divide Array Into Arrays With Max Difference
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        nums.sort()
        result = []

        for i in range(2, len(nums), 3):
            if nums[i] - nums[i - 2] > k:
                return []
            result.append([nums[i - 2], nums[i - 1], nums[i]])

        return result

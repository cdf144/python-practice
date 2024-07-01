import collections
import itertools
import math
from itertools import pairwise, starmap
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

    # 41. First Missing Positive
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)

        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

        for i, num in enumerate(nums):
            if num != i + 1:
                return i + 1

        return n + 1

    # 274. H-Index
    def hIndex(self, citations: List[int]) -> int:
        max_h = len(citations)
        count_citation_num = [0] * (max_h + 1)

        for citation in citations:
            count_citation_num[min(citation, max_h)] += 1

        accumulative_citation = 0
        for i, citation_count in reversed(list(enumerate(count_citation_num))):
            accumulative_citation += citation_count
            if accumulative_citation >= i:
                return i
        return 0

    # 442. Find All Duplicates in an Array
    def findDuplicates(self, nums: List[int]) -> List[int]:
        result = []

        for num in nums:
            idx = abs(num) - 1
            if nums[idx] < 0:
                result.append(abs(num))
            nums[idx] *= -1

        return result

    # 448. Find All Numbers Disappeared in an Array
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for num in nums:
            idx = abs(num) - 1
            if nums[idx] < 0:
                continue
            nums[idx] *= -1

        result = []
        for i, num in enumerate(nums):
            if num > 0:
                result.append(i + 1)

        return result

    # 463. Island Perimeter
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        result = 0

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == 1:
                    result += 4
                    if i > 0 and grid[i - 1][j] == 1:
                        result -= 2
                    if j > 0 and grid[i][j - 1] == 1:
                        result -= 2

        return result

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
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
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

    # 1550. Three Consecutive Odds
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        count_odd = 0
        for num in arr:
            count_odd = count_odd + 1 if num & 1 else 0
            if count_odd == 3:
                return True
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

    # 1630. Arithmetic Subarrays
    def checkArithmeticSubarrays(
        self, nums: List[int], l: List[int], r: List[int]
    ) -> List[bool]:
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
            result.append(is_arithmetic(nums[left : right + 1]))
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
        return -min(starmap(sub, pairwise(sorted(x for x, _ in points))), default=0)

    # 2125. Number of Laser Beams in a Bank
    def numberOfBeams(self, bank: List[str]) -> int:
        above, below = -1, -1
        result = 0

        for row in bank:
            if above == -1:
                above = row.count("1")
                continue

            below = row.count("1")
            if below == 0:
                continue

            result += above * below
            above = below

        return result

    # 2373. Largest Local Values in a Matrix
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        def find_max(y: int, x: int) -> int:
            """Find maximum in 3x3 matrix with grid[y][x] as the center."""
            m = 0
            for i in range(y - 1, y + 2):
                for j in range(x - 1, x + 2):
                    m = max(m, grid[i][j])
            return m

        n = len(grid)
        max_local = [[0] * (n - 2) for _ in range(n - 2)]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                max_local[i - 1][j - 1] = find_max(i, j)
        return max_local

    # 2391. Minimum Amount of Time to Collect Garbage
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        # Prefix sum of total travel cost to index i
        prefix = [0] + list(itertools.accumulate(travel))
        total_garbage = len("".join(garbage))

        m_idx, p_idx, g_idx = 0, 0, 0
        for i, house in enumerate(garbage):
            m_idx = m_idx if "M" not in house else i
            p_idx = p_idx if "P" not in house else i
            g_idx = g_idx if "G" not in house else i

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

        assert isinstance(min1, int) and isinstance(min2, int)
        min_cost = min1 + min2
        return money - min_cost if min_cost <= money else money

    # 2938. Separate Black and White Balls
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        result = 0
        count_zero = 0

        for i in reversed(range(n)):
            if s[i] == "1":
                result += count_zero
            else:
                count_zero += 1

        return result

    # 3011. Find if Array Can Be Sorted
    def canSortArray(self, nums: List[int]) -> bool:
        prev_max = curr_max = 0
        curr_min = 256 + 1
        curr_set_bit = 0

        for num in nums + [0]:
            set_bit = num.bit_count()
            if set_bit == curr_set_bit:
                curr_max = max(curr_max, num)
                curr_min = min(curr_min, num)
            else:
                if curr_min < prev_max:
                    return False
                prev_max = curr_max
                curr_max = curr_min = num
                curr_set_bit = set_bit

        return True

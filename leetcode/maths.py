import math
from typing import List


class Solution:
    # 7. Reverse Integer
    def reverse(self, x: int) -> int:
        sign = 1 if x >= 0 else -1
        x *= sign
        reverse = 0

        while x:
            reverse = reverse * 10 + x % 10
            x //= 10

        return sign * reverse

    # 50. Pow(x,n)
    def my_pow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.my_pow(x, -n)
        if n & 1:
            return x * self.my_pow(x, n - 1)
        return self.my_pow(x * x, n // 2)

    # 54. Spiral Matrix
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # # Shrinking walls
        # top = left = 0
        # bottom = len(matrix) - 1
        # right = len(matrix[0]) - 1
        #
        # result = []
        # while True:
        #     for x in range(left, right + 1):
        #         result.append(matrix[top][x])
        #     top += 1
        #     if top > bottom:
        #         break
        #
        #     for y in range(top, bottom + 1):
        #         result.append(matrix[y][right])
        #     right -= 1
        #     if right < left:
        #         break
        #
        #     for x in range(right, left - 1, -1):
        #         result.append(matrix[bottom][x])
        #     bottom -= 1
        #     if bottom < top:
        #         break
        #
        #     for y in range(bottom, top - 1, -1):
        #         result.append(matrix[y][left])
        #     left += 1
        #     if left > right:
        #         break
        #
        # return result

        # Switching directions
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        curr_dir = 0
        m = len(matrix)
        n = len(matrix[0])

        x = y = 0
        result = []
        while len(result) < m * n:
            result.append(matrix[y][x])

            check_x, check_y = x + dirs[curr_dir][1], y + dirs[curr_dir][0]
            if (
                check_x < 0
                or check_x >= n
                or check_y < 0
                or check_y >= m
                or matrix[check_y][check_x] == -101
            ):
                curr_dir = (curr_dir + 1) % 4

            matrix[y][x] = -101
            x += dirs[curr_dir][1]
            y += dirs[curr_dir][0]

        return result

    # 66. Plus One
    def plusOne(self, digits: List[int]) -> List[int]:
        for i, d in reversed(list(enumerate(digits))):
            if d < 9:
                digits[i] += 1
                return digits
            digits[i] = 0
        return [1] + digits

    # 73. Set Matrix Zeroes
    def setZeroes(self, matrix: List[List[int]]) -> None:
        # # Straightforward O(mn) space
        # m = len(matrix)
        # n = len(matrix[0])
        # zeroes = []
        # for i, row in enumerate(matrix):
        #     for j, val in enumerate(row):
        #         if val == 0:
        #             zeroes.append((i, j))
        #
        # for i, j in zeroes:
        #     matrix[i] = [0] * n
        #     for k in range(m):
        #         matrix[k][j] = 0

        # Constant space
        # We are going to use the first row and first column as our marker to
        # indicate that certain rows and columns must be filled with 0.
        # As such, we have to figure out if the first row and column needs to
        # be filled beforehand, because when we update the matrix later, there
        # might be discrepancies.
        m = len(matrix)
        n = len(matrix[0])
        fill_first_row = 0 in matrix[0]
        fill_first_col = 0 in list(zip(*matrix))[0]

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if fill_first_row:
            matrix[0] = [0] * len(matrix[0])

        if fill_first_col:
            for row in matrix:
                row[0] = 0

    # 202. Happy Number
    def isHappy(self, n: int) -> bool:
        # When we do the computation over and over again, intuitively there
        # can be 3 cases for the process:
        # - Ends with 1
        # - Loops infinitely (a cycle)
        # - The number approaches infinity
        # Of the 3 cases, it can be proven that case 3 can never happen:
        #
        # Suppose in the process, we run into a number N that is M digits
        # long (M >= 3).
        # Since we are dealing with base 10 numbers, the greatest value of any
        # digit in a number is '9', so suppose the M-digit number we ran into
        # is full of '9's, in other words, the greatest possible M-digit number.
        # To prove that we cannot approach infinity when we run the process,
        # we prove that N is the upper bound for any M-digit number that runs
        # through the process.
        # It can be seen that:
        #   compute(N) = 9^2 * M < 100 * M
        # Meaning compute(N) can have at most
        #   2 + d(M) digits
        # with d(M) being the number of digits of M. It is obvious that
        #   2 + d(M) <= M for all M >= 3
        # For some number N that has number of digits M < 3, compute(N) may
        # 'leak' into M + 1 digits range, even then it would eventually fall
        # into the M >= 3 range where it has an upper bound, i.e. cannot
        # approach infinity.
        #
        # With our proof that case 3 cannot happen, the problem becomes a
        # cycle detection problem in the form of finding the duplicate number
        # in an array.
        def compute(i: int) -> int:
            s = str(i)
            result = 0
            for c in s:
                result += int(c) ** 2
            return result

        # # Using Set
        # computed = set()
        # while n not in computed and n != 1:
        #     computed.add(n)
        #     n = compute(n)

        # return True if n == 1 else False

        # Tortoise and Hare
        slow = compute(n)
        fast = compute(compute(n))

        while slow != fast:
            slow = compute(slow)
            fast = compute(compute(fast))

        return slow == 1

    # 279. Perfect Squares
    def numSquares(self, n: int) -> int:
        def is_perfect_square(num: int) -> bool:
            return math.floor(math.sqrt(num)) ** 2 == num

        # As per Lagrange's four square theorem, there can only be
        # 4 possible results: 1, 2, 3, and 4
        # Case 1
        if is_perfect_square(n):
            return 1
        # Case 2
        for i in range(1, math.floor(math.sqrt(n) + 1)):
            if is_perfect_square(n - i**2):
                return 2

        # 2 cases left is 3 and 4
        # As per Legendre's three-square theorem, n can be
        # represented as the sum of three perfect squares iff
        # n is *not* of the form 4^a * (8*b + 7)
        # Case 3 and 4
        while n % 4 == 0:
            n //= 4
        if n % 8 == 7:
            return 4
        return 3

    # 1232. Check If It Is a Straight Line
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        x1, y1 = coordinates[0][0], coordinates[0][1]
        x2, y2 = coordinates[1][0], coordinates[1][1]

        for c in coordinates[2:]:
            x3, y3 = c[0], c[1]
            if (y2 - y1) * (x3 - x1) != (y3 - y1) * (x2 - x1):
                return False

        return True

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
        return (
            28 * weeks_num
            + (7 * weeks_num * (weeks_num - 1)) // 2
            + (days_left * weeks_num + (days_left * (days_left + 1)) // 2)
        )

    # 2485. Find the Pivot Integer
    def pivotInteger(self, n: int) -> int:
        # # O(n) with prefix sum
        # prefix_sum = [0] * (n + 1)
        # for i in range(1, n + 1):
        #     prefix_sum[i] = prefix_sum[i - 1] + i
        #
        # for i in range(1, n + 1):
        #     left = prefix_sum[i]
        #     right = prefix_sum[-1] - prefix_sum[i - 1]
        #     if left == right:
        #         return i
        #
        # return -1

        # O(1) math
        # 1 + 2 + ... + x = x + ... + n
        # x * (x + 1) // 2 = (n - x + 1) * (n + x) // 2
        # x^2 + x = n^2 - x^2 + n + x
        # 2 * x^2 = n^2 + n
        # x = sqrt((n^2 + n) // 2)
        # There is a pivot iff x is an integer, otherwise DNE
        y = (n**2 + n) // 2
        x = int(math.sqrt(y))
        return x if x**2 == y else -1

    # 3101. Count Alternating Subarrays
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        # # DP
        # # dp[i] will be the number of alternating subarrays ending with nums[i]
        # dp = [1]
        #
        # for i in range(1, len(nums)):
        #     if nums[i] != nums[i - 1]:
        #         dp.append(1 + dp[i - 1])
        #     else:
        #         dp.append(1)
        #
        # return sum(dp)

        # Math/Window
        result = 0
        prev = nums[0]
        segment = 1

        for num in nums[1:]:
            if num != prev:
                segment += 1
            else:
                result += (segment * (segment + 1)) // 2
                segment = 1
            prev = num

        result += (segment * (segment + 1)) // 2
        return result

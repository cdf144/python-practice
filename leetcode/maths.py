from typing import List


class Solution:
    # 7. Reverse Integer
    def reverse(self, x: int) -> int:
        sign = 1 if x >= 0 else -1
        x *= sign
        reverse = 0

        while x:
            reverse = reverse*10 + x % 10
            x //= 10

        return sign*reverse

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
                    check_x < 0 or check_x >= n
                    or check_y < 0 or check_y >= m
                    or matrix[check_y][check_x] == -101
            ):
                curr_dir = (curr_dir + 1) % 4

            matrix[y][x] = -101
            x += dirs[curr_dir][1]
            y += dirs[curr_dir][0]

        return result

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

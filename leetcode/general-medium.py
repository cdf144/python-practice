import collections
from itertools import starmap, pairwise
from operator import sub
from typing import List


class UF:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
        self.size = [1] * n

    def root(self, p: int) -> int:
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def connected(self, p: int, q: int) -> bool:
        return self.root(p) == self.root(q)

    def union(self, p: int, q: int) -> None:
        p_root = self.root(p)
        q_root = self.root(q)
        if p_root == q_root:
            return
        if self.size[p_root] <= self.size[q_root]:
            self.parent[p_root] = q_root
            self.size[q_root] += self.size[p_root]
        else:
            self.parent[q_root] = p_root
            self.size[p_root] += self.size[q_root]


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
        """
        Shrinking walls
        """
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

        """
        Switching directions
        """
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

    # 238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        Extra space
        """
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

        """
        Constant space
        """
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

    # 395. Longest Substring with At Least K Repeating Characters
    def longestSubstring(self, s: str, k: int) -> int:
        # If the frequency k cannot be reached
        if len(s) < k:
            return 0

        # Count the frequency of all characters in string
        # If we find a character whose frequency is less than k, we know
        # that that character cannot appear in *any* substring that
        # satisfies the requirement, and so we split the original string
        # with that character as separator, and do a recursive call for each
        # split part
        counter = collections.Counter(s)
        for c, count in counter.items():
            if count < k:
                return max(
                    self.longestSubstring(substr, k) for substr in s.split(c)
                )

        # If we reach here, all characters in string have frequency >= k
        return len(s)

    # 684. Redundant Connection
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        size = 0
        for edge in edges:
            size = max(size, edge[0], edge[1])

        uf = UF(size)
        result = [0] * 2
        for edge in edges:
            if not uf.connected(edge[0] - 1, edge[1] - 1):
                uf.union(edge[0] - 1, edge[1] - 1)
            else:
                result[0] = edge[0]
                result[1] = edge[1]

        return result

    # 1637. Widest Vertical Area Between Two Points Containing No Points
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        """
        Sorted set
        """
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

        """
        Starmap with tuple of adjacent pairs in sorted list
        """
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

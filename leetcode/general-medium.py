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
    # 50. Pow(x,n)
    def my_pow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.my_pow(x, -n)
        if n & 1:
            return x * self.my_pow(x, n - 1)
        return self.my_pow(x * x, n // 2)

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

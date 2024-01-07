import math
from typing import List

# Helpful guide to Binary Search
# https://leetcode.com/discuss/study-guide/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems


class Solution:
    # 35. Search Insert Position
    def searchInsert(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1
        mid = 0

        while low <= high:
            mid = low + (high - low)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        if nums[mid] < target:
            return mid + 1
        return mid

    # 69. Sqrt(x)
    def mySqrt(self, x: int) -> int:
        low, high = 0, x

        while low < high:
            mid = low + (high - low) // 2
            if mid * mid <= x:
                low = mid + 1
            else:
                high = mid

        return low - 1

    # 74. Search a 2D Matrix
    def search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        row_num = len(matrix)
        col_num = len(matrix[0])

        target_row = 0
        low = 0
        high = row_num - 1
        while low <= high:
            mid = low + (high - low) // 2
            if matrix[mid][0] <= target <= matrix[mid][col_num - 1]:
                target_row = mid
                break
            elif target > matrix[mid][col_num - 1]:
                low = mid + 1
            else:
                high = mid - 1

        if (target < matrix[target_row][0]
                or target > matrix[target_row][col_num - 1]):
            return False

        low = 0
        high = col_num - 1
        while low <= high:
            mid = low + (high - low) // 2

            if target == matrix[target_row][mid]:
                return True
            elif target > matrix[target_row][mid]:
                low = mid + 1
            else:
                high = mid - 1
        return False

    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        Naive linear search, TLE
        """
        # speed = max(piles)
        # hours_finish = len(piles)
        #
        # while hours_finish <= h:
        #     speed -= 1
        #     hours_finish = 0
        #     for pile in piles:
        #         hours_finish += math.ceil(pile / speed)
        #
        # return speed + 1

        """
        Binary search
        """
        low, high = 1, max(piles)

        # There can be cases where there are multiple speeds which Koko can eat
        # to finish in h hours. We want to find the lowest speed in that bunch.
        while low < high:
            speed = low + (high - low) // 2
            hours_finish = 0
            for pile in piles:
                hours_finish += math.ceil(pile / speed)
            if hours_finish > h:
                low = speed + 1
            else:
                high = speed

        return low

    # 704. Binary Search
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1

        while low <= high:
            mid = low + (high - low) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        return -1

import bisect
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

    # 153. Find Minimum in Rotated Sorted Array
    def findMin(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1

        # The main idea is to see that there is a 'pivot' after rotating the
        # array.
        # low will be indices in the range < pivot
        # high will be indices in the range >= pivot
        # So when the loop ends, low will be the minimal index in the range
        # >= pivot
        while low < high:
            mid = low + (high - low) // 2
            if nums[mid] <= nums[high]:
                # The pivot must have occurred at mid or to the left of mid.
                # It can't be to the right of middle, because then the values
                # would wrap around and nums[mid] > nums[high]
                high = mid
            else:
                # The pivot must have occurred to the right of mid, which is
                # why the values wrapped around and nums[mid] > nums[high]
                low = mid + 1

        return nums[low]

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

    # 1011. Capacity To Ship Packages Within D Days
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        """
        Manual Binary Search
        """
        # low, high = max(weights), sum(weights)
        #
        # while low < high:
        #     capacity = low + (high - low) // 2
        #
        #     days_needed = 1
        #     curr_weight = 0
        #     for weight in weights:
        #         curr_weight += weight
        #         if curr_weight > capacity:
        #             days_needed += 1
        #             curr_weight = weight
        #
        #     if days_needed <= days:
        #         high = capacity
        #     else:
        #         low = capacity + 1
        #
        # return low

        """
        Library (bisect)
        """
        def can_ship(capacity: int) -> bool:
            days_needed = 1
            curr_weight = 0
            for weight in weights:
                curr_weight += weight
                if curr_weight > capacity:
                    days_needed += 1
                    curr_weight = weight
            return days_needed <= days

        low, high = max(weights), sum(weights)
        return bisect.bisect_left(
            range(low, high),
            True,
            key=lambda mid: can_ship(mid)
        ) + low

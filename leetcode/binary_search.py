import bisect
import math
from typing import List

# Helpful guide to Binary Search
# https://leetcode.com/discuss/study-guide/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems


class Solution:
    # 4. Median of Two Sorted Arrays
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # # Easy way (Can also manually merge two lists instead of sorting
        # # which would be O(n) time)
        # return statistics.median(sorted(nums1 + nums2))

        # Binary search (O(log(min(m, n)) time)
        len1, len2 = len(nums1), len(nums2)
        if len1 > len2:
            return self.findMedianSortedArrays(nums2, nums1)

        MAX = 1000001
        MIN = -1000001
        total_half = (len1 + len2) // 2  # half partition of merged array
        low, high = 0, len1 - 1

        # A bit risky if input arrays are not actually sorted, but since test
        # cases are all valid, it is guaranteed that a solution is found.
        while True:
            # For other languages, '//' in Python is the floor division
            # operator, not integer division. So -1 // 2 == -1, not 0.
            partition1 = low + (high - low) // 2
            partition2 = total_half - (partition1 + 1) - 1

            max_l1 = nums1[partition1] if partition1 >= 0 else MIN
            max_l2 = nums2[partition2] if partition2 >= 0 else MIN

            min_r1 = nums1[partition1 + 1] if (partition1 + 1) < len1 else MAX
            min_r2 = nums2[partition2 + 1] if (partition2 + 1) < len2 else MAX

            if max_l1 <= min_r2 and max_l2 <= min_r1:
                return (
                    min(min_r1, min_r2)
                    if (len1 + len2) % 2 == 1
                    else (max(max_l1, max_l2) + min(min_r1, min_r2)) / 2
                )
            elif max_l1 > min_r2:
                high = partition1 - 1
            else:
                low = partition1 + 1

    # 33. Search in Rotated Sorted Array
    def search_rotated(self, nums: List[int], target: int) -> int:
        length = len(nums)
        low, high = 0, length - 1

        while low < high:
            mid = low + (high - low) // 2
            if nums[mid] <= nums[high]:
                high = mid
            else:
                low = mid + 1

        pivot = low
        low, high = 0, length - 1

        while low < high:
            mid = low + (high - low) // 2
            real_mid = (pivot + mid) % length
            if nums[real_mid] >= target:
                high = mid
            else:
                low = mid + 1

        result = (pivot + low) % length
        return result if nums[result] == target else -1

    # 34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        result = [-1, -1]
        if not nums:
            return result

        def binary_search(low: int, high: int, key: int) -> int:
            while low < high:
                mid = (low + high) // 2
                if nums[mid] >= key:
                    high = mid
                else:
                    low = mid + 1
            return low

        start = binary_search(0, len(nums) - 1, target)
        if nums[start] != target:
            return result
        result[0] = start
        end = binary_search(0, len(nums), target + 1) - 1
        result[1] = end
        return result

    # 35. Search Insert Position
    def searchInsert(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums)

        while low < high:
            mid = low + (high - low) // 2
            if nums[mid] >= target:
                high = mid
            else:
                low = mid + 1

        return low

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

        if target < matrix[target_row][0] or target > matrix[target_row][col_num - 1]:
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

    # 278. First Bad Version
    def firstBadVersion(self, n: int) -> int:
        # The isBadVersion API is already defined for you.
        def isBadVersion(version: int) -> bool:
            """
            Fake `isBadVersion` API to stop Linter from warning. Do not
            include this on LeetCode.
            """
            return True

        low, high = 1, n

        while low < high:
            mid = low + (high - low) // 2
            if isBadVersion(mid):
                high = mid
            else:
                low = mid + 1

        return low

    # 704. Binary Search
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1

        while low < high:
            mid = low + (high - low) // 2
            if nums[mid] >= target:
                high = mid
            else:
                low = mid + 1

        return low if nums[low] == target else -1

    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
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
        def can_ship(capacity: int) -> bool:
            """
            Calculate if it is possible to ship within the number of days
            required with the given capacity.
            :param capacity: The capacity given.
            :return: A Boolean value indicating if it is possible to ship
            within the number of days required with the given capacity.
            """
            days_needed = 1
            curr_weight = 0
            for weight in weights:
                curr_weight += weight
                if curr_weight > capacity:
                    days_needed += 1
                    curr_weight = weight
            return days_needed <= days

        low, high = max(weights), sum(weights)
        return (
            bisect.bisect_left(range(low, high), True, key=lambda mid: can_ship(mid))
            + low
        )

    # 1482. Minimum Number of Days to Make m Bouquets
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        if len(bloomDay) < m * k:
            return -1

        def get_bouquet_count(wait_days: int) -> int:
            """
            Return the number of bouquets (number of k adjacent bloomed flowers we can
            find) we can make if we wait `wait_days` days
            """
            bouquet_count = 0
            required_flowers = k

            for day in bloomDay:
                if day > wait_days:
                    # Reset `required_flowers` since there's not enough adjacent flowers
                    required_flowers = k
                else:
                    required_flowers -= 1
                if required_flowers == 0:
                    required_flowers = k
                    bouquet_count += 1

            return bouquet_count

        left, right = min(bloomDay), max(bloomDay)

        while left < right:
            mid = left + (right - left) // 2
            if get_bouquet_count(mid) >= mid:
                right = mid
            else:
                left = mid + 1

        return left

    # 1552. Magnetic Force Between Two Balls
    def maxDistance(self, position: List[int], m: int) -> int:
        position.sort()

        def can_distribute(force: int) -> bool:
            """
            Return `True` if it is possible to distribute `m` balls so that the
            minimum magnetic force between any two balls is `force`, else `False`.
            """
            balls_left = m
            prev = -force

            for pos in position:
                if pos >= prev + force:
                    balls_left -= 1
                    prev = pos
                if balls_left == 0:
                    break

            if balls_left != 0:
                return False
            return True

        left, right = 1, position[-1] + position[0]

        while left < right:
            mid = left + (right - left) // 2
            if not can_distribute(mid):
                right = mid
            else:
                left = mid + 1

        return left - 1

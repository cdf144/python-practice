import collections
import math
from typing import List


class Solution:
    # 3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        longest = 0
        counter = collections.Counter()

        left = 0
        for right, c in enumerate(s):
            counter[c] += 1
            while counter[c] > 1:
                counter[s[left]] -= 1
                left += 1
            longest = max(longest, right - left + 1)

        return longest

    # 209. Minimum Size Subarray Sum
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_length = math.inf
        left = 0
        curr_sum = 0

        for right, num in enumerate(nums):
            curr_sum += num
            if curr_sum >= target:
                while curr_sum >= target:
                    min_length = min(min_length, right - left + 1)
                    curr_sum -= nums[left]
                    left += 1

        return min_length if min_length != math.inf else 0

    # 567. Permutation in String
    def checkInclusion(self, s1: str, s2: str) -> bool:
        requiring = s1_len = len(s1)
        counter = collections.Counter(s1)

        left = 0
        for right, c in enumerate(s2):
            counter[c] -= 1
            if counter[c] >= 0:
                requiring -= 1
            while requiring == 0:
                if right - left + 1 == s1_len:
                    return True
                counter[s2[left]] += 1
                left += 1
                if counter[s2[left]] > 0:
                    requiring += 1

        return False

    # 643. Maximum Average Subarray I
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        length = len(nums)
        window_sum = sum(num for i, num in enumerate(nums[:k]))
        max_sum = window_sum

        for i in range(k, length):
            window_sum = window_sum - nums[i - k] + nums[i]
            max_sum = max(max_sum, window_sum)

        return max_sum / k

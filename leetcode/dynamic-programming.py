import collections
import math
import functools
from typing import List


class Solution:
    # 53. Maximum Subarray
    def maxSubArray(self, nums: List[int]) -> int:
        """
        Extra space
        """
        # length = len(nums)
        # dp = [0] * length
        #
        # dp[0] = nums[0]
        # for i in range(1, length):
        #     dp[i] = max(nums[i], dp[i-1] + nums[i])
        #
        # return max(dp)

        """
        Constant space
        """
        length = len(nums)
        curr = nums[0]
        max_sum = curr

        for i in range(1, length):
            prev = curr
            curr = max(nums[i], prev + nums[i])
            max_sum = max(max_sum, curr)

        return max_sum

    # 91. Decode Ways
    def numDecodings(self, s: str) -> int:
        length = len(s)
        if length == 0 or s[0] == '0':
            return 0

        dp = [1, 1]
        for i in range(1, length):
            curr_decode_way_count = 0
            if s[i - 1] != '0' and int(s[i - 1:i + 1]) <= 26:
                curr_decode_way_count += dp[-2]
            if s[i] != '0':  # 1 <= s[i] <= 9, can be individually decoded
                curr_decode_way_count += dp[-1]
            if curr_decode_way_count == 0:  # string having an invalid zero
                return 0
            dp.append(curr_decode_way_count)

        return dp[-1]

    # 1155. Number of Dice Rolls With Target Sum
    MOD = 10**9 + 7

    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        """
        Memoization / Top-Down
        """
        # lookup_table = {}
        #
        # def count_ways(die_remain: int, target_remain: int) -> int:
        #     if die_remain == 0:
        #         return 1 if target_remain == 0 else 0
        #     if (die_remain, target_remain) in lookup_table:
        #         return lookup_table[(die_remain, target_remain)]
        #
        #     count = 0
        #     for rolled in range(1, k + 1):  # possible rolled values
        #         count = (
        #             count + count_ways(die_remain - 1, target_remain - rolled)
        #         ) % self.MOD
        #
        #     lookup_table[(die_remain, target_remain)] = count
        #     return count
        #
        # return count_ways(n, target)

        """
        Tabulation / Bottom-Up
        """
        dp = [0] * (target + 1)  # dp[i] = number of ways to roll target=i
        dp[0] = 1

        for _ in range(n):  # for [1, n] dice
            new_dp = [0] * (target + 1)
            for rolled in range(1, k + 1):
                for t in range(rolled, target + 1):
                    new_dp[t] = (
                        new_dp[t] + dp[t - rolled]
                    ) % self.MOD
            dp = new_dp

        return dp[target]

    # 1335. Minimum Difficulty of a Job Schedule
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        length = len(jobDifficulty)
        if length < d:
            return -1

        @functools.lru_cache(None)
        def dp(i: int, d_remain: int, curr_day_max: int) -> int:
            if i == length:
                return 0 if d_remain == 0 else 2**32
            if d_remain == 0:
                return 2**32

            curr_day_max = max(curr_day_max, jobDifficulty[i])

            result = min(
                curr_day_max + dp(i + 1, d_remain - 1, -1),  # end day now
                dp(i + 1, d_remain, curr_day_max)  # continue day
            )

            return result

        return dp(0, d, -1)

    # 1531. String Compression II
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        def get_compressed_length(freq: int) -> int:
            if freq == 1:
                return 1
            if freq < 10:
                return 2
            if freq < 100:
                return 3
            return 4

        length = len(s)
        lookup_table = {}

        def dp(i: int, curr_k: int) -> int:
            if (i, curr_k) in lookup_table:
                return lookup_table[(i, curr_k)]
            # Invalid call
            if curr_k < 0:
                return 101  # inf
            # If substring is empty or its length is
            # lower than k (can be all deleted)
            if i == length or length - i <= curr_k:
                return 0

            # Our priority will be to keep letters with high frequency in the
            # string and remove the singular letters with the goal of grouping
            # the repeated letters together, making compression efficient
            max_freq = 0
            counter = collections.Counter()
            result = math.inf

            for j in range(i, length):
                counter[s[j]] += 1
                max_freq = max(max_freq, counter[s[j]])
                result = min(
                    result,
                    get_compressed_length(max_freq) + dp(
                        j + 1, curr_k - (j - i + 1 - max_freq)
                    )
                )

            lookup_table[(i, curr_k)] = result
            return result

        return dp(0, k)

import bisect
import collections
import math
import functools
from typing import List, Tuple


class Solution:
    # 5. Longest Palindromic Substring
    def longestPalindrome(self, s: str) -> str:
        # # Naive, DP-like, O(n^2) time, O(1) space if return string don't count
        # length = len(s)
        #
        # def search(left: int, right: int) -> Tuple[int, int]:
        #     while left >= 0 and right < length and s[left] == s[right]:
        #         right += 1
        #         left -= 1
        #     return left + 1, right - 1
        #
        # # [start, end] indices of the longest palindromic substring in s
        # indices = [0, 0]
        #
        # for i in range(length):
        #     l1, r1 = search(i, i)
        #     if r1 - l1 > indices[1] - indices[0]:
        #         indices[0] = l1
        #         indices[1] = r1
        #     if i + 1 < length and s[i] == s[i + 1]:
        #         l2, r2 = search(i, i + 1)
        #         if r2 - l2 > indices[1] - indices[0]:
        #             indices[0] = l2
        #             indices[1] = r2
        #
        # return s[indices[0]:indices[1] + 1]

        # Manacher's Algorithm, O(n) time, O(n) space
        # https://en.wikipedia.org/wiki/Longest_palindromic_substring#Manacher's_algorithm
        t = '|'.join('|' + s + '|')
        n = len(t)
        palindrome_radii = [0] * n
        center = 1
        radius = 0

        while center < n:
            while (
                    center + (radius + 1) < n
                    and center - (radius + 1) >= 0
                    and t[center + (radius + 1)] == t[center - (radius + 1)]
            ):
                radius += 1

            palindrome_radii[center] = radius

            old_center = center
            old_radius = radius
            center += 1
            radius = 0
            while center <= old_center + old_radius:
                mirrored_center = old_center - (center - old_center)
                max_mirrored_radius = old_center + old_radius - center

                if palindrome_radii[mirrored_center] < max_mirrored_radius:
                    palindrome_radii[center] = palindrome_radii[mirrored_center]
                    center += 1
                elif palindrome_radii[mirrored_center] > max_mirrored_radius:
                    palindrome_radii[center] = max_mirrored_radius
                    center += 1
                else:  # palindrome_radii[mirrored_center] = max_mirrored_radius
                    radius = max_mirrored_radius
                    break

        max_radius, best_center = max(
            (radius, center) for center, radius in enumerate(palindrome_radii)
        )
        return s[(best_center - max_radius)//2:(best_center + max_radius)//2]

    # 53. Maximum Subarray
    def maxSubArray(self, nums: List[int]) -> int:
        # # Extra space
        # length = len(nums)
        # dp = [0] * length
        #
        # dp[0] = nums[0]
        # for i in range(1, length):
        #     dp[i] = max(nums[i], dp[i-1] + nums[i])
        #
        # return max(dp)

        # Constant space
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
        if s[0] == '0':
            return 0
        # dp[i] will be the number of decoding ways of string length = i
        # dp[0] = 1 for base case, dp[1] = 1 because for a single character
        # '1' - '9' we can only decode one way.

        # # O(n) space
        # dp = [1, 1]
        # for i in range(1, length):
        #     curr_decode_way = 0
        #     if s[i] != '0':
        #         curr_decode_way += dp[-1]
        #     if s[i - 1] != '0' and int(s[i - 1:i + 1]) <= 26:
        #         curr_decode_way += dp[-2]
        #     if curr_decode_way == 0:
        #         return 0
        #     dp.append(curr_decode_way)
        #
        # return dp[-1]

        # O(1) space
        prev_1 = prev_2 = 1
        for i in range(1, length):
            # If a 1-digit or 2-digit can be decoded by itself, it can be
            # appended to another string and the number of ways to decode the
            # joined string is the number of ways to decode the string that was
            # appended to.
            curr_decode_way = 0
            # Valid digit that can be decoded by itself.
            if s[i] != '0':
                curr_decode_way += prev_1
            # Valid 2-digit that can be decoded.
            if s[i - 1] != '0' and int(s[i - 1:i + 1]) <= 26:
                curr_decode_way += prev_2
            # If we reach here, it means we have encountered an invalid '0' in
            # the middle of our string.
            if curr_decode_way == 0:
                return 0
            prev_2 = prev_1
            prev_1 = curr_decode_way

        return prev_1

    # 198. House Robber
    def rob1(self, nums: List[int]) -> int:
        house_num = len(nums)

        # # Memoization
        # @functools.lru_cache(None)
        # def dp(i: int) -> int:
        #     if i == house_num - 1:
        #         return nums[i]
        #     if i == house_num - 2:
        #         return max(nums[i], nums[i + 1])
        #
        #     max_money = max(nums[i] + dp(i + 2), dp(i + 1))
        #     return max_money
        #
        # return dp(0)

        # Tabulation
        # # O(n) space
        # dp = [0] * (house_num + 1)
        # dp[-2] = nums[-1]
        #
        # for i in range(house_num - 2, -1, -1):
        #     dp[i] = max(nums[i] + dp[i + 2], dp[i + 1])
        #
        # return dp[0]

        # O(1) space
        prev_1 = nums[-1]  # dp[i + 1]
        prev_2 = 0         # dp[i + 2]

        for i in range(house_num - 2, -1, -1):
            dp = max(nums[i] + prev_2, prev_1)
            prev_2 = prev_1
            prev_1 = dp

        return prev_1

    # 213. House Robber II
    def rob(self, nums: List[int]) -> int:
        # Basically the same as House Robber I, except we do 2 DP passes, one
        # in range(0, end - 1) and one in range(1, end)
        house_num = len(nums)
        if house_num == 1:
            return nums[0]

        def rob_range(low: int, high: int) -> int:
            prev_1 = nums[low]
            prev_2 = 0

            for i in range(low + 1, high):
                dp = max(nums[i] + prev_2, prev_1)
                prev_2 = prev_1
                prev_1 = dp

            return prev_1

        return max(rob_range(0, house_num - 1), rob_range(1, house_num))

    # 300. Longest Increasing Subsequence
    def lengthOfLIS(self, nums: List[int]) -> int:
        # # Memoization, Memory Limit Exceeded
        # length = len(nums)
        #
        # @functools.lru_cache(None)
        # def dp(i: int, prev: int) -> int:
        #     if i >= length:
        #         return 0
        #
        #     result = max(
        #         1 + dp(i + 1, i)
        #         if (prev == -1 or nums[i] > nums[prev])
        #         else 0,
        #         dp(i + 1, prev)
        #     )
        #
        #     return result
        #
        # return dp(0, -1)

        # Tabulation
        # dp[i] will be the length of the Longest Increasing Subsequence that
        # ends with nums[i] (in other words, with nums[i] being the greatest
        # number in the subsequence)
        dp = [1] * len(nums)

        # dp[i] = max(dp[j]) + 1
        # for all j preceding i and nums[j] < nums[i]
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        # Alternative top-down approach
        # length = len(nums)
        # for i in range(length -1, -1, -1):
        #     for j in range(i + 1, length):
        #         if nums[i] < nums[j]:
        #             dp[i] = max(dp[i], 1 + dp[j])

        return max(dp)

    # 322. Coin Change
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[i] is the fewest number of coins needed to make up i money

        # # Memoization / Top-Down
        # @functools.lru_cache(None)
        # def dp(i: int) -> int:
        #     if i == 0:
        #         return 0
        #
        #     result = -1
        #     for coin in coins:
        #         if coin <= i:
        #             coin_needed = dp(i - coin) + 1
        #             if coin_needed <= 0:
        #                 continue
        #             if result == -1:
        #                 result = coin_needed
        #             else:
        #                 result = min(result, coin_needed)
        #
        #     return result
        #
        # return dp(amount)

        # Tabulation / Bottom-Up
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != amount + 1 else -1

    # 413. Arithmetic Slices
    def numberOfArithmeticSlicesI(self, nums: List[int]) -> int:
        length = len(nums)
        if length < 3:
            return 0
        # dp[i] is the number of arithmetic slices ending with nums[i]

        # # O(n) space
        # dp = [0] * length
        #
        # for i in range(2, length):
        #     if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
        #         dp[i] = dp[i - 1] + 1
        #
        # return sum(dp)

        # O(1) space
        result = 0
        dp = 0

        for i in range(2, length):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp += 1
            else:
                dp = 0
            result += dp

        return result

    # 446. Arithmetic Slices II - Subsequence
    MIN_DIFF = -2**31
    MAX_DIFF = 2**31 - 1

    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # dp[i][diff] will be the number of arithmetic subsequences that ends
        # with nums[i] and have a difference between elements of diff
        length = len(nums)
        # Since i is bounded 0 <= i < len(nums), we can store dp[i] as an array,
        # but for diff, it is unbounded and sparse, so we store each dp[i][diff]
        # as a map diff to number of subsequences
        dp = [collections.defaultdict(int) for _ in range(length)]
        total_subsequence = 0

        for i in range(length):
            for j in range(i):
                # Since any arithmetic subsequence can't have a difference that
                # is other than in the range (-2^31, 2^31 - 1), since that would
                # mean the subsequence can only have 2 elements, we have to
                # check for this edge case
                diff = nums[i] - nums[j]
                if not self.MIN_DIFF < diff < self.MAX_DIFF:
                    continue

                # Since a subsequence is only valid when it has >= 3 elements,
                # we do a trick to add dp[j][diff] instead, because every
                # first access of dp[i][diff] marks it as the 2nd element in
                # the subsequence. When we access dp[i][diff] a second time,
                # it will be the 3rd element in the subsequence, and only then
                # do we add it to the total.
                total_subsequence += dp[j][diff]
                dp[i][diff] += dp[j][diff] + 1

        return total_subsequence
        # Another way to calculate the total number of arithmetic subsequences
        # would be to, instead of doing:
        # total_subsequence += dp[j][diff]
        # We do:
        # dp[i][diff] += dp[j][diff] + 1
        # total_subsequence += dp[i][diff]
        # Which effectively also include subsequences of length 2 in our result.
        # To subtract those, we subtract total_subsequence by
        # (length * (length - 1)) // 2 as per Combinatorics (number of ways
        # to Choose 2 elements from n elements)

    # 746. Min Cost Climbing Stairs
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        top = len(cost)

        # # Memoization
        # @functools.lru_cache(None)
        # def dp(i: int) -> int:
        #     if i <= 1:
        #         return 0
        #
        #     min_cost = min(cost[i - 1] + dp(i - 1), cost[i - 2] + dp(i - 2))
        #     return min_cost
        #
        # return dp(top)

        # Tabulation
        # O(n) space
        # dp = [0] * (top + 1)
        # dp[0] = dp[1] = 0
        #
        # for i in range(2, top + 1):
        #     dp[i] = min(cost[i - 1] + dp[i - 1], cost[i - 2] + dp[i - 2])
        #
        # return dp[-1]

        # O(1) space
        prev_1 = prev_2 = 0  # dp[i - 1] and dp[i - 2]
        for i in range(2, top + 1):
            dp = min(cost[i - 1] + prev_1, cost[i - 2] + prev_2)
            prev_2 = prev_1
            prev_1 = dp

        return prev_1

    # 1155. Number of Dice Rolls With Target Sum
    MOD = 10**9 + 7

    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        # # Memoization / Top-Down
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

        # Tabulation / Bottom-Up
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

    # 1235. Maximum Profit in Job Scheduling
    def jobScheduling(self, startTime: List[int], endTime: List[int],
                      profit: List[int]) -> int:
        # dp[i] will be the maximum profit we can make doing jobs starting
        # from time (interval) i
        job_intervals = sorted(zip(startTime, endTime, profit))
        num_intervals = len(job_intervals)

        # # Memoization
        # @functools.lru_cache(None)
        # def dp(i: int) -> int:
        #     # No job left
        #     if i == num_intervals:
        #         return 0
        #
        #     # At each interval, we have 2 choices
        #     # Not taking the job that is at this interval:
        #     result = dp(i + 1)
        #
        #     # Taking the job:
        #     # Find the next job that does not overlap with current job
        #     j = bisect.bisect(job_intervals, (job_intervals[i][1], -1, -1))
        #     result = max(result, job_intervals[i][2] + dp(j))
        #
        #     return result
        #
        # return dp(0)

        # Tabulation
        dp = [0] * (num_intervals + 1)

        for i in range(num_intervals - 1, -1, -1):
            j = bisect.bisect(job_intervals, (job_intervals[i][1], -1, -1))
            dp[i] = max(dp[i + 1], job_intervals[i][2] + dp[j])

        return dp[0]

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

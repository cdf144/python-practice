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

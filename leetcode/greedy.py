from typing import List


class Solution:
    # 55. Jump Game
    def canJump(self, nums: List[int]) -> bool:
        can_reach = 0
        curr_index = 0
        length = len(nums)

        while curr_index <= can_reach < length - 1:
            can_reach = max(can_reach, curr_index + nums[curr_index])
            curr_index += 1

        return can_reach >= length - 1

    # 122. Best Time to Buy and Sell Stock II
    def maxProfit(self, prices: List[int]) -> int:
        total = 0
        buy = 0

        for i in range(1, len(prices)):
            if prices[i] < prices[i - 1]:
                total += -prices[buy] + prices[i - 1]
                buy = i

        total += -prices[buy] + prices[len(prices) - 1]
        return total

    # 134. Gas Station
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start = 0
        gas_cost = 0
        curr_gas = 0

        for i in range(len(gas)):
            gas_cost += gas[i] - cost[i]
            curr_gas += gas[i] - cost[i]
            if curr_gas < 0:
                start = i + 1
                curr_gas = 0

        return start if gas_cost >= 0 else -1

    # 1578. Minimum Time to Make Rope Colorful
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        curr_group_sum = group_max = neededTime[0]
        count = 1
        min_needed_time = 0

        for i in range(1, len(colors)):
            if colors[i] != colors[i - 1]:
                if count != 1:
                    min_needed_time += curr_group_sum - group_max
                curr_group_sum = group_max = neededTime[i]
                count = 1
            else:
                curr_group_sum += neededTime[i]
                group_max = max(group_max, neededTime[i])
                count += 1

        if count != 1:
            min_needed_time += curr_group_sum - group_max

        return min_needed_time

    # 1903. Largest Odd Number in String
    def largestOddNumber(self, num: str) -> str:
        for i, c in enumerate(reversed(num)):
            if ord(c) & 1:
                return num[:len(num) - i]
        return ''

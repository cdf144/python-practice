from typing import List


class Solution:
    # 45. Jump Game II
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        curr_level = max_reach = 0
        min_jump = 0

        # Implicit BFS
        # For each jump we make, we can reach a next 'zone', that is, we can
        # reach any element in a range that is in front of the current index.
        # We do a greedy algorithm where the end of next zone will be the
        # maximum index we can reach considering all the jumps in our current
        # zone; we move to the next zone when we have considered all jump paths
        # in our current zone; if our current zone contains the last index,
        # that means we can reach the last index with the number of jump needed
        # to reach the current zone, which is the minimum jump needed.
        for i in range(n - 1):
            if curr_level >= n - 1:
                break
            max_reach = max(max_reach, i + nums[i])
            if i == curr_level:  # Considered all options in this zone (level)
                min_jump += 1  # Need to make a jump to move to next zone
                curr_level = max_reach  # Move to the next zone

        return min_jump

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

    # 455. Assign Cookies
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        children = len(g)
        cookies = len(s)

        child = 0
        cookie = 0
        while cookie < cookies and child < children:
            if g[child] <= s[cookie]:
                child += 1
            cookie += 1

        return child

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

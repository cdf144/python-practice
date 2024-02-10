import collections
import itertools
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

    # 763. Partition Labels
    def partitionLabels(self, s: str) -> List[int]:
        # We consider an example letter, let us call this letter 'a'.
        # Because each letter appears in at most one part, the partition 'a'
        # will be in cannot be shorter than
        # last_appearance(a) - first_appearance(a)
        # In other words, the partition must span all occurrences of 'a'.

        # Now consider this example: 'abfcab' (0-indexed)
        # We see that first_appearance(a) = 0, last_appearance(a) = 4
        # So our partition cannot be shorter than length 5. If we stop here
        # we would have a partition 'abfca'. However, notice the 'b' in the
        # partition, if we look at the example string again, we notice that
        # it appears again after this partition, which is not correct,
        # so the end of our partition must expand to last_appearance(b).
        # => The partition must contain last_appearance(char) for every char
        # in the partition
        result = []
        last_appearance = {c: i for i, c in enumerate(s)}

        start, end = 0, 0
        for i, c in enumerate(s):
            end = max(end, last_appearance[c])
            if i == end:
                result.append(end - start + 1)
                start = end + 1

        return result

    # 846. Hand of Straights
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        count = collections.Counter(hand)

        for group_start in sorted(count):
            cnt = count[group_start]
            if cnt > 0:
                # Groups must have consecutive cards
                for i in range(group_start, group_start + groupSize):
                    # Since there are at least 'cnt' groups, we also need 'cnt'
                    # number of each element that belongs to the group
                    count[i] -= cnt
                    if count[i] < 0:
                        return False

        return True

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

    # 1899. Merge Triplets to Form Target Triplet
    def mergeTriplets(self, triplets: List[List[int]],
                      target: List[int]) -> bool:
        found = [False] * 3
        for triplet in triplets:
            if all(a <= x for a, x in zip(triplet, target)):
                for i in range(3):
                    if triplet[i] == target[i]:
                        found[i] = True
                if all(found):
                    return True
        return False

    # 1903. Largest Odd Number in String
    def largestOddNumber(self, num: str) -> str:
        for i, c in enumerate(reversed(num)):
            if ord(c) & 1:
                return num[:len(num) - i]
        return ''

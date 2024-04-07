import collections
import heapq
import itertools
import math
from typing import List


class Solution:
    # 12. Integer to Roman
    def intToRoman(self, num: int) -> str:
        symbols = {
            "M": 1000,
            "CM": 900,
            "D": 500,
            "CD": 400,
            "C": 100,
            "XC": 90,
            "L": 50,
            "XL": 40,
            "X": 10,
            "IX": 9,
            "V": 5,
            "IV": 4,
            "I": 1,
        }

        result = ""
        for symbol, value in symbols.items():
            while num >= value:
                result += symbol
                num -= value

        return result

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

    # 452. Minimum Number of Arrows to Burst Balloons
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda point: point[1])
        n = len(points)
        result = 0

        i = 0
        while i < n:
            end = points[i][1]
            while i < n and points[i][0] <= end:
                i += 1
            result += 1

        return result

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

    # 678. Valid Parenthesis String
    def checkValidString(self, s: str) -> bool:
        # # DP
        # n = len(s)

        # # dp[i][j]: Is s[i...n) a valid string with balance factor j
        # @functools.cache
        # def dp(i: int, j: int) -> bool:
        #     if j < 0:
        #         return False
        #     if i == n:
        #         return True if j == 0 else False
        #     if s[i] == '(':
        #         return dp(i + 1, j + 1)
        #     elif s[i] == ')':
        #         return dp(i + 1, j - 1)
        #     else:
        #         return dp(i + 1, j - 1) or dp(i + 1, j) or dp(i + 1, j + 1)

        # return dp(0, 0)

        # Greedy
        # [low, high]: range of numbers of valid '('s
        low = high = 0

        for c in s:
            if c == "(":
                low += 1
                high += 1
            elif c == ")":
                low = max(low - 1, 0)
                high -= 1
            else:
                low = max(low - 1, 0)
                high += 1
            if high < low:
                return False

        return low == 0

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

    # 881. Boats to Save People
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        left, right = 0, len(people) - 1
        result = 0

        while left <= right:
            remain = limit - people[right]
            right -= 1
            if remain >= people[left]:
                left += 1
            result += 1

        return result

    # 948. Bag of Tokens
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        queue = collections.deque(sorted(tokens))
        result = score = 0

        while queue and (power >= queue[0] or score):
            while queue and power >= queue[0]:
                # Buy low
                power -= queue.popleft()
                score += 1
            result = max(result, score)

            if queue and score:
                # Sell high
                power += queue.pop()
                score -= 1

        return result

    # 1481. Least Number of Unique Integers after K Removals
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        freq_heap = list(collections.Counter(arr).values())
        heapq.heapify(freq_heap)
        while freq_heap and k >= freq_heap[0]:
            k -= heapq.heappop(freq_heap)
        return len(freq_heap)

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

    # 1642. Furthest Building You Can Reach
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        n = len(heights)

        # # DP, O(n * |bricks| * |ladders|) time and space
        # @functools.cache
        # def dp(i: int, j: int, k: int) -> int:
        #     if j < 0 or k < 0:
        #         return -1
        #     if i == n - 1 or (j == 0 and k == 0):
        #         return i

        #     diff = heights[i + 1] - heights[i]
        #     if diff <= 0:
        #         return dp(i + 1, j, k)
        #     return max(i, dp(i + 1, j - diff, k), dp(i + 1, j, k - 1))

        # return dp(0, bricks, ladders)

        # Greedy with Heap, O(n * |ladders| * log(|ladders|)) time,
        # O(|ladders|) space.
        # Use ladders for the largest differences, then greedily use bricks for
        # the smallest differences.
        heap = []

        for i, (a, b) in enumerate(itertools.pairwise(heights)):
            diff = b - a
            if diff <= 0:
                continue
            heapq.heappush(heap, diff)
            if ladders >= len(heap):
                continue
            bricks -= heapq.heappop(heap)
            if bricks < 0:  # No way to reach next building
                return i

        return n - 1

    # 1727. Largest Submatrix With Rearrangements
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        n = len(matrix[0])
        heights = [0] * n
        result = 0

        for row in matrix:
            for i, num in enumerate(row):
                heights[i] = heights[i] + 1 if num != 0 else 0

            histogram = sorted(heights)
            for i, height in enumerate(histogram):
                # width = (n - i)
                result = max(result, (n - i) * height)

        return result

    # 1899. Merge Triplets to Form Target Triplet
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
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
                return num[: len(num) - i]
        return ""

    # 2895. Minimum Processing Time
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        assert len(tasks) == len(processorTime) * 4
        processorTime.sort()
        tasks.sort()

        result = -math.inf
        for processor in processorTime:
            max_task = tasks.pop()
            result = max(result, processor + max_task)
            for _ in range(3):
                tasks.pop()

        assert isinstance(result, int)
        return result

    # 2971. Find Polygon With the Largest Perimeter
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()

        # # Left to right
        # result = -1
        # # Running sum of sides of a (possible) polygon
        # curr_sum = nums[0] + nums[1]
        # for i in range(2, len(nums)):
        #     if curr_sum > nums[i]:
        #         result = curr_sum + nums[i]
        #     curr_sum += nums[i]
        # return result

        # Right to left
        curr_sum = sum(nums)
        # Being greedy until we find a valid polygon
        # (perimeter > 2 * longest_side)
        # The result will be the polygon with the longest
        # possible longest_side
        while nums and curr_sum <= 2 * nums[-1]:
            curr_sum -= nums.pop()
        return curr_sum if len(nums) > 2 else -1

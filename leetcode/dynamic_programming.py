import bisect
import collections
import functools
import math
from typing import List


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
        t = "|".join("|" + s + "|")
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
        return s[(best_center - max_radius) // 2 : (best_center + max_radius) // 2]

    # 10. Regular Expression Matching
    def isMatch(self, s: str, p: str) -> bool:
        m = len(s)
        n = len(p)

        # dp[i][j]: Does s[i...m) match p[j...n)
        @functools.cache
        def dp(i: int, j: int) -> bool:
            if i == m and j == n:
                return True
            elif j == n:
                return False
            elif i == m:
                if j + 1 != n and p[j + 1] == "*":
                    return dp(i, j + 2)
                return False

            if j + 1 != n and p[j + 1] == "*":
                if p[j] != "." and s[i] != p[j]:
                    return dp(i, j + 2)
                else:
                    return (
                        dp(i, j + 2)  # matches zero
                        or dp(i + 1, j + 2)  # matches once
                        or dp(i + 1, j)  # matches > 1
                    )

            if p[j] == "." or s[i] == p[j]:
                return dp(i + 1, j + 1)

            return False

        return dp(0, 0)

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
        curr_sum = 0
        max_sum = -math.inf

        for num in nums:
            curr_sum = max(num, curr_sum + num)
            max_sum = max(max_sum, curr_sum)

        assert isinstance(max_sum, int)
        return max_sum

    # 62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        # # 2-D DP
        # @functools.lru_cache(None)
        # def dp(i: int, j: int) -> int:
        #     if not 0 <= i <= m or not 0 <= j <= n:
        #         return 0
        #     if i + 1 == m or j + 1 == n:
        #         return 1
        #     return dp(i + 1, j) + dp(i, j + 1)

        # return dp(0, 0)

        # Combinatorics
        return math.comb(m - 1 + n - 1, n - 1)

    # 70. Climbing Stairs
    def climbStairs(self, n: int) -> int:
        # dp[i] will be the number of distinct ways to climb to i. This problem
        # can also be thought of as calculating the nth Fibonacci

        # # Memoization
        # @functools.lru_cache(None)
        # def dp(i: int) -> int:
        #     if i == 1:
        #         return 1
        #     if i == 2:
        #         return 2
        #     return dp(i - 1) + dp(i - 2)
        #
        # return dp(n)

        # Tabulation
        if n == 1:
            return 1
        if n == 2:
            return 2

        dp_1 = 2  # dp[i - 1]
        dp_2 = 1  # dp[i - 2]
        for _ in range(3, n + 1):
            dp_i = dp_1 + dp_2
            dp_2 = dp_1
            dp_1 = dp_i

        return dp_1

    # 72. Edit Distance
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        # dp[i][j] will be the minimum number of operations needed to convert
        # word1[i...m) to word2[j...n)
        @functools.cache
        def dp(i: int, j: int) -> int:
            if i == m or j == n:
                return m - i if j == n else n - j
            if word1[i] == word2[j]:
                return dp(i + 1, j + 1)
            return 1 + min(dp(i + 1, j), dp(i, j + 1), dp(i + 1, j + 1))

        return dp(0, 0)

    # 91. Decode Ways
    def numDecodings(self, s: str) -> int:
        length = len(s)
        if s[0] == "0":
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
            if s[i] != "0":
                curr_decode_way += prev_1
            # Valid 2-digit that can be decoded.
            if s[i - 1] != "0" and int(s[i - 1 : i + 1]) <= 26:
                curr_decode_way += prev_2
            # If we reach here, it means we have encountered an invalid '0' in
            # the middle of our string.
            if curr_decode_way == 0:
                return 0
            prev_2 = prev_1
            prev_1 = curr_decode_way

        return prev_1

    # 97. Interleaving String
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m = len(s1)
        n = len(s2)
        if m + n != len(s3):
            return False

        @functools.lru_cache(None)
        def dp(i: int, j: int) -> bool:
            if i == m and j == n:
                return True
            elif i == m or j == n:
                remain = s3[i + j :]
                if i == m:
                    return remain == s2[j:]
                return remain == s1[i:]

            c = s3[i + j]
            if s1[i] != c and s2[j] != c:
                return False
            if s1[i] == c and s2[j] != c:
                return dp(i + 1, j)
            if s1[i] != c and s2[j] == c:
                return dp(i, j + 1)
            return dp(i + 1, j) or dp(i, j + 1)

        return dp(0, 0)

    # 115. Distinct Subsequences
    def numDistinct(self, s: str, t: str) -> int:
        # dp[i][j] is the number of distinct subsequences of s[0...i) which
        # equals t[0...j)
        @functools.lru_cache(None)
        def dp(i: int, j: int) -> int:
            if j == 0:
                # An empty string is one subsequence of any string
                return 1
            if i == 0 or i < j:
                return 0

            if s[i - 1] == t[j - 1]:
                return dp(i - 1, j) + dp(i - 1, j - 1)
            return dp(i - 1, j)

        return dp(len(s), len(t))

    # 139. Word Break
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        min_len = 21
        max_len = 0
        for word in wordDict:
            length = len(word)
            max_len = max(max_len, length)
            min_len = min(min_len, length)

        @functools.lru_cache(None)
        def dp(i: int) -> bool:
            if i == n:
                return True
            return any(
                s[i:j] in wordDict and dp(j)
                for j in range(min(i + min_len, n), min(i + max_len + 1, n + 1))
            )

        return dp(0)

    # 152. Maximum Product Subarray
    def maxProduct(self, nums: List[int]) -> int:
        max_product = nums[0]
        dp_min = nums[0]
        dp_max = nums[0]

        for i in range(1, len(nums)):
            num = nums[i]
            prev_min = dp_min
            prev_max = dp_max
            if num >= 0:
                dp_min = min(prev_min * num, num)
                dp_max = max(prev_max * num, num)
            else:
                dp_min = min(prev_max * num, num)
                dp_max = max(prev_min * num, num)
            max_product = max(max_product, dp_max)

        return max_product

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
        prev_2 = 0  # dp[i + 2]

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

    # 279. Perfect Squares
    def numSquares(self, n: int) -> int:
        def is_perfect_square(num: int) -> bool:
            return math.floor(math.sqrt(num)) ** 2 == num

        if is_perfect_square(n):
            return 1

        # # DP, Similar to the 'Coin Change' problem
        # squares = [i ** 2 for i in range(1, math.floor(math.sqrt(n) + 1))]

        # @functools.cache
        # def dp(i: int) -> int:
        #     if i <= 1:
        #         return i

        #     result = i
        #     for square in squares:
        #         if square > i:
        #             break
        #         result = min(result, dp(i - square) + 1)
        #     return result
        #
        # return dp(n)

        # Implicit BFS
        squares = [i**2 for i in range(1, math.floor(math.sqrt(n) + 1))]
        queue = set()
        for square in squares:
            queue.add(n - square)

        result = 1
        while queue:
            next_queue = set()
            for i in queue:
                if is_perfect_square(i):
                    return result + 1
                for square in squares:
                    if square > i:
                        break
                    next_queue.add(i - square)
            queue = next_queue
            result += 1

        return result

    # 300. Longest Increasing Subsequence
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp[i] will be the length of the Longest Increasing Subsequence that
        # ends with nums[i] (in other words, with nums[i] being the greatest
        # number in the subsequence)
        # dp[i] = max(dp[j]) + 1 for all j preceding i and nums[j] < nums[i]
        nums_len = len(nums)

        # # Memoization, Memory Limit Exceeded
        # @functools.lru_cache(None)
        # def dp(i: int, nxt: int) -> int:
        #     if i < 0:
        #         return 0
        #
        #     result = max(
        #         1 + dp(i - 1, i)
        #         if nxt == nums_len or nums[i] < nums[nxt]
        #         else 0,
        #         dp(i - 1, nxt)
        #     )
        #     return result
        #
        # return dp(nums_len - 1, nums_len)

        # # Tabulation
        # dp = [1] * nums_len
        #
        # for i in range(1, nums_len):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             dp[i] = max(dp[i], dp[j] + 1)
        #
        # return max(dp)

        # DP + Binary Search
        # dp[i] will be the ending value of Increasing Subsequence
        # of length i + 1
        dp = []

        # To update dp, for each nums[i], we find the maximal index j in dp
        # s. t. dp[j] < nums[i]. We then update dp[j + 1] = nums[i].
        # This has the meaning of: For each nums[i], we find the LIS *before*
        # i (suppose it is of length j), we can add nums[i] to that LIS,
        # and when we do, we find the LIS ending with nums[i] of length j + 1.
        # Hence, we update dp[j + 1].
        # It is trivial to see that dp[i] > dp[i - 1] > dp[i - 2]... This is why
        # we can use Binary Search to find dp[j], achieving O(n*log(n)) time.
        for i in range(len(nums)):
            if not dp or nums[i] > dp[-1]:
                dp.append(nums[i])
            else:
                dp[bisect.bisect_left(dp, nums[i])] = nums[i]

        return len(dp)

    # 309. Best Time to Buy and Sell Stock with Cooldown
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)

        @functools.lru_cache(None)
        def dp(i: int, bought: bool) -> int:
            if i >= n:
                return 0

            if bought:
                return max(dp(i + 1, True), dp(i + 2, False) + prices[i])
            else:
                return max(dp(i + 1, False), dp(i + 1, True) - prices[i])

        return dp(0, False)

    # 312. Burst Balloons
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        nums = [1] + nums + [1]

        # dp[i][j] will be the maximum coins can be collected by bursting
        # balloons in nums[i...j]
        @functools.lru_cache(None)
        def dp(i: int, j: int) -> int:
            if i > j:
                return 0

            result = 0
            for k in range(i, j + 1):
                # nums[k] will be the *last* balloon to be burst, so at the end,
                # when it is burst, its adjacent balloons will be nums[i - 1]
                # and nums[j + 1]
                burst = nums[i - 1] * nums[k] * nums[j + 1]
                result = max(result, dp(i, k - 1) + dp(k + 1, j) + burst)

            return result

        return dp(1, n)

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

    # 329. Longest Increasing Path in a Matrix
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            curr = matrix[i][j]
            result = 1
            for d in directions:
                next_i = i + d[0]
                next_j = j + d[1]
                if (
                    0 <= next_i < m
                    and 0 <= next_j < n
                    and curr < matrix[next_i][next_j]
                ):
                    result = max(result, 1 + dfs(next_i, next_j))
            return result

        return max(dfs(row, col) for row in range(m) for col in range(n))

    # 368. Largest Divisible Subset
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)

        # Similar to LIS problem
        # dp[i][prev] will be the largest divisible subset in nums[i...n) that
        # are all divisible by prev
        @functools.cache
        def dp(i: int, prev: int) -> List[int]:
            if i == n:
                return []
            num = nums[i]
            skip = dp(i + 1, prev)
            take = []
            if prev == -1 or nums[i] % prev == 0:
                take = [num] + dp(i + 1, num)

            return max(skip, take, key=len)

        return dp(0, -1)

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

    # 416. Partition Equal Subset Sum
    def canPartition(self, nums: List[int]) -> bool:
        sum_total = sum(nums)
        if sum_total % 2 == 1:
            return False
        return self._knapsack_can_partition(nums, sum_total // 2)

    def _knapsack_can_partition(self, nums: List[int], max_sum: int) -> bool:
        n = len(nums)
        # dp[i][j]: Can j be formed by choosing numbers in nums[:i]
        dp = [[False] * (max_sum + 1) for _ in range(n + 1)]
        dp[0][0] = True

        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(max_sum + 1):
                if num <= j:
                    # Sum can either include num or not
                    dp[i][j] = dp[i - 1][j - num] or dp[i - 1][j]
                else:
                    # Sum cannot include num
                    dp[i][j] = dp[i - 1][j]

        return dp[n][max_sum]

    # 446. Arithmetic Slices II - Subsequence
    MIN_DIFF = -(2**31)
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

    # 494. Target Sum
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # A way to think of this problem is we partition the array into 2
        # subsequences S1 and S2, where S1 - S2 = target.
        # We also have S1 + S2 = total => S1 = (total + target) / 2
        sum_total = sum(nums)
        if sum_total < abs(target) or (sum_total + target) % 2 == 1:
            return 0
        return self._knapsack_find_target(nums, (sum_total + target) // 2)

    def _knapsack_find_target(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(target + 1):
                if num <= target:
                    dp[i][j] = dp[i - 1][j - num] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]

        return dp[n][target]

    # 518. Coin Change II
    def change(self, amount: int, coins: List[int]) -> int:
        # dp(i, j) will be the number of combinations that make up i amount
        # using coins from 0 to j
        @functools.lru_cache(None)
        def dp(i: int, include: int) -> int:
            if i == 0:
                return 1
            if i < 0 or include == -1:
                return 0

            return dp(i - coins[include], include) + dp(i, include - 1)

        return dp(amount, len(coins) - 1)

    # 576. Out of Boundary Paths
    def findPaths(
        self, m: int, n: int, maxMove: int, startRow: int, startColumn: int
    ) -> int:
        mod = 10**9 + 7

        # dp[i][j][k] will be the number of ways to move out of boundary with
        # i as startRow and j as startColumn within k moves
        @functools.lru_cache(None)
        def dp(i: int, j: int, remain: int) -> int:
            if not (0 <= i < m) or not (0 <= j < n):
                return 1

            if (
                i - remain >= 0
                and i + remain < m
                and j - remain >= 0
                and j + remain < n
            ):
                # Stuck in the boundary regardless which path is taken
                return 0

            return (
                dp(i + 1, j, remain - 1)
                + dp(i - 1, j, remain - 1)
                + dp(i, j + 1, remain - 1)
                + dp(i, j - 1, remain - 1)
            ) % mod

        return dp(startRow, startColumn, maxMove)

    # 629. K Inverse Pairs Array
    def kInversePairs(self, n: int, k: int) -> int:
        # Detailed explanation:
        # https://hackmd.io/@ZGXH3BY8Sl2AyzcSB-leHA/B18Ek-G56
        # Not time and space efficient, in fact this is very inefficient
        # because it uses memoization and there are a lot of things that can
        # be optimized, but this is probably the easiest to understand.
        mod = 10**9 + 7

        # dp[i][j] is the number of permutations of {1,...i} such that there
        # are j inverse pairs
        @functools.lru_cache(None)
        def dp(i: int, j: int) -> int:
            if j == 0:
                return 1
            if i == 1 or j < 0:
                return 0

            # # O(n*k*min(n, k)), TLE
            # result = 0
            # for inverse in range(min(i, j + 1)):
            #     result += dp(i - 1, j - inverse)

            # O(n*k) using recurrence relation formula
            return (dp(i, j - 1) + dp(i - 1, j) - dp(i - 1, j - i)) % mod

        return dp(n, k)

    # 647. Palindromic Substrings
    def countSubstrings(self, s: str) -> int:
        # The same as Problem 5, except adding a counter.
        length = len(s)

        def search(low: int, high: int) -> None:
            nonlocal count_palindrome
            while low >= 0 and high < length and s[low] == s[high]:
                low -= 1
                high += 1
                count_palindrome += 1

        count_palindrome = 0
        for i in range(length):
            search(i, i)
            if i + 1 < length and s[i] == s[i + 1]:
                search(i, i + 1)

        return count_palindrome

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

    # 931. Minimum Falling Path Sum
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        # Since n == len(matrix) == len(matrix[i]), n is both row and col count.
        n = len(matrix)
        # dp[i][j] will be the min sum path ending at matrix[i][j]

        # # Memoization, TLE
        # @functools.lru_cache
        # def dp(i: int, j: int) -> int:
        #     if i == 0:
        #         return matrix[i][j]
        #
        #     min_s = 10005
        #     for k in range(max(0, j - 1), min(n, j + 2)):
        #         min_s = min(min_s, matrix[i][j] + dp(i - 1, k))
        #
        #     return min_s
        #
        # min_sum = 10005
        # for col in range(n):
        #     min_sum = min(min_sum, dp(n - 1, col))
        #
        # return min_sum

        # Tabulation
        # # O(n) space
        # dp = [[0] * n for _ in range(n)]
        # for i in range(n):
        #     dp[0][i] = matrix[0][i]
        #
        # for row in range(1, n):
        #     for col in range(n):
        #         prev_min_sum = 10005
        #         for k in range(max(0, col - 1), min(n, col + 2)):
        #             prev_min_sum = min(prev_min_sum, dp[row - 1][k])
        #         dp[row][col] = prev_min_sum + matrix[row][col]
        #
        # return min(dp[n - 1])

        # O(1) space
        for row in range(1, n):
            for col in range(n):
                prev_min_sum = 10005
                for k in range(max(0, col - 1), min(n, col + 2)):
                    prev_min_sum = min(prev_min_sum, matrix[row - 1][k])
                matrix[row][col] += prev_min_sum

        return min(matrix[-1])

    # 956. Tallest Billboard
    def tallestBillboard(self, rods: List[int]) -> int:
        n = len(rods)
        max_support = sum(rods) // 2
        # We will memoize (i, diff) with diff = abs(j - k); this has the
        # meaning of "The fixed distance to the tallest height possible of any
        # j, k pair of left, right supports using rods[i...n)."
        # For any pair of supports s1, s2 having diff = k, the 'distance' to
        # the tallest height possible for the pair is fixed, because we are
        # considering the same rods[i...n) range.
        memo = {}

        def dp(i: int, j: int, k: int) -> int:
            if j > max_support or k > max_support:
                return -1
            if i == n:
                return j if j == k else -1

            diff = abs(j - k)
            if (i, diff) in memo:
                # Since the 'distance' to the tallest height possible for the
                # pair is fixed, we know that the maximum of the 2 support sides
                # must be able to extend 'distance' amount
                return max(j, k) + memo[(i, diff)] if memo[(i, diff)] != -1 else -1

            rod = rods[i]
            result = max(
                dp(i + 1, j, k),  # Skip use
                dp(i + 1, j + rod, k),  # Weld onto left side
                dp(i + 1, j, k + rod),  # Weld onto right side
            )
            memo[(i, diff)] = result - max(j, k) if result != -1 else -1
            return result

        return max(0, dp(0, 0, 0))

    # 1043. Partition Array for Maximum Sum
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)

        # dp[i] will be the max sum after partitioning starting from index i
        @functools.lru_cache(None)
        def dp(i: int) -> int:
            if i == n:
                return 0

            result = 0
            maximum = -math.inf
            for j in range(1, min(n - i, k) + 1):
                maximum = max(maximum, arr[i + j - 1])
                result = max(result, dp(i + j) + maximum * j)

            assert isinstance(result, int)
            return result

        return dp(0)

    # 1049. Last Stone Weight II
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # Why this problem ties into Knapsack problem:
        # https://leetcode.com/problems/last-stone-weight-ii/solutions/653550/trying-to-explain-a-bit-logic-behind-trick
        stone_num = len(stones)
        total_weight = sum(stones)
        dp = [[0] * (total_weight // 2 + 1) for _ in range(stone_num + 1)]

        for i in range(1, stone_num + 1):
            stone = stones[i - 1]
            for j in range(total_weight // 2 + 1):
                if stone <= j:
                    dp[i][j] = max(dp[i - 1][j - stone] + stone, dp[i - 1][j])
                else:
                    dp[i][j] = dp[i - 1][j]

        return total_weight - 2 * dp[stone_num][total_weight // 2]

    # 1143. Longest Common Subsequence
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # Detailed explanation: https://ics.uci.edu/~eppstein/161/960229.html

        # # Memoization
        # @functools.lru_cache(None)
        # def dp(i: int, j: int) -> int:
        #     if i == len(text1) or j == len(text2):
        #         return 0
        #     if text1[i] == text2[j]:
        #         return 1 + dp(i + 1, j + 1)
        #     return max(dp(i + 1, j), dp(i, j + 1))
        #
        # return dp(0, 0)

        # Tabulation
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[0][0]

    # 1155. Number of Dice Rolls With Target Sum
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10**9 + 7
        # dp[i][j] will be the number of ways to roll to target j with i dices
        # that have k sides

        # # DP Top-Down
        # @functools.lru_cache(None)
        # def dp(i: int, j: int) -> int:
        #     if i == 0:
        #         return 1 if j == 0 else 0
        #     if i > j or j < 0:
        #         return 0

        #     result = 0
        #     for roll in range(1, k + 1):
        #         result += dp(i - 1, j - roll)

        #     return result % mod

        # return dp(n, target)

        # Knapsack
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            for j in range(1, target + 1):
                for roll in range(1, min(j, k) + 1):
                    dp[i][j] += dp[i - 1][j - roll]

        return dp[n][target] % mod

    # 1235. Maximum Profit in Job Scheduling
    def jobScheduling(
        self, startTime: List[int], endTime: List[int], profit: List[int]
    ) -> int:
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

    # 1269. Number of Ways to Stay in the Same Place After Some Steps
    def numWays(self, steps: int, arrLen: int) -> int:
        # Since we can only move right at most `steps / 2` steps to the right
        # (because then we would have to come back to index 0, and that takes
        # another `steps / 2` steps), we can reduce our maximum range `arrLen`
        # in case it is too big.
        # Because `arrLen` is exclusive, we reduce to `steps / 2 + 1` instead.
        arrLen = min(steps // 2 + 1, arrLen)
        mod = 10**9 + 7

        # dp[i][j] will be the number of ways to end up at index 0
        # after j steps while in the ith index
        @functools.cache
        def dp(i: int, j: int) -> int:
            if not 0 <= i < arrLen:
                return 0
            if j == 0:
                return 1 if i == 0 else 0
            return (
                dp(i - 1, j - 1) % mod + dp(i, j - 1) % mod + dp(i + 1, j - 1) % mod
            ) % mod

        return dp(0, steps)

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
                dp(i + 1, d_remain, curr_day_max),  # continue day
            )

            return result

        return dp(0, d, -1)

    # 1420. Build Array Where You Can Find The Maximum Exactly K Comparisons
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7

        # dp[i][j][cost] will be the number of ways to build array of i elements
        # with previous max element being j and current search_cost == cost
        @functools.cache
        def dp(i: int, j: int, cost: int) -> int:
            if cost > k:
                return 0
            if i == 0:
                return 1 if cost == k else 0

            # Appending any value in [1...j] does not change the max value and
            # search_cost
            result = (j * dp(i - 1, j, cost)) % mod
            # Appending any value in this range changes new max and search_cost
            for num in range(j + 1, m + 1):
                result = (result + dp(i - 1, num, cost + 1)) % mod
            return result

        return dp(n, 0, 0)

    # 1458. Max Dot Product of Two Subsequences
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)
        n = len(nums2)

        # dp[i][j] will be the max dot product of two subsequences of
        # nums1[i...m) and nums2[j...n)
        @functools.cache
        def dp(i: int, j: int):
            if i == m or j == n:
                return -math.inf
            return max(
                dp(i + 1, j),
                dp(i, j + 1),
                nums1[i] * nums2[j] + max(0, dp(i + 1, j + 1)),
            )

        result = dp(0, 0)
        assert isinstance(result, int)
        return result

    # 1463. Cherry Pickup II
    def cherryPickup(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # # Top-down, faster but less memory-efficient
        # # dp[i][j][k] will be the maximum number of cherries picked starting
        # # from row i, robot 1 and 2 at column j and k respectively
        # @functools.cache
        # def dp(i: int, j: int, k: int) -> int:
        #     if i == m or not 0 <= j < n or not 0 <= k < n:
        #         return 0

        #     curr_row_sum = grid[i][j] + (grid[i][k] if j != k else 0)
        #     result = 0
        #     for r1 in range(j - 1, j + 2):
        #         for r2 in range(k - 1, k + 2):
        #             result = max(result, curr_row_sum + dp(i + 1, r1, r2))

        #     return result

        # return dp(0, 0, n - 1)

        # Bottom-up, slower but more memory-efficient
        # dp[i][j][k] will be the maximum number of cherries collected ending
        # at row i, robot 1 and 2 on column j and k respectively
        dp = [[[-7001] * n for i in range(n)] for j in range(m)]
        dp[0][0][n - 1] = grid[0][0] + grid[0][n - 1]

        for i in range(1, m):
            for j in range(n):
                for k in range(n):
                    row_sum = grid[i][j] + (grid[i][k] if j != k else 0)
                    for r1 in range(max(j - 1, 0), min(j + 2, n)):
                        for r2 in range(max(k - 1, 0), min(k + 2, n)):
                            if dp[i - 1][r1][r2] == -7001:
                                continue
                            dp[i][j][k] = max(dp[i][j][k], row_sum + dp[i - 1][r1][r2])

        return max(max(row) for row in dp[m - 1])

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
                    get_compressed_length(max_freq)
                    + dp(j + 1, curr_k - (j - i + 1 - max_freq)),
                )

            assert isinstance(result, int)
            lookup_table[(i, curr_k)] = result
            return result

        return dp(0, k)

    # 2466. Count Ways To Build Good Strings
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10**9 + 7
        # dp[i] will be the number of good strings of length i that can be
        # constructed

        # # Memoization
        # @functools.cache
        # def dp(i: int) -> int:
        #     if i <= 0:
        #         return 1 if i == 0 else 0
        #     return dp(i - zero) % mod + dp(i - one) % mod

        # result = 0
        # for length in range(low, high + 1):
        #     result = (result + dp(length)) % mod
        # return result % mod

        # Tabulation
        dp = [0] * (high + 1)
        dp[0] = 1

        result = 0
        for i in range(1, high + 1):
            if i >= zero:
                dp[i] = (dp[i] + dp[i - zero]) % mod
            if i >= one:
                dp[i] = (dp[i] + dp[i - one]) % mod
            if low <= i <= high:
                result = (result + dp[i]) % mod

        return result

    # 2742. Painting the Walls
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        inf = (10**6) * 500 + 1

        # dp[i][j] will be the minimum cost to paint j walls if the paid
        # painter paint walls in range [i...n)
        @functools.cache
        def dp(i: int, j: int) -> int:
            if j <= 0:
                return 0
            if i == n:
                return inf
            return min(cost[i] + dp(i + 1, j - time[i] - 1), dp(i + 1, j))

        return dp(0, n)

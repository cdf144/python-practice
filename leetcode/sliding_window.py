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

    # 76. Minimum Window Substring
    def minWindow(self, s: str, t: str) -> str:
        requiring = len(t)
        counter = collections.Counter(t)
        min_left = -1
        min_len = math.inf

        left = 0
        for right, c in enumerate(s):
            counter[c] -= 1
            if counter[c] >= 0:
                requiring -= 1

            if requiring == 0:
                while requiring == 0:
                    counter[s[left]] += 1
                    if counter[s[left]] > 0:
                        requiring += 1
                    left += 1
                if min_len > right - (left - 1) + 1:
                    min_len = right - (left - 1) + 1
                    min_left = left - 1

        return s[min_left : min_left + min_len] if min_len != math.inf else ""

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

        return min_length if isinstance(min_length, int) else 0

    # 239. Sliding Window Maximum
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # We use a queue to store the 'potential' max's in a window.
        # The element at the top of the queue will be the current max,
        # and when that max isn't in the window anymore, the 2nd top
        # will be the max. This results in a monotonic decreasing queue.
        monotonic_queue = collections.deque()
        result = []

        for i, num in enumerate(nums):
            # If we find a new max, this max will invalidate all the
            # previous max's that are smaller, and it will last until
            # it is not in the sliding window anymore.
            while monotonic_queue and monotonic_queue[-1] < num:
                monotonic_queue.pop()
            monotonic_queue.append(num)

            # When the current max exits the window
            if i >= k and monotonic_queue[0] == nums[i - k]:
                monotonic_queue.popleft()

            # When window reaches size k, that is when we start adding max's
            if i >= k - 1:
                result.append(monotonic_queue[0])

        return result

    # 424. Longest Repeating Character Replacement
    def characterReplacement(self, s: str, k: int) -> int:
        counter = collections.Counter()
        max_freq = 0
        result = 0

        # The longest substring will contain characters with max frequency
        # plus k characters we will replace to get the target substring
        left = 0
        for right, c in enumerate(s):
            counter[c] += 1
            max_freq = max(max_freq, counter[c])
            # While our window is invalid, whether because there is too many
            # characters we have to replace (greater than k), or there is a
            # new character that has higher frequency
            while right - left + 1 > max_freq + k:
                counter[s[left]] -= 1
                left += 1
            result = max(result, right - left + 1)

        return result

    # 438. Find All Anagrams in a String
    def findAnagrams(self, s: str, p: str) -> List[int]:
        length_p = len(p)
        if len(s) < length_p:
            return []

        counter = collections.Counter(p)
        requiring = len(p)
        result = []

        for right, c in enumerate(s):
            counter[c] -= 1
            if counter[c] >= 0:
                requiring -= 1

            if right >= length_p:
                counter[s[right - length_p]] += 1
                if counter[s[right - length_p]] > 0:
                    requiring += 1

            if requiring == 0:
                result.append(right - length_p + 1)

        return result

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
                if counter[s2[left]] > 0:
                    requiring += 1
                left += 1

        return False

    # 643. Maximum Average Subarray I
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        length = len(nums)
        window_sum = sum(num for _, num in enumerate(nums[:k]))
        max_sum = window_sum

        for i in range(k, length):
            window_sum = window_sum - nums[i - k] + nums[i]
            max_sum = max(max_sum, window_sum)

        return max_sum / k

    # 713. Subarray Product Less Than K
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        result = 0

        curr_prod = 1
        left = 0
        for right, num in enumerate(nums):
            curr_prod *= num
            while curr_prod >= k and left <= right:
                curr_prod //= nums[left]
                left += 1
            result += right - left + 1

        return result

    # 930. Binary Subarrays With Sum
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        # # Prefix-sum
        # result = 0
        # prefix_sum = 0
        # # Number of sub-arrays with prefix-sum of val
        # prefix_count = collections.Counter({0: 1})
        #
        # for num in nums:
        #     prefix_sum += num
        #     result += prefix_count[prefix_sum - goal]
        #     prefix_count[prefix_sum] += 1
        #
        # return result

        # Sliding window
        n = len(nums)

        # Number of sub-arrays with sum of at most `max_sum`
        def num_subarray_at_most(max_sum: int) -> int:
            if max_sum < 0:
                return 0

            count = 0
            left = 0
            dist_to_max = max_sum
            for right in range(n):
                dist_to_max -= nums[right]
                while dist_to_max < 0:
                    dist_to_max += nums[left]
                    left += 1
                count += right - left + 1

            return count

        return num_subarray_at_most(goal) - num_subarray_at_most(goal - 1)

    # 992. Subarrays with K Different Integers
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def subarrays_with_at_most_k_distinct(k: int) -> int:
            result = 0
            count = collections.Counter()

            left = 0
            for right, num in enumerate(nums):
                count[num] += 1
                if count[num] == 1:
                    k -= 1
                while k < 0:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        k += 1
                    left += 1
                # If nums[left : right + 1] has k distinct integers, then
                # nums[i : right + 1] has at most k distinct integers
                # for all i in [left, right]
                # This has the meaning of the number of subarrays with k distinct
                # integers that ends with nums[right]
                result += right - left + 1

            return result

        return subarrays_with_at_most_k_distinct(k) - subarrays_with_at_most_k_distinct(
            k - 1
        )

    # 1004. Max Consecutive Ones III
    def longestOnes(self, nums: List[int], k: int) -> int:
        count = [0, 0]
        result = 0

        left = 0
        for right, num in enumerate(nums):
            count[num] += 1
            while count[0] > k:
                count[nums[left]] -= 1
                left += 1
            result = max(result, right - left + 1)

        return result

    # 1052. Grumpy Bookstore Owner
    def maxSatisfied(
        self, customers: List[int], grumpy: List[int], minutes: int
    ) -> int:
        satisfied = sum(c for i, c in enumerate(customers) if grumpy[i] == 0)
        # number of customers satisfied by keeping the owner not grumpy
        keep_satisfied = 0
        window_satisfied = 0

        for i, c in enumerate(customers):
            if grumpy[i] == 1:
                window_satisfied += c
            if i >= minutes and grumpy[i - minutes] == 1:
                window_satisfied -= customers[i - minutes]
            keep_satisfied = max(keep_satisfied, window_satisfied)

        return satisfied + keep_satisfied

    # 1208. Get Equal Substrings Within Budget
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        n = len(s)
        j = 0

        for i in range(n):
            maxCost -= abs(ord(s[i]) - ord(t[i]))
            # Don't need while loop, because the length of the max window will never
            # be smaller, it'll only get bigger.
            if maxCost < 0:
                maxCost += abs(ord(s[j]) - ord(t[j]))
                j += 1

        return n - j

    # 1248. Count Number of Nice Subarrays
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        def at_most_k_odd(k: int) -> int:
            """
            Count the number of subarrays with at most `k` odd numbers.
            """
            result = 0
            count_odd = 0
            left = 0

            for right, num in enumerate(nums):
                count_odd += num % 2
                while count_odd > k:
                    count_odd -= nums[left] % 2
                    left += 1
                result += right - left + 1

            return result

        return at_most_k_odd(k) - at_most_k_odd(k - 1)

    # 1838. Frequency of the Most Frequent Element
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        result = 1
        window_sum = 0

        left = 0
        for right, num in enumerate(nums):
            window_sum += num
            while num * (right - left + 1) > window_sum + k:
                window_sum -= nums[left]
                left += 1
            result = max(result, right - left + 1)

        return result

    # 1876. Substrings of Size Three with Distinct Characters
    def countGoodSubstrings(self, s: str) -> int:
        good_substr_count = 0

        # By default, zip() stops when the shortest iterable is exhausted.
        for c1, c2, c3 in zip(s, s[1:], s[2:]):
            if c1 == c2 or c2 == c3 or c3 == c1:
                continue
            good_substr_count += 1

        return good_substr_count

    # 2269. Find the K-Beauty of a Number
    def divisorSubstrings(self, num: int, k: int) -> int:
        k_beauty = 0
        num_str = str(num)

        for i in range(len(num_str) - k + 1):
            curr_num = int(num_str[i : i + k])
            if curr_num != 0 and num % curr_num == 0:
                k_beauty += 1

        return k_beauty

    # 2444. Count Subarrays With Fixed Bounds
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        result = 0

        last_min_k_idx = -1  # nums[last_min_k_idx] == minK
        last_max_k_idx = -1  # nums[last_max_k_idx] == maxK
        last_bad_idx = -1  # nums[last_bad_idx] > maxK or nums[last_bad_idx] < minK
        for i, num in enumerate(nums):
            if not minK <= num <= maxK:
                last_bad_idx = i
            if num == minK:
                last_min_k_idx = i
            if num == maxK:
                last_max_k_idx = i
            # We are now counting the number of valid subarrays that ends at index i.
            # For the starting index, we can choose any index in range
            # [last_bad_idx + 1, min(last_min_k_idx, last_max_k_idx)]
            result += max(0, min(last_min_k_idx, last_max_k_idx) - last_bad_idx)

        return result

    # 2958. Length of Longest Subarray With at Most K Frequency
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        count = collections.defaultdict(int)
        result = 0

        left = 0
        for right, num in enumerate(nums):
            count[num] += 1
            while count[num] > k:
                count[nums[left]] -= 1
                left += 1
            result = max(result, right - left + 1)

        return result

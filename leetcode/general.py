import collections
import itertools
from typing import List

# Topics like Sorting, Prefix Sum, Simulation, etc. can be added here


class Solution:
    # 238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # # Extra space
        # n = len(nums)
        # prefix = [1] * n
        # suffix = [1] * n
        #
        # for i in range(1, n):
        #     prefix[i] = prefix[i - 1] * nums[i - 1]
        # for i in range(n - 2, -1, -1):
        #     suffix[i] = suffix[i + 1] * nums[i + 1]
        #
        # result = [0] * n
        # for i in range(n):
        #     result[i] = prefix[i] * suffix[i]
        #
        # return result

        # Constant space
        n = len(nums)
        result = [1] * n

        for i in range(1, n):
            result[i] = result[i - 1] * nums[i - 1]

        curr_suffix = 1
        for i in range(n - 2, -1, -1):
            curr_suffix *= nums[i + 1]
            result[i] *= curr_suffix

        return result

    # 386. Lexicographical Numbers
    def lexicalOrder(self, n: int) -> List[int]:
        result = []
        i = 1
        for _ in range(n):
            result.append(i)
            if i * 10 <= n:
                i *= 10
            elif i % 10 != 9 and i + 1 <= n:
                i += 1
            else:
                while i % 10 == 9 or i + 1 > n:
                    i //= 10
                i += 1
        return result

    # 506. Relative Ranks
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        sorted_score = sorted(score, reverse=True)
        ranks = ["Gold Medal", "Silver Medal", "Bronze Medal"] + list(
            map(str, range(4, len(score) + 1))
        )
        score_rank = dict(zip(sorted_score, ranks))
        return [score_rank[sc] for sc in score]

    # 525. Contiguous Array
    def findMaxLength(self, nums: List[int]) -> int:
        result = 0
        prefix_to_idx = {}
        prefix_to_idx[0] = -1

        curr_prefix = 0
        for i, num in enumerate(nums):
            curr_prefix += 1 if num else -1
            result = max(result, i - prefix_to_idx.setdefault(curr_prefix, i))

        return result

    # 950. Reveal Cards In Increasing Order
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        n = len(deck)
        deck.sort()
        queue = collections.deque(range(n))
        result = [0] * n

        # Simulate card revealing process
        for i in range(n):
            # Reveal card
            result[queue.popleft()] = deck[i]
            # Move next card to bottom
            if queue:
                queue.append(queue.popleft())

        return result

    # 1291. Sequential Digits
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        # # Recursion
        # result = []

        # def dfs(s: str) -> None:
        #     if low <= int(s) <= high:
        #         result.append(int(s))
        #     elif int(s) > high:
        #         return
        #     if int(s[-1]) < 9:
        #         dfs(s + str(int(s[-1]) + 1))

        # for i in range(1, 9):
        #     dfs(str(i))

        # result.sort()
        # return result

        # Sliding Window
        s = "123456789"
        min_len = len(str(low))
        max_len = len(str(high))

        result = []
        for length in range(min_len, max_len + 1):
            for left in range(0, 10 - length):
                num = int(s[left : left + length])
                if low <= num <= high:
                    result.append(num)

        return result

    # 1442. Count Triplets That Can Form Two Arrays of Equal XOR
    def countTriplets(self, arr: List[int]) -> int:
        result = 0

        for i, start in enumerate(arr):
            prefix = start
            for k, num in enumerate(arr[i + 1 :], 1):
                prefix ^= num
                if prefix == 0:
                    result += k

        return result

    # 1608. Special Array With X Elements Greater Than or Equal X
    def specialArray(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        if nums[0] >= n:
            return n

        for i, (a, b) in enumerate(itertools.pairwise(nums)):
            count = n - i - 1
            if a < count and b >= count:
                return count

        return -1

    # 1685. Sum of Absolute Differences in a Sorted Array
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = list(itertools.accumulate(nums))
        suffix = list(itertools.accumulate(nums[::-1]))[::-1]
        return [
            num * (i + 1) - prefix[i] + suffix[i] - num * (n - i)
            for i, num in enumerate(nums)
        ]

    # 1877. Minimize Maximum Pair Sum in Array
    def minPairSum(self, nums: List[int]) -> int:
        n = len(nums)
        assert n % 2 == 0
        nums.sort()
        return max([nums[i] + nums[n - i - 1] for i in range(n // 2)])

    # 1887. Reduction Operations to Make the Array Elements Equal
    def reductionOperations(self, nums: List[int]) -> int:
        # # Counting sort, O(n) time, O(n) space
        # buckets = [0] * 50001  # lazy counting

        # for num in nums:
        #     buckets[num] += 1

        # count = []
        # for c in buckets:
        #     if c != 0:
        #         count.append(c)

        # result = 0
        # for i in range(len(count) - 1, 0, -1):
        #     result += count[i]
        #     count[i - 1] += count[i]

        # return result

        # Sort, O(n*log(n)) time, O(1) space
        nums.sort()
        result = 0

        for i in range(len(nums) - 2, -1, -1):
            if nums[i] != nums[i + 1]:
                result += len(nums) - 1 - i

        return result

    # 1915. Number of Wonderful Substrings
    def wonderfulSubstrings(self, word: str) -> int:
        result = 0
        # The running prefix of word. Since we only care about if characters in the
        # prefix appears even or odd number of times, we can use a bitmask. "1" means
        # the character appears odd number of times, "0" means even number of times.
        state = 0
        # How many times we have run into a state. Since we only have letters from 'a'
        # to 'j' (10 letters), there are 2^10 possible states.
        state_count = [0] * 1024
        state_count[0] = 1

        for c in word:
            state ^= 1 << (ord(c) - ord("a"))
            # If we have previously run into the same state, for example word[0..i] has
            # the same state as word[0..j], that means all characters in word[i..j]
            # appears even number of times.
            result += state_count[state]
            # If a substring have one character that appears odd number of times, it is
            # also a valid substring. To check that, we try to "flip" each bit of the
            # current state to see if after flipping, it turns into a state which we
            # have previously run into, that means word[i..j] contains exactly one
            # character appearing odd times. In other words, we only need to remove one
            # character in word[i..j] for it to be even substring.
            result += sum(state_count[state ^ (1 << i)] for i in range(10))
            state_count[state] += 1

        return result

    # 1930. Unique Length-3 Palindromic Subsequences
    def countPalindromicSubsequence(self, s: str) -> int:
        # Since we are only considering subsequences of length 3, we can think of it
        # as choosing 1 character for left and right, and choosing another character
        # for middle.
        # That way, our strategy will be to track the first and last occurrences of
        # every character in `s`, then everything in the middle of those 2 occurrences
        # is a candidate for middle character.
        n = len(s)
        result = 0
        first = [n] * 26
        last = [0] * 26

        for i, c in enumerate(s):
            c = ord(c) - ord("a")
            first[c] = min(first[c], i)
            last[c] = i

        for f, l in zip(first, last):
            if f < l:
                result += len(set(s[f + 1 : l]))

        return result

    # 2073. Time Needed to Buy Tickets
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        result = 0
        k_person = tickets[k]
        for person in tickets[:k]:
            result += min(person, k_person)
        for person in tickets[k + 1 :]:
            result += min(person, k_person - 1)
        return result + k_person

    # 3096. Minimum Levels to Gain More Points
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        possible = [1 if level == 1 else -1 for level in possible]
        summ = sum(possible)

        prefix = 0
        for i in range(n - 1):
            prefix += possible[i]
            if prefix > summ - prefix:
                return i + 1

        return -1

    # 3107. Minimum Operations to Make Median of Array Equal to K
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        result = 0
        mid = n // 2
        if nums[mid] > k:
            for i in range(mid + 1):
                result += max(nums[i] - k, 0)
        else:
            for i in range(mid, n):
                result += max(k - nums[i], 0)
        return result

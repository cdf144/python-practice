import collections
import heapq
import itertools
from typing import List


class Solution:
    # 1. Two Sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict_num_to_index = {}
        for i, num in enumerate(nums):
            if target - num in dict_num_to_index:
                return [i, dict_num_to_index[target - num]]
            dict_num_to_index[num] = i

    # 13. Roman to Integer
    def romanToInt(self, s: str) -> int:
        symbols = {'M': 1000, 'D': 500, 'C': 100,
                   'L': 50, 'X': 10, 'V': 5, 'I': 1}
        s = s.replace('IV', 'IIII').replace('IX', 'VIIII')
        s = s.replace('XL', 'XXXX').replace('XC', 'LXXXX')
        s = s.replace('CD', 'CCCC').replace('CM', 'DCCCC')

        result = 0
        for c in s:
            result += symbols[c]
        return result

    # 36. Valid Sudoku
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        seen = set()
        for i, row in enumerate(board):
            for j, c in enumerate(row):
                if c == '.':
                    continue

                if (
                    (c + '@row' + str(i)) in seen
                    or (c + '@col' + str(j)) in seen
                    or (c + '@box' + str(i // 3) + str(j // 3)) in seen
                ):
                    return False

                seen.add(c + '@row' + str(i))
                seen.add(c + '@col' + str(j))
                seen.add(c + '@box' + str(i // 3) + str(j // 3))

        return True

    # 49. Group Anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map_anagrams = collections.defaultdict(list)

        for s in strs:
            original = ''.join(sorted(s))
            map_anagrams[original].append(s)

        return list(map_anagrams.values())

    # 169. Majority Element
    def majorityElement(self, nums: List[int]) -> int:
        candidate = -1
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1
        return candidate

    # 205. Isomorphic Strings
    def isIsomorphic(self, s: str, t: str) -> bool:
        return list(map(s.index, s)) == list(map(t.index, t))

    # 242. Valid Anagram
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        counter = collections.Counter(s)
        counter.subtract(collections.Counter(t))
        return all(occurrence == 0 for occurrence in counter.values())

    # 347. Top K Frequent Elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # # Library function -> O(n) + O(k) + O((n - k) log k) + O(k log k)
        # counter = collections.Counter(nums)
        # return [num for num, freq in counter.most_common(k)]

        # # Max Heap -> O(n) + O(n log n) + O(n log k)
        # counter = collections.Counter(nums)
        # heap = []
        # for num, freq in counter.items():
        #     heapq.heappush(heap, (-freq, num))
        #
        # result = []
        # while heap and len(result) < k:
        #     result.append(heapq.heappop(heap)[1])
        # return result

        # Bucket sort -> O(n) + O(n)
        # bucket of frequencies, max frequency is len(nums)
        bucket = [[] for _ in nums]
        for num, freq in collections.Counter(nums).items():
            # We insert in -freq bucket, the thing is the higher
            # the freq, the closer to the start it will be in our
            # bucket. When we unpack and chain the bucket later,
            # the highest freq group will appear first
            bucket[-freq].append(num)
        return list((itertools.chain(*bucket)))[:k]

    # 383. Ransom Note
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magazine_char_counter = collections.Counter(magazine)
        ransom_char_counter = collections.Counter(ransomNote)
        for c in ransomNote:
            if ransom_char_counter[c] > magazine_char_counter[c]:
                return False
        return True

    # 387. First Unique Character in a String
    def firstUniqChar(self, s: str) -> int:
        count = collections.Counter(s)
        for c, cnt in count.items():
            if cnt == 1:
                return s.index(c)
        return -1

    # 395. Longest Substring with At Least K Repeating Characters
    def longestSubstring(self, s: str, k: int) -> int:
        # If the frequency k cannot be reached
        if len(s) < k:
            return 0

        # Count the frequency of all characters in string
        # If we find a character whose frequency is less than k, we know
        # that that character cannot appear in *any* substring that
        # satisfies the requirement, and so we split the original string
        # with that character as separator, and do a recursive call for each
        # split part
        counter = collections.Counter(s)
        for c, count in counter.items():
            if count < k:
                return max(
                    self.longestSubstring(substr, k) for substr in
                    s.split(c)
                )

        # If we reach here, all characters in string have frequency >= k
        return len(s)

    # 451. Sort Characters By Frequency
    def frequencySort(self, s: str) -> str:
        count = collections.Counter(s)

        # # Heap, O(n + m*log(m)) time, O(n + m) memory
        # heap = []
        # for c, cnt in count.items():
        #     heapq.heappush(heap, (-cnt, c))

        # result = ''
        # while heap:
        #     cnt, c = heapq.heappop(heap)
        #     result += c * (-cnt)
        # return result

        # Bucket sort, O(n + max_freq) time, O(n + max_freq) space
        max_freq = max(count.values())
        buckets = [[] for _ in range(max_freq + 1)]
        for c, cnt in count.items():
            buckets[cnt].append(c)

        result = []
        for freq in range(max_freq, 0, -1):
            for c in buckets[freq]:
                result.append(c * freq)
        return ''.join(result)

    # 645. Set Mismatch
    def findErrorNums(self, nums: List[int]) -> List[int]:
        # # Hash Table Counting
        # count = {i: 0 for i in range(1, len(nums) + 1)}
        # result = [0, 0]
        # for num in nums:
        #     count[num] += 1
        # for num, cnt in count.items():
        #     if cnt == 0:
        #         result[1] = num
        #     if cnt == 2:
        #         result[0] = num
        # return result

        # Math
        n = len(nums)
        A = n*(n + 1)//2 - sum(nums)
        B = n*(n + 1)*(2*n + 1)//6 - sum(i*i for i in nums)
        return [(B - A*A)//(2*A), (B + A*A)//(2*A)]

    # 1207. Unique Number of Occurrences
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        # return all(
        #     x != y for x, y in itertools.pairwise(
        #         sorted(collections.Counter(arr))
        #     )
        # )
        count = collections.Counter(arr)
        return len(set(count.values())) == len(count.values())

    # 1347. Minimum Number of Steps to Make Two Strings Anagram
    def minSteps(self, s: str, t: str) -> int:
        count_s = collections.Counter(s)
        requiring = len(s)

        count_t = collections.Counter(t)
        for c, count in count_t.items():
            if c in count_s:
                requiring -= count_s[c] if count > count_s[c] else count

        return requiring

    # 1424. Diagonal Traverse II
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        diagonals = collections.defaultdict(list)
        for i, row in enumerate(nums):
            for j, num in enumerate(row):
                diagonals[i + j].append(num)
        return itertools.chain.from_iterable(
            reversed(i) for i in diagonals.values()
        )

    # 1496. Path Crossing
    def isPathCrossing(self, path: str) -> bool:
        x, y = 0, 0
        visited = {(x, y)}

        for p in path:
            if p == 'N':
                y += 1
            elif p == 'E':
                x += 1
            elif p == 'S':
                y -= 1
            elif p == 'W':
                x -= 1

            if (x, y) in visited:
                return True
            else:
                visited.add((x, y))

        return False

    # 1624. Largest Substring Between Two Equal Characters
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        # dict_char_index = {}
        # for i, c in enumerate(s):
        #     dict_char_index.setdefault(c, []).append(i)
        #
        # max_substr = -1
        # for c, indices in dict_char_index.items():
        #     if len(indices) > 1:
        #         max_substr = max(max_substr, indices[-1] - indices[0] - 1)
        #
        # return max_substr

        max_substr = -1
        first_seen = {}

        for i, c in enumerate(s):
            if c not in first_seen:
                first_seen[c] = i
            else:
                max_substr = max(max_substr, i - first_seen[c] - 1)

        return max_substr

    # 1647. Minimum Deletions to Make Character Frequencies Unique
    def minDeletions(self, s: str) -> int:
        count = collections.Counter(s)
        freq_set = set()

        result = 0
        for freq in count.values():
            while freq and freq in freq_set:
                freq -= 1
                result += 1
            if freq:
                freq_set.add(freq)

        return result

    # 1657. Determine if Two Strings Are Close
    def closeStrings(self, word1: str, word2: str) -> bool:
        if len(word1) != len(word2):
            return False

        count_1 = collections.Counter(word1)
        count_2 = collections.Counter(word2)
        return (
            count_1.keys() == count_2.keys()
            and sorted(count_1.values()) == sorted(count_2.values())
        )

    # 1814. Count Nice Pairs in an Array
    def countNicePairs(self, nums: List[int]) -> int:
        # nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])
        # => nums[i] - rev(nums[i]) == nums[j] - rev[nums[j])
        def rev(x: int) -> int:
            return int(str(x)[::-1])

        mod = 10**9 + 7
        nice_groups = collections.defaultdict(int)
        result = 0
        for num in nums:
            group = num - rev(num)
            result = (result + nice_groups[group]) % mod
            nice_groups[group] += 1
        return result

    # 1897. Redistribute Characters to Make All Strings Equal
    def makeEqual(self, words: List[str]) -> bool:
        length = len(words)
        return all(
            count % length == 0
            for count in collections.Counter(
                itertools.chain.from_iterable(words)
            ).values()
        )

    # 2225. Find Players With Zero or One Losses
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        # Counting Hash Table, O(nlogn) time but more space efficient if
        # players are sparse
        player_lose_count = collections.Counter()
        for winner, loser in matches:
            if winner not in player_lose_count:
                player_lose_count[winner] = 0
            player_lose_count[loser] += 1

        result = [[] for _ in range(2)]
        for player, lose_count in player_lose_count.items():
            if lose_count < 2:
                result[lose_count].append(player)

        result[0].sort()
        result[1].sort()
        return result

        # # Counting sort, O(m + n) time but less space efficient if players
        # # are sparse
        # max_player = max(itertools.chain(*matches))
        # players = [0] * max_player
        # for winner, loser in matches:
        #     if players[winner - 1] >= 0:
        #         players[winner - 1] = 1
        #
        #     if players[loser - 1] >= 0:
        #         players[loser - 1] = -1
        #     elif players[loser - 1] < 0:
        #         players[loser - 1] -= 1
        #
        # result = [[] for _ in range(2)]
        # for player, count in enumerate(players):
        #     if count == 1:
        #         result[0].append(player + 1)
        #     elif count == -1:
        #         result[1].append(player + 1)
        #
        # return result

    # 2244. Minimum Rounds to Complete All Tasks
    def minimumRounds(self, tasks: List[int]) -> int:
        # Same as 2870
        counter = collections.Counter(tasks)
        result = 0

        for freq in counter.values():
            if freq < 2:
                return -1

            if freq % 3 == 0:
                result += freq // 3
            else:
                result += freq // 3 + 1

        return result

    # 2610. Convert an Array Into a 2D Array With Conditions
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        # # Brute force
        # result = []
        #
        # for num in nums:
        #     if not result:
        #         result.append([num])
        #     else:
        #         inserted = False
        #         for arr in result:
        #             if num not in arr:
        #                 arr.append(num)
        #                 inserted = True
        #                 break
        #         if not inserted:
        #             result.append([num])
        #
        # return result

        # Frequency hash table
        result = []
        count = collections.Counter()

        # Each row contains distinct elements -> the total number of row we
        # will end up with is the maximum frequency of a number in the given
        # list.
        # With a for loop, whenever a new max frequency is found, we will
        # create a new row (on demand).
        # If a number appears for the 1st time, it'll be in the 1st row.
        # If it appears a 2nd time, it'll be in the 2nd row, and so on...
        for num in nums:
            count[num] += 1
            if count[num] > len(result):
                result.append([])
            result[count[num] - 1].append(num)

        return result

    # 2870. Minimum Number of Operations to Make Array Empty
    def minOperations(self, nums: List[int]) -> int:
        # Same as 2244
        counter = collections.Counter(nums)
        result = 0

        for freq in counter.values():
            if freq % 3 == 0:
                result += freq // 3
                continue
            count = 0
            while freq > 4:
                freq -= 3
                count += 1
            if freq % 3 == 0 or freq % 2 == 0:
                count += max(freq // 3, freq // 2)
                result += count
            else:
                return -1

        return result

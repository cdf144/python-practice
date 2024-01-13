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

    # 1347. Minimum Number of Steps to Make Two Strings Anagram
    def minSteps(self, s: str, t: str) -> int:
        count_s = collections.Counter(s)
        requiring = len(s)

        count_t = collections.Counter(t)
        for c, count in count_t.items():
            if c in count_s:
                requiring -= count_s[c] if count > count_s[c] else count

        return requiring

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

    # 1897. Redistribute Characters to Make All Strings Equal
    def makeEqual(self, words: List[str]) -> bool:
        length = len(words)
        return all(
            count % length == 0
            for count in collections.Counter(
                itertools.chain.from_iterable(words)
            ).values()
        )

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

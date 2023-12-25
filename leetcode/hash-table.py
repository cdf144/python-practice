import collections
from typing import List


class Solution:
    # 1. Two Sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict_num_to_index = {}
        for i, num in enumerate(nums):
            if target - num in dict_num_to_index:
                return [i, dict_num_to_index[target - num]]
            dict_num_to_index[num] = i

    # 17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        dict_number_letter = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        result = ['']
        for d in digits:
            new_result = []
            for combination in result:
                for c in dict_number_letter[d]:
                    new_result.append(combination + c)
            result = new_result

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

    # 383. Ransom Note
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magazine_char_counter = collections.Counter(magazine)
        ransom_char_counter = collections.Counter(ransomNote)
        for c in ransomNote:
            if ransom_char_counter[c] > magazine_char_counter[c]:
                return False
        return True

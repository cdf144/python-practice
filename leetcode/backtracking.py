import itertools
from typing import List


class Solution:
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

    # 46. Permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        # # Library
        # return list(itertools.permutations(nums))

        # Backtracking
        result = []

        def dfs(curr_perm: List[int]) -> None:
            """
            Do a DFS on permutation decision tree.
            :param curr_perm: The current permutation built through making
                previous choices
            """
            if len(curr_perm) == len(nums):
                result.append(curr_perm)
                return

            for num in nums:
                if num not in curr_perm:
                    dfs(curr_perm + [num])

        dfs([])
        return result

    # 78. Subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums_len = len(nums)

        def dfs(start: int, curr_set: List[int]) -> None:
            """
            Do a DFS on the problem's decision tree to get all subsets.
            :param start: The index at which to start considering all the next
                choices (whether to add a number to the current set or not)
            :param curr_set: The current set built through making previous
                decisions
            """
            result.append(curr_set)

            for i in range(start, nums_len):
                dfs(i + 1, curr_set + [nums[i]])

        dfs(0, [])
        return result

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

    # 39. Combination Sum
    def combinationSum(self, candidates: List[int],
                       target: int) -> List[List[int]]:
        result = []
        num_candidates = len(candidates)

        def dfs(curr_comb: List[int], curr_sum: int, start: int) -> None:
            """
            Do a DFS on combination (with duplicate elements) decision tree.
            :param curr_comb: The current list of combination made by previous
                decisions
            :param curr_sum: The current sum of the combinations in the list. If
                the current sum is equal to target, the current list will be
                appended as 1 correct combination sum; if this is larger than
                target, we took the wrong decision and turn back.
            :param start: The index at which to start considering all
                possibilities of continuing to add the same candidate or add
                one of the next candidates.
            """
            if curr_sum >= target:
                if curr_sum == target:
                    result.append(curr_comb)
                return

            for i in range(start, num_candidates):
                dfs(curr_comb + [candidates[i]], curr_sum + candidates[i], i)

        dfs([], 0, 0)
        return result

    # 40. Combination Sum II
    def combinationSum2(self, candidates: List[int],
                        target: int) -> List[List[int]]:
        candidates.sort()
        result = []
        num_candidates = len(candidates)

        def dfs(curr_comb: List[int], curr_sum: int, start: int) -> None:
            if curr_sum >= target:
                if curr_sum == target:
                    result.append(curr_comb.copy())
                return

            for i in range(start, num_candidates):
                # Avoiding making duplicating choices. When we did the
                # DFS, all the possibilities of including duplicates are
                # already explored on the leftmost branches. Hence, for the
                # next branch that starts from here, we must choose another
                # candidate that is different.
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                curr_comb.append(candidates[i])
                dfs(curr_comb, curr_sum + candidates[i], i + 1)
                curr_comb.pop()

        dfs([], 0, 0)
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

    # 90. Subsets II
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        nums_len = len(nums)

        def dfs(curr_set: List[int], start: int) -> None:
            result.append(curr_set.copy())

            for i in range(start, nums_len):
                # The intuition is that for each path, we start with a unique
                # number. Then DFS will go through all possibilities of sets
                # having duplicates of that number in the leftmost branch, then
                # we will consider the next branch that does not include that
                # unique number at all, and we start with the next unique
                # number, guaranteeing us getting all subsets without
                # duplicates.
                if i > start and nums[i] == nums[i - 1]:
                    continue
                curr_set.append(nums[i])
                dfs(curr_set, i + 1)
                curr_set.pop()

        dfs([], 0)
        return result

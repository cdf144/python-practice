import collections
import functools
import itertools
from typing import List


class Solution:
    # 17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        number_letter = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        # # Iterative BFS
        # result = ['']
        # for digit in digits:
        #     new_result = []
        #     for prev_combination in result:
        #         for letter in number_letter[digit]:
        #             new_result.append(prev_combination + letter)
        #     result = new_result
        #
        # return result

        # DFS
        result = []
        combination_length = len(digits)

        def dfs(i: int, curr_comb: List[str]) -> None:
            """
            Do a DFS through the combination tree.
            :param i: The current digit decision node
            :param curr_comb: The current combination of letters
            """
            if len(curr_comb) == combination_length:
                result.append(''.join(curr_comb))
                return

            for letter in number_letter[digits[i]]:
                curr_comb.append(letter)
                dfs(i + 1, curr_comb)
                curr_comb.pop()

        dfs(0, [])
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

    # 51. N-Queens
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []
        invalid_col = [False] * n
        # If a Queen will meet another Queen if she goes diagonally up right
        invalid_diag_up = [False] * (2 * n - 1)
        # If a Queen will meet another Queen if she goes diagonally down right
        invalid_diag_down = [False] * (2 * n - 1)

        def dfs(row_num: int, curr_placement: List[str]) -> None:
            if row_num == n:
                result.append(curr_placement.copy())
                return

            curr_row = '.' * n
            for i in range(n):
                # for j, row in enumerate(curr_placement):
                #     if (
                #         i - row_num + j >= 0 and row[i - row_num + j] == 'Q'
                #         or i + row_num - j < n and row[i + row_num - j] == 'Q'
                #     ):
                #         invalid = True
                #         break
                if (
                    invalid_col[i]
                    or invalid_diag_up[i + row_num]
                    or invalid_diag_down[i + (n - 1 - row_num)]
                ):
                    continue

                curr_placement.append(curr_row[:i] + 'Q' + curr_row[i + 1:])
                invalid_col[i] = invalid_diag_up[i + row_num] = \
                    invalid_diag_down[i + (n - 1 - row_num)] = True
                dfs(row_num + 1, curr_placement)
                curr_placement.pop()
                invalid_col[i] = invalid_diag_up[i + row_num] = \
                    invalid_diag_down[i + (n - 1 - row_num)] = False

        dfs(0, [])
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

    # 79. Word Search
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        if len(word) > m * n:
            return False

        count = collections.Counter(list(itertools.chain.from_iterable(board)))
        for c, cnt in collections.Counter(word).items():
            if cnt > count[c]:
                return False

        # Reduce the number of starting cells for DFS.
        if count[word[0]] > count[word[-1]]:
            word = word[::-1]

        visited = set()
        dirs = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        def dfs(i: int, curr_y: int, curr_x: int) -> bool:
            """
            Do a DFS through choices of characters to choose.
            :param i: The current index of the character in the `word` we have
                to match
            :param curr_y: The row we are in
            :param curr_x: The column we are in
            :return: If the character at the current position does not match
                the ith character in `word`, or we fail to find a matching
                string no matter where we go down in the decision tree starting
                from this point, this returns False. Else we will find the
                matching string and return True
            """
            if i == len(word):
                return True

            if (
                    not 0 <= curr_x < n
                    or not 0 <= curr_y < m
                    or (curr_y, curr_x) in visited
            ):
                return False

            if word[i] != board[curr_y][curr_x]:
                return False

            visited.add((curr_y, curr_x))
            choose = (
                    dfs(i + 1, curr_y + dirs[0][0], curr_x + dirs[0][1])
                    or dfs(i + 1, curr_y + dirs[1][0], curr_x + dirs[1][1])
                    or dfs(i + 1, curr_y + dirs[2][0], curr_x + dirs[2][1])
                    or dfs(i + 1, curr_y + dirs[3][0], curr_x + dirs[3][1])
            )

            visited.remove((curr_y, curr_x))  # Backtrack
            return choose

        for y in range(m):
            for x in range(n):
                if dfs(0, y, x):
                    return True

        return False

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

    # 131. Palindrome Partitioning
    def partition(self, s: str) -> List[List[str]]:
        result = []
        s_len = len(s)

        @functools.lru_cache(None)
        def is_palindrome(t: str) -> bool:
            if len(t) == 1:
                return True
            return all(t[i] == t[-i - 1] for i in range(len(t) // 2))

        def dfs(start: int, partition: List[str]) -> None:
            """
            Do a DFS on the partitioning decision tree, whether to continue
            partitioning with current part substring of length 1 or more.
            """
            if start == s_len:
                result.append(partition.copy())
                return

            part = ''
            for i in range(start, s_len):
                part += s[i]
                if not is_palindrome(part):
                    continue
                partition.append(part)
                dfs(i + 1, partition)
                partition.pop()

        dfs(0, [])
        return result

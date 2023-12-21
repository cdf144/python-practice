from typing import List


class Solution:
    # 74. Search a 2D Matrix
    def search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        row_num = len(matrix)
        col_num = len(matrix[0])

        target_row = 0
        low = 0
        high = row_num - 1
        while low <= high:
            mid = low + (high - low) // 2
            if matrix[mid][0] <= target <= matrix[mid][col_num - 1]:
                target_row = mid
                break
            elif target > matrix[mid][col_num - 1]:
                low = mid + 1
            else:
                high = mid - 1

        if (target < matrix[target_row][0]
                or target > matrix[target_row][col_num - 1]):
            return False

        low = 0
        high = col_num - 1
        while low <= high:
            mid = low + (high - low) // 2

            if target == matrix[target_row][mid]:
                return True
            elif target > matrix[target_row][mid]:
                low = mid + 1
            else:
                high = mid - 1
        return False

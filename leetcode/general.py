from typing import List


class Solution:
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

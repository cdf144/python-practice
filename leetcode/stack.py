from typing import List


class Solution:
    # 20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        stack = []
        valid = ["()", "[]", "{}"]
        for c in s:
            if c in "([{":
                stack.append(c)
            elif not stack or stack.pop() + c not in valid:
                return False
        return not stack

    # 71. Simplify Path
    def simplifyPath(self, path: str) -> str:
        stack = []

        for directory in path.split("/"):
            if directory in ("", "."):
                continue

            if directory == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(directory)

        return "/" + "/".join(stack)

    # 84. Largest Rectangle in Histogram
    def largestRectangleArea(self, heights: List[int]) -> int:
        # We are basically finding the nearest left smaller and right smaller
        # column of each column in the histogram.
        # Then the maximum area that can be made with that column will be
        # column_height * (r_smaller_idx - l_smaller_idx)
        result = 0
        n = len(heights)

        # Here, we're using a stack to keep track of the indices of the
        # nearest left smaller column when we are iterating over the columns.
        # So nearest smaller of heights[stack[top]] is heights[stack[top - 1]].
        # We can see that this is a mono-increasing stack from bottom to top in
        # terms of the heights that the indices point to.
        # When we find a column which is lower than the top of the stack, that
        # will be the nearest right smaller column of the column at the top of
        # stack, and we calculate the area using the formula above.
        stack = []
        for i in range(n + 1):
            while stack and (i == n or heights[i] < heights[stack[-1]]):
                # Note that this also works if there are multiple equal
                # columns in a row and if there are leftovers in the stack after
                # we have gone through all columns (i == n).
                height = heights[stack.pop()]
                width = i - stack[-1] - 1 if stack else i
                result = max(result, width * height)
            stack.append(i)

        return result

    # 85. Maximal Rectangle
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # Treat each row as if they are the x-axis of a Histogram, then we
        # solve each 'Largest Rectangle in Histogram' problem and take the max.
        m = len(matrix)
        n = len(matrix[0])
        heights = [0] * n
        result = 0

        for row in matrix:
            for i, num in enumerate(row):
                heights[i] = heights[i] + 1 if num != "0" else 0
            result = max(result, self.largestRectangleArea(heights))

        return result

    # 150. Evaluate Reverse Polish Notation
    def evalRPN(self, tokens: List[str]) -> int:
        operations = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            # Since the division between two integers always truncates toward
            # zero, we use int() instead of math.floor()
            "/": lambda a, b: int(a / b),
        }
        number_stack = []

        for token in tokens:
            if token in operations:
                num2 = number_stack.pop()
                num1 = number_stack.pop()
                number_stack.append(operations[token](num1, num2))
            else:
                number_stack.append(int(token))

        return number_stack.pop()

    # 224. Basic Calculator
    def calculate(self, s: str) -> int:
        result = 0
        curr_num = 0
        # 1 == '+', -1 == '-'
        sign = 1
        stack = [sign]

        for c in s:
            if c == "(":
                stack.append(sign)
            elif c == ")":
                stack.pop()
            elif c.isdigit():
                curr_num = curr_num * 10 + (ord(c) - 48)  # ord('0') = 48
            elif c == "+" or c == "-":
                result += sign * curr_num
                curr_num = 0
                sign = (1 if c == "+" else -1) * stack[-1]

        return result + sign * curr_num

    # 316. Remove Duplicate Letters
    def removeDuplicateLetters(self, s: str) -> str:
        last_appearance = {}
        for i, c in enumerate(s):
            last_appearance[c] = i

        used = [False] * 26
        stack = []
        for i, c in enumerate(s):
            if used[ord(c) - ord("a")]:
                continue
            while stack and stack[-1] > c and i < last_appearance[stack[-1]]:
                used[ord(stack.pop()) - ord("a")] = False
            stack.append(c)
            used[ord(c) - ord("a")] = True

        return "".join(stack)

    # 496. Next Greater Element I
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        num_to_next_greater = {}
        stack = []

        for num in nums2:
            while stack and stack[-1] < num:
                num_to_next_greater[stack.pop()] = num
            stack.append(num)

        return [num_to_next_greater.get(num, -1) for num in nums1]

    # 503. Next Greater Element II
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        stack = []
        result = [-1] * n

        for i in range(n * 2):
            num = nums[i % n]
            while stack and nums[stack[-1]] < num:
                result[stack.pop()] = num
            stack.append(i % n)

        return result

    # 735. Asteroid Collision
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []

        for a in asteroids:
            if a < 0:
                while stack and 0 < stack[-1] < -a:
                    stack.pop()
                if stack and stack[-1] > 0:
                    if stack[-1] == -a:
                        stack.pop()
                    continue
            stack.append(a)

        return stack

    # 739. Daily Temperatures
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        result = [0] * len(temperatures)
        stack = []

        # # Right to left
        # for i in range(len(temperatures) - 1, -1, -1):
        #     curr_temp = temperatures[i]
        #     while stack and temperatures[stack[-1]] <= curr_temp:
        #         stack.pop()
        #     if stack:
        #         result[i] = stack[-1] - i
        #     stack.append(i)
        #
        # return result

        # Left to right
        for i, t in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < t:
                idx = stack.pop()
                result[idx] = i - idx
            stack.append(i)

        return result

    # 853. Car Fleet
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        # First model this problem on an integer number line with each car 'i'
        # being a point starting at 'position[i]' with velocity 'speed[i]'
        # moving towards the 'target' point, then see how faster cars get
        # 'speed limited' by the slower cars in front of them, forming a fleet.
        #
        # Then model again viewing the cars as a system of linear equations
        # y = position[i] + x*speed[i], see how on the Cartesian plane, the
        # lines with more slope gets 'limited' by ones with less slope once
        # they intersect, and this solution will make sense.
        times_to_reach_target = [
            (target - p) / s for p, s in sorted(zip(position, speed), reverse=True)
        ]

        stack = []
        for time in times_to_reach_target:
            stack.append(time)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()

        return len(stack)

    # 907. Sum of Subarray Minimums
    def sumSubarrayMins(self, arr: List[int]) -> int:
        # This problem is similar to 'Largest Rectangle in Histogram' problem,
        # except for this, we compute the nearest smaller indices on the left
        # and right of an element to determine how many sub-arrays will that
        # element be a minimum in.
        mod = 10**9 + 7
        arr_len = len(arr)
        # smaller_right[i] is the nearest index k such that arr[k] < arr[i]
        smaller_left = [-1] * arr_len
        # Similarly for min_right[i]
        smaller_right = [arr_len] * arr_len
        stack = []

        for i, num in enumerate(arr):
            while stack and arr[stack[-1]] > num:
                idx = stack.pop()
                smaller_right[idx] = i
            if stack:
                smaller_left[i] = stack[-1]
            stack.append(i)

        result = 0
        for i, num in enumerate(arr):
            # Think of this as 'How many ways to choose the length of the left
            # side and the right side of the sub-array with i as the center
            # index', then multiply the two together, and we get the number of
            # sub-arrays that num appears in.
            result += num * (i - smaller_left[i]) * (smaller_right[i] - i)
            result %= mod

        return result

    # 1544. Make The String Great
    def makeGood(self, s: str) -> str:
        stack = []
        for c in s:
            if stack and abs(ord(c) - ord(stack[-1])) == 32:
                stack.pop()
            else:
                stack.append(c)
        return "".join(stack)

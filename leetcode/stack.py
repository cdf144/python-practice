import collections
import math
from typing import List


class Solution:
    # 20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        stack = []
        valid = ['()', '[]', '{}']
        for c in s:
            if c in '([{':
                stack.append(c)
            elif (
                    not stack
                    or stack.pop() + c not in valid
            ):
                return False
        return not stack

    # 71. Simplify Path
    def simplifyPath(self, path: str) -> str:
        stack = []

        for directory in path.split('/'):
            if directory in ('', '.'):
                continue

            if directory == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(directory)

        return '/' + '/'.join(stack)

    # 150. Evaluate Reverse Polish Notation
    def evalRPN(self, tokens: List[str]) -> int:
        operations = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            # Since the division between two integers always truncates toward
            # zero, we use int() instead of math.floor()
            '/': lambda a, b: int(a / b)
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
            if c == '(':
                stack.append(sign)
            elif c == ')':
                stack.pop()
            elif c.isdigit():
                curr_num = curr_num * 10 + (ord(c) - 48)  # ord('0') = 48
            elif c == '+' or c == '-':
                result += sign * curr_num
                curr_num = 0
                sign = (1 if c == '+' else -1) * stack[-1]

        return result + sign * curr_num

    # 739. Daily Temperatures
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        Right to left
        """
        # result = [0] * len(temperatures)
        # stack = []
        #
        # for i in range(len(temperatures) - 1, -1, -1):
        #     curr_temp = temperatures[i]
        #     while stack and temperatures[stack[-1]] <= curr_temp:
        #         stack.pop()
        #
        #     if stack:
        #         result[i] = stack[-1] - i
        #
        #     stack.append(i)
        #
        # return result

        """
        Left to right
        """
        result = [0] * len(temperatures)
        stack = []

        for i, temperature in enumerate(temperatures):
            while stack and temperature > temperatures[stack[-1]]:
                idx = stack.pop()
                result[idx] = i - idx
            stack.append(i)

        return result

    # 853. Car Fleet
    def carFleet(self, target: int, position: List[int],
                 speed: List[int]) -> int:
        """
        First model this problem on an integer number line with each car 'i'
        being a point starting at 'position[i]' with velocity 'speed[i]'
        moving towards the 'target' point, then see how faster cars get
        'speed limited' by the slower cars in front of them, forming a fleet.

        Then model again viewing the cars as a system of linear equations
        y = position[i] + x*speed[i], see how on the Cartesian plane, the
        lines with more slope gets 'limited' by ones with less slope once
        they intersect, and this solution will make sense.
        """
        times_to_reach_target = [
            (target - p) / s
            for p, s in sorted(zip(position, speed), reverse=True)
        ]

        stack = []
        for time in times_to_reach_target:
            stack.append(time)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()

        return len(stack)

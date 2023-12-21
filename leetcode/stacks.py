class Solution:
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

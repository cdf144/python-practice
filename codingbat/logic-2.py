import collections


def make_bricks(small: int, big: int, goal: int) -> bool:
    return goal - 5 * big <= small and goal % 5 <= small


def lone_sum(a: int, b: int, c: int) -> int:
    # counter = collections.Counter([a, b, c])
    # return sum(num for num, count in counter.items() if count == 1)
    return (0 if a in (b, c) else a) + \
        (0 if b in (a, c) else b) + \
        (0 if c in (a, b) else c)


def lucky_sum(a: int, b: int, c: int) -> int:
    return 0 if a == 13 else a + (0 if b == 13 else b + (0 if c == 13 else c))


def no_teen_sum(a: int, b: int, c: int) -> int:
    def fix_teen(n: int) -> int:
        if 13 <= n <= 19 and n not in (15, 16):
            return 0
        return n

    return fix_teen(a) + fix_teen(b) + fix_teen(c)


def round_sum(a: int, b: int, c: int) -> int:
    def round10(num: int) -> int:
        unit = num % 10
        return num - unit if unit < 5 else num + 10 - unit

    return round10(a) + round10(b) + round10(c)


def close_far(a, b, c):
    diff_ab = abs(a - b)
    diff_ac = abs(a - c)
    diff_bc = abs(b - c)
    return (diff_bc > 1
            and (
                diff_ab <= 1 < diff_ac
                or diff_ac <= 1 < diff_ab
            ))


def make_chocolate(small, big, goal):
    if goal - 5 * big > small or goal % 5 > small:
        return -1
    need_big = goal // 5
    return goal - min(need_big, big)

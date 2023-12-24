import functools
import operator
from typing import List


def first_last6(nums: List[int]) -> bool:
    return nums[0] == 6 or nums[-1] == 6


def same_first_last(nums: List[int]) -> bool:
    return len(nums) > 0 and nums[0] == nums[-1]


def make_pi() -> List[int]:
    return [3, 1, 4]


def common_end(a: List[int], b: List[int]) -> bool:
    return a[0] == b[0] or a[-1] == b[-1]


def sum3(nums: List[int]) -> int:
    return functools.reduce(operator.add, nums)


def rotate_left3(nums: List[int]) -> List[int]:
    return [nums[1], nums[2], nums[0]]


def reverse3(nums: List[int]) -> List[int]:
    return nums[::-1]


def max_end3(nums: List[int]) -> List[int]:
    max_val = max(nums[0], nums[2])
    return [max_val] * 3


def sum2(nums: List[int]) -> int:
    if len(nums) < 2:
        if len(nums) == 0:
            return 0
        return nums[0]
    else:
        return nums[0] + nums[1]


def make_ends(nums: List[int]) -> List[int]:
    return [nums[0], nums[-1]]


def has23(nums: List[int]) -> bool:
    return 2 in nums or 3 in nums


def middle_way(a: List[int], b: List[int]) -> List[int]:
    return [a[1], b[1]]


if __name__ == '__main__':
    print(sum3([5, 11, 2]))

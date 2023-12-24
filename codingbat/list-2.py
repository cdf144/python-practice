from typing import List


def count_evens(nums: List[int]) -> int:
    return len([num for num in nums if num % 2 == 0])


def big_diff(nums: List[int]) -> int:
    # max_num = 2
    # min_num = 10
    # for num in nums:
    #     max_num = max(max_num, num)
    #     min_num = min(min_num, num)
    # return max_num - min_num
    return max(nums) - min(nums)


def centered_average(nums: List[int]) -> int:
    return (sum(nums) - min(nums) - max(nums)) // (len(nums) - 2)


def sum13(nums: List[int]) -> int:
    # length = len(nums)
    # s = 0
    # i = 0
    # while i < length:
    #     if nums[i] == 13:
    #         i += 2
    #     else:
    #         s += nums[i]
    #         i += 1
    # return s
    return sum(
        num for i, num in enumerate(nums)
        if num != 13 and not (i > 0 and nums[i-1] == 13)
    )


def sum67(nums: List[int]) -> int:
    # s = 0
    # length = len(nums)
    # skip = False
    # for num in nums:
    #     if skip:
    #         if num == 7:
    #             skip = False
    #     else:
    #         if num == 6:
    #             skip = True
    #         else:
    #             s += num
    # return s
    return (
        (
            sum(nums[:nums.index(6)]) +
            sum67(nums[nums.index(7, nums.index(6)) + 1:])
        )
        if 6 in nums else sum(nums)
    )


def has22(nums: List[int]) -> bool:
    # for i in range(len(nums) - 1):
    #     if nums[i] == 2:
    #         if nums[i + 1] == 2:
    #             return True
    # return False
    return nums[:2] == [2, 2] or (has22(nums[1:]) if len(nums) > 2 else False)

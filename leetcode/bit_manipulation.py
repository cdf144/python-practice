from typing import List


class Solution:
    # 136. Single Number
    def singleNumber(self, nums: List[int]) -> int:
        # XOR gives 0 if both bits are the same, 1 otherwise.
        # If a number appears twice, the bits added to 'mask' the first time the
        # number is encountered will be reverted to 0 the second time it is met.
        mask = 0
        for num in nums:
            mask = mask ^ num
        return mask

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

    # 201. Bitwise AND of Numbers Range
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        if left == 0 or left.bit_length() != right.bit_length():
            return 0
        # Find the common bit prefix between left and right
        while right != left:
            # a & (a - 1) removes rightmost significant bit
            right &= right - 1
        return right

    # 231. Power of Two
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    # 268. Missing Number
    def missingNumber(self, nums: List[int]) -> int:
        result = len(nums)
        for i, num in enumerate(nums):
            result ^= i ^ num
        return result

    # 2997. Minimum Number of Operations to Make Array XOR Equal to K
    def minOperations(self, nums: List[int], k: int) -> int:
        # General rule: no. of 1 bits is odd -> XOR bit = 0, else 1
        xor = 0
        for num in nums:
            xor ^= num
        # For every bit that is different than k in array XOR, we only need to flip
        # one bit of an element in `nums` to make that bit equal.
        return (xor ^ k).bit_count()

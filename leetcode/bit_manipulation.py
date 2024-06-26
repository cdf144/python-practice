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

    # 260. Single Number III
    def singleNumberIII(self, nums: List[int]) -> List[int]:
        # XOR of array gives XOR of the two number we need to find.
        xor = 0
        for num in nums:
            xor ^= num
        # Find the rightmost set bit in XOR. This bit is one of the place where the
        # two numbers differ (one will have this bit '0', the other '1').
        mask = xor & -xor
        # Separate the two unique numbers based on the bit.
        result = [0, 0]
        for num in nums:
            if num & mask:
                result[1] ^= num
            else:
                result[0] ^= num
        return result

    # 231. Power of Two
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    # 268. Missing Number
    def missingNumber(self, nums: List[int]) -> int:
        result = len(nums)
        for i, num in enumerate(nums):
            result ^= i ^ num
        return result

    # 1310. XOR Queries of a Subarray
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        n = len(arr)
        prefix = [0] * n
        prefix[0] = arr[0]

        for i, num in enumerate(arr[1:], 1):
            prefix[i] = prefix[i - 1] ^ num

        result = []
        for i, j in queries:
            result.append(prefix[j] ^ prefix[i - 1] if i > 0 else prefix)
        return result

    # 2275. Largest Combination With Bitwise AND Greater Than Zero
    def largestCombination(self, candidates: List[int]) -> int:
        return max(sum(num >> i & 1 for num in candidates) for i in range(24))

    # 2433. Find The Original Array of Prefix Xor
    def findArray(self, pref: List[int]) -> List[int]:
        n = len(pref)
        result = [0] * n
        result[0] = pref[0]
        for i in range(1, n):
            result[i] = pref[i - 1] ^ pref[i]
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

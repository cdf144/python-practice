from typing import List


def string_times(str, n: int) -> str:
    return n * str


def front_times(str, n: int) -> str:
    return n * str[:3]


def string_bits(str) -> str:
    return str[::2]


def string_splosion(str) -> str:
    s = ""
    for i in range(1, len(str) + 1):
        s += str[:i]

    return s


def last2(str) -> int:
    if len(str) < 2:
        return 0

    last_2 = str[-2:]
    count = 0
    for i in range(len(str) - 2):
        substr = str[i:i + 2]
        if substr == last_2:
            count += 1

    return count


def array_count9(nums: List[int]) -> int:
    return nums.count(9)


def array_front9(nums: List[int]) -> bool:
    length = 4
    if len(nums) < length:
        length = len(nums)

    for i in range(length):
        if nums[i] == 9:
            return True

    return False


def array123(nums: List[int]) -> bool:
    if len(nums) < 3:
        return False

    for i in range(len(nums) - 2):
        if (nums[i] == 1
                and nums[i + 1] == 2
                and nums[i + 2] == 3):
            return True

    return False


def string_match(a: str, b: str) -> int:
    length = min(len(a), len(b))
    count = 0
    for i in range(length - 1):
        substr1 = a[i:i + 2]
        substr2 = b[i:i + 2]
        if substr1 == substr2:
            count += 1

    return count


if __name__ == "__main__":
    print(front_times("Ab", 5))

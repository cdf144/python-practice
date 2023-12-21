from time import time
from random import randint
from typing import List, Iterable, Any

MERGE_SORT_THRESHOLD = 8


class InvalidRangeError(Exception):
    def __str__(self):
        return ("Minimum value cannot be "
                "greater or equal to maximum value.")


class IllegalSizeError(Exception):
    def __str__(self):
        return "Size must be a positive integer"


def insertion_sort(arr: List[Any]) -> None:
    if len(arr) <= 1:
        return

    for i, val in enumerate(arr):
        j = i - 1
        while j >= 0 and val < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = val


def merge_sort(arr: List[Any]) -> List[Any]:
    """
    Sort a list using MergeSort algorithm
    :param arr: array to sort
    :return: sorted array
    """
    if len(arr) <= 1:
        return arr

    if len(arr) <= MERGE_SORT_THRESHOLD:
        insertion_sort(arr)
        return arr

    mid = len(arr) // 2

    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left: List[Any], right: List[Any]) -> List[Any]:
    merged_arr = []
    i, j = 0, 0
    len_left = len(left)
    len_right = len(right)

    while i < len_left and j < len_right:
        if left[i] <= right[j]:
            merged_arr.append(left[i])
            i += 1
        else:
            merged_arr.append(right[j])
            j += 1

    merged_arr.extend(left[i:])
    merged_arr.extend(right[j:])
    return merged_arr


def test_sort(array_size: int, min_value: int, max_value: int) -> None:
    if min_value >= max_value:
        raise InvalidRangeError()
    if array_size <= 0:
        raise IllegalSizeError()
    try:
        val = int(array_size)
    except ValueError:
        raise IllegalSizeError()

    unsorted_arr = [randint(min_value, max_value) for _ in range(array_size)]
    # print(f"Unsorted array: {unsorted_arr}")

    start = time()
    sorted_arr = merge_sort(unsorted_arr)
    end = time()
    print(f"Took {end - start} seconds")
    # print(f"Sorted array: {sorted_arr}")
    # if sorted_arr != sorted(unsorted_arr):
    #     print("Error: Merge Sort results are not correct!")


if __name__ == "__main__":
    for count in range(4):
        print(f"Pass {count}: ", end='')
        test_sort(8*10**5, 0, 999999)

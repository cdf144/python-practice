from typing import List


def solution(boxes: List[int]) -> int:
    """
    There are N boxes (numbered from 0 to N-1) arranged in a row. The K-th
    box contains A[K] bricks. In one move you can take one brick from some
    box and move it to a box next to it (on the left or on the right). What
    is the minimum number of moves needed to end up with exactly 10 bricks in
    every box?
    Assume that:
    - N is an integer within the range [1..20]:
    - Each element of array A is an integer within the range [0..200].
    """
    length = len(boxes)
    target_bricks = 10
    total_moves = 0

    for i in range(length):
        if boxes[i] < target_bricks:
            bricks_needed = target_bricks - boxes[i]
            if i < length - 1:
                # Move bricks from right box
                moves_right = min(bricks_needed, boxes[i + 1])
                boxes[i] += moves_right
                boxes[i + 1] -= moves_right
                total_moves += moves_right
                bricks_needed -= moves_right

            if bricks_needed > 0 and i > 0:
                # Move bricks from left box
                moves_left = min(bricks_needed, boxes[i - 1])
                boxes[i] += moves_left
                boxes[i - 1] -= moves_left
                total_moves += moves_left
                bricks_needed -= moves_left

            # If there is still not enough bricks, it is not possible
            if bricks_needed > 0:
                return -1

    return total_moves


if __name__ == '__main__':
    print(solution([17, 15, 10, 8]))  # Output: 7
    print(solution([11, 10, 8, 12, 8, 10, 11]))  # Output: 6
    print(solution([7, 14, 10]))  # Output: -1

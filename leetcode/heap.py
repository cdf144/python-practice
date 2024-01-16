import heapq
from typing import List


class Solution:
    # 1046. Last Stone Weight
    def lastStoneWeight(self, stones: List[int]) -> int:
        pq = list(map(lambda x: -x, stones))
        heapq.heapify(pq)

        while len(pq) > 1:
            stone_1 = heapq.heappop(pq)
            stone_2 = heapq.heappop(pq)
            if stone_1 == stone_2:
                continue
            heapq.heappush(pq, stone_1 - stone_2)

        return -pq[0] if pq else 0

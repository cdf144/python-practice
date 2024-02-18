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

    # 2402. Meeting Rooms III
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()  # key=lambda m: m[0]
        meetings_held = [0] * n

        available = [i for i in range(n)]
        using = []  # (end_time, room_no)

        for start, end in meetings:
            while using and using[0][0] <= start:
                heapq.heappush(available, heapq.heappop(using)[1])

            room = -1
            if available:
                room = heapq.heappop(available)
                heapq.heappush(using, (end, room))
            else:
                delayed_start, room = heapq.heappop(using)
                duration = end - start
                heapq.heappush(using, (delayed_start + duration, room))
            meetings_held[room] += 1

        return meetings_held.index(max(meetings_held))

import heapq


class MedianFinder:
    def __init__(self):
        self.left_heap = []
        self.right_heap = []
        self.size = 0

    def addNum(self, num: int) -> None:
        if self.left_heap and num > -self.left_heap[0]:
            heapq.heappush(self.right_heap, num)
        else:
            heapq.heappush(self.left_heap, -num)
        self.size += 1

        # Balancing 2 halves
        if len(self.right_heap) > len(self.left_heap) + 1:
            heapq.heappush(self.left_heap, -heapq.heappop(self.right_heap))
        elif len(self.left_heap) > len(self.right_heap) + 1:
            heapq.heappush(self.right_heap, -heapq.heappop(self.left_heap))

    def findMedian(self) -> float:
        if self.size % 2 == 1:
            return float(-self.left_heap[0]) \
                   if len(self.left_heap) > len(self.right_heap) \
                   else float(self.right_heap[0])
        else:
            return (-self.left_heap[0] + self.right_heap[0]) / 2

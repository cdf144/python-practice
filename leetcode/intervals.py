import heapq
from typing import List


class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Solution:
    # 56. Merge Intervals
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda i: i[0])
        result = [intervals[0]]

        for interval in intervals[1:]:
            top = result[-1]
            if top[1] >= interval[0]:
                top[1] = max(top[1], interval[1])
            else:
                result.append(interval)

        return result

    # 57. Insert Interval
    def insert(self, intervals: List[List[int]],
               newInterval: List[int]) -> List[List[int]]:
        n = len(intervals)
        result = []

        i = 0
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        result.append(newInterval)

        while i < n:
            result.append(intervals[i])
            i += 1

        return result

    # 252. Meeting Rooms
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        if not intervals:
            return True
        intervals.sort(key=lambda i: i.start)

        curr_end = intervals[0].end
        for interval in intervals[1:]:
            if curr_end > interval.start:
                return False
            curr_end = interval.end
        return True

    # 253. Meeting Rooms II
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        if not intervals:
            return 0

        heap = []
        for interval in sorted(intervals, key=lambda i: i.start):
            if interval.start == interval.end:
                continue
            # Previous meeting is done in this room and can be reused
            if heap and heap[0] <= interval.start:
                heapq.heappop(heap)
            # Room will be occupied until the end of this meeting
            heapq.heappush(heap, interval.end)

        return len(heap)

    # 435. Non-overlapping Intervals
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda i: i[0])

        # The below space complexities does not count space used to sort
        # and iterate over.

        # # O(n) space, worse time complexity too due to accessing last element
        # erased = [intervals[0]]
        # result = 0
        # for interval in intervals[1:]:
        #     top = erased[-1]
        #     if top[1] > interval[0]:
        #         result += 1
        #         top[1] = min(top[1], interval[1])
        #     else:
        #         erased.append(interval)
        # return result

        # O(1) space
        curr_end = intervals[0][1]
        result = 0
        for interval in intervals[1:]:
            if curr_end > interval[0]:
                result += 1
                curr_end = min(curr_end, interval[1])
            else:
                curr_end = interval[1]
        return result

    # 1288. Remove Covered Intervals
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        # Sort so that for all intervals with the same l value:
        # intervals[i][1] >= intervals[i + 1][1] >= ... >= intervals[i + k][1]
        intervals.sort(key=lambda i: (i[0], -i[1]))
        # No need to consider curr_l because it is guaranteed that:
        # intervals[i][0] <= intervals[i + 1][0] <= ... <= intervals[n][0]
        curr_r = intervals[0][1]
        result = len(intervals)
        for interval in intervals[1:]:
            if curr_r >= interval[1]:
                result -= 1
            else:
                curr_r = interval[1]

        return result

    # 1851. Minimum Interval to Include Each Query
    def minInterval(self, intervals: List[List[int]],
                    queries: List[int]) -> List[int]:
        n = len(intervals)
        intervals.sort(key=lambda inter: inter[0])
        result = {}

        # Heap to store all the intervals that include the current query, sorted
        # by interval size
        heap = []
        i = 0  # index for iterating over intervals
        for query in sorted(queries):
            # Push all intervals that can (possibly) include query
            while i < n and intervals[i][0] <= query:
                interval = intervals[i]
                left, right = interval[0], interval[1]
                heapq.heappush(heap, (right - left + 1, right))
                i += 1
            # Filter the intervals which do not include query
            while heap and heap[0][1] < query:
                heapq.heappop(heap)
            result[query] = heap[0][0] if heap else -1

        return [result[query] for query in queries]

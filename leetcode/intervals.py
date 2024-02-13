from typing import List


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

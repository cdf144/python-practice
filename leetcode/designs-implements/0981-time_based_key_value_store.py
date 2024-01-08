import bisect
import collections


class TimeMap:
    def __init__(self):
        self.values = collections.defaultdict(list)
        self.timestamps = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.values[key].append(value)
        self.timestamps[key].append(timestamp)

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.timestamps:
            return ''

        """
        Manual binary search
        """
        # low, high = 0, len(self.timestamps[key]) - 1
        #
        # while low < high:
        #     mid = low + (high - low) // 2
        #     if self.timestamps[key][mid] > timestamp:
        #         high = mid
        #     else:
        #         low = mid + 1

        """
        Library (bisect)
        """
        low = bisect.bisect(self.timestamps[key], timestamp)

        return self.values[key][low - 1] if low > 0 else ''

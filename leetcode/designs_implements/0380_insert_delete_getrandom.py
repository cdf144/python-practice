import random


class RandomizedSet:
    def __init__(self):
        self.vals = []
        self.map_val_to_idx = {}

    def insert(self, val: int) -> bool:
        if val in self.map_val_to_idx:
            return False

        self.vals.append(val)
        self.map_val_to_idx[val] = len(self.vals) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.map_val_to_idx:
            return False

        idx = self.map_val_to_idx[val]
        # Swap val in our array with the last element.
        # That way, delete in array is O(1).
        self.map_val_to_idx[self.vals[-1]] = idx
        self.vals[idx], self.vals[-1] = self.vals[-1], self.vals[idx]

        del self.map_val_to_idx[val]
        self.vals.pop()
        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)

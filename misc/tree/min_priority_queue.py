import math


class MinPriorityQueue:
    """An implementation of an integer Min Priority Queue (aka. Min Heap)."""

    def __init__(self):
        self.pq = []

    def append(self, num: int) -> None:
        """Insert an integer into the Priority Queue."""
        self.pq.append(num)
        self._swim(len(self.pq) - 1)

    def pop(self) -> int:
        """Remove and return the integer at the top of the queue."""
        top = self.pq[0]

        self.pq[0], self.pq[-1] = self.pq[-1], self.pq[0]
        self.pq.pop()
        self._sink(0)

        return top

    def _sink(self, old_father: int) -> None:
        father = old_father
        l_child = 2 * old_father + 1
        r_child = 2 * old_father + 2

        if l_child < len(self.pq) and self.pq[l_child] < self.pq[father]:
            father = l_child

        if r_child < len(self.pq) and self.pq[r_child] < self.pq[father]:
            father = r_child

        if father != old_father:
            self.pq[father], self.pq[old_father] = (self.pq[old_father],
                                                    self.pq[father])
            self._sink(father)

    def _swim(self, child: int) -> None:
        if child == 0:
            return

        father = child // 2
        if child % 2 == 0:
            father -= 1

        if self.pq[child] < self.pq[father]:
            self.pq[child], self.pq[father] = self.pq[father], self.pq[child]
            self._swim(father)

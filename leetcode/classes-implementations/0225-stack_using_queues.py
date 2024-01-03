import collections


# 225. Implement Stack using Queues
class MyStack:
    queue = None

    def __init__(self):
        self.queue = collections.deque()

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        return self.queue.pop()

    def top(self) -> int:
        return self.queue[len(self.queue) - 1]

    def empty(self) -> bool:
        return len(self.queue) == 0

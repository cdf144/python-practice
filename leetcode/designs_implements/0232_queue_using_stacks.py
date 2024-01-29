class MyQueue:
    def __init__(self):
        self.back = []
        self.front = []

    def push(self, x: int) -> None:
        self.back.append(x)

    def pop(self) -> int:
        self._switch()
        return self.front.pop()

    def peek(self) -> int:
        self._switch()
        return self.front[-1]

    def empty(self) -> bool:
        return not self.back and not self.front

    def _switch(self) -> None:
        if not self.front:
            while self.back:
                self.front.append(self.back.pop())

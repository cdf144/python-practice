class UF:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1 for i in range(n)]

    def _root(self, x: int) -> int:
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def connected(self, x: int, y: int) -> bool:
        return self._root(x) == self._root(y)

    def union(self, x: int, y: int) -> bool:
        x, y = self._root(x), self._root(y)
        if x == y:
            return False

        if self.size[x] < self.size[y]:
            x, y = y, x

        self.parent[y] = x
        self.size[x] += self.size[y]
        return True

    def reset(self, x: int) -> None:
        self.parent[x] = x

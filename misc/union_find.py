class UF:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
        self.size = [1 for i in range(n)]

    def _root(self, p: int) -> int:
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def connected(self, p: int, q: int) -> bool:
        return self._root(p) == self._root(q)

    def union(self, p: int, q: int) -> bool:
        pr, qr = self._root(p), self._root(q)
        if pr == qr:
            return False

        if self.size[pr] < self.size[qr]:
            pr, qr = qr, pr

        self.parent[qr] = pr
        self.size[pr] += self.size[qr]
        return True

    def reset(self, p: int) -> None:
        self.parent[p] = p

from typing import List


class UF:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
        self.size = [1] * n

    def _root(self, p: int) -> int:
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def connected(self, p: int, q: int) -> bool:
        return self._root(p) == self._root(q)

    def union(self, p: int, q: int) -> bool:
        p_root = self._root(p)
        q_root = self._root(q)
        if p_root == q_root:
            return False

        if self.size[p_root] <= self.size[q_root]:
            self.parent[p_root] = q_root
            self.size[q_root] += self.size[p_root]
        else:
            self.parent[q_root] = p_root
            self.size[p_root] += self.size[q_root]

        return True


class Solution:
    # 997. Find the Town Judge
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        out_degree = [0] * n
        in_degree = [0] * n
        for t in trust:
            out_degree[t[0] - 1] += 1
            in_degree[t[1] - 1] += 1

        for i in range(n):
            if out_degree[i] == 0 and in_degree[i] == n - 1:
                return i + 1
        return -1

    # 684. Redundant Connection
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        size = 0
        for edge in edges:
            size = max(size, edge[0], edge[1])

        uf = UF(size)
        result = [0] * 2
        for edge in edges:
            if not uf.connected(edge[0] - 1, edge[1] - 1):
                uf.union(edge[0] - 1, edge[1] - 1)
            else:
                result[0] = edge[0]
                result[1] = edge[1]

        return result
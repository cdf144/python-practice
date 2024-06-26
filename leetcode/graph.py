import collections
import bisect
import math
from typing import List, Set


class UF:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]
        self.count_edges = 0

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
        self.count_edges += 1
        return True

    def reset(self, x: int) -> None:
        self.parent[x] = x


class Solution:
    # 200. Number of Islands
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        result = 0
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        visited = set()

        def dfs(i: int, j: int) -> None:
            if (
                not 0 <= i < m
                or not 0 <= j < n
                or grid[i][j] == "0"
                or (i, j) in visited
            ):
                return
            visited.add((i, j))
            for d in dirs:
                dfs(i + d[0], j + d[1])

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == "1" and (i, j) not in visited:
                    dfs(i, j)
                    result += 1

        return result

    # 310. Minimum Height Trees
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        for e in edges:
            assert len(e) == 2

        graph = collections.defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)

        leaves = []
        for v, adj in graph.items():
            if len(adj) == 1:
                leaves.append(v)

        while n > 2:
            n -= len(leaves)
            next_leaves = []
            for leaf in leaves:
                parent = graph[leaf].pop()
                graph[parent].remove(leaf)
                if len(graph[parent]) == 1:
                    next_leaves.append(parent)
            leaves = next_leaves

        return leaves

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

    # 752. Open the Lock
    def openLock(self, deadends: List[str], target: str) -> int:
        # View each state of the lock as a node in a tree, and each edge represent a
        # wheel turn.
        target_tuple = tuple(int(c) for c in target)
        visited = set([tuple(int(c) for c in d) for d in deadends])
        if target_tuple in visited or (0, 0, 0, 0) in visited:
            return -1

        queue = collections.deque()
        state = (0, 0, 0, 0)

        queue.append(state)
        visited.add(state)
        turns = 0
        while queue:
            for _ in range(len(queue)):
                state = queue.popleft()
                if state == target_tuple:
                    return turns
                for i in range(4):
                    for delta in [-1, 1]:
                        new_state = list(state)
                        new_state[i] = (new_state[i] + delta) % 10
                        new_state = tuple(new_state)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append(new_state)
            turns += 1

        return -1

    # 787. Cheapest Flights Within K Stops
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        # BFS, Greedy-ish
        adj = [[] for _ in range(n)]
        for u, v, price in flights:
            adj[u].append((v, price))

        dist_to = [math.inf for _ in range(n)]
        dist_to[src] = 0
        queue = collections.deque()

        queue.append((src, 0))
        stops = 0
        while queue and stops <= k:
            for _ in range(len(queue)):
                u, cost = queue.popleft()
                for v, price in adj[u]:
                    if dist_to[v] <= cost + price:
                        continue
                    dist_to[v] = cost + price
                    queue.append((v, dist_to[v]))
            stops += 1

        result = dist_to[dst]
        return result if isinstance(result, int) else -1

    # 834. Sum of Distances in Tree
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        adj = collections.defaultdict(list)
        res = [0] * n
        # Number of nodes in subtree with root as i
        count = [1] * n

        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        def dfs_init(node: int, parent: int = -1) -> None:
            """
            Postorder traversal to compute count[i] (node count of each subtree)
            and res[root]
            """
            for child in adj[node]:
                if child == parent:
                    continue
                dfs_init(child, node)
                count[node] += count[child]
                res[node] += res[child] + count[child]

        def dfs_main(node: int, parent: int = -1) -> None:
            """
            Preorder traversal to compute res[i]
            """
            for child in adj[node]:
                if child == parent:
                    continue
                res[child] = res[node] - count[child] + (n - count[child])
                dfs_main(child, node)

        dfs_init(0)
        dfs_main(0)
        return res

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

    # 1061. Lexicographically Smallest Equivalent String
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        # Custom UF that each character's root is the lexicographically smallest
        # character in the group.
        parent = [i for i in range(26)]

        def root(p: int) -> int:
            while p != parent[p]:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        def union(p: int, q: int) -> None:
            pr, qr = root(p), root(q)
            if pr < qr:
                parent[qr] = pr
            else:
                parent[pr] = qr

        for c1, c2 in zip(s1, s2):
            union(ord(c1) - ord("a"), ord(c2) - ord("a"))

        result = []
        for c in baseStr:
            result.append(chr(ord("a") + root(ord(c) - ord("a"))))

        return "".join(result)

    # 1579. Remove Max Number of Edges to Keep Graph Fully Traversable
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice, bob = UF(n), UF(n)
        edges_required = 0

        # Greedy, use edges that benefit both Alice and Bob first, then individually.
        for type, x, y in sorted(edges, reverse=True):
            x = x - 1
            y = y - 1
            if type == 3:  # Alice and Bob
                # Use `|` instead of `or` because with `or`, due to short-circuiting,
                # if the first expression is `True`, the second expression won't be
                # evaluated.
                if alice.union(x, y) | bob.union(x, y):
                    edges_required += 1
            elif type == 2:  # Bob
                if bob.union(x, y):
                    edges_required += 1
            else:  # Alice
                if alice.union(x, y):
                    edges_required += 1

        # In the end, both Alice and Bob graph should be MSTs
        return (
            len(edges) - edges_required
            if alice.count_edges == n - 1 and bob.count_edges == n - 1
            else -1
        )

    # 1791. Find Center of Star Graph
    def findCenter(self, edges: List[List[int]]) -> int:
        return edges[0][0] if edges[0][0] in edges[1] else edges[0][1]

    # 1971. Find if Path Exists in Graph
    def validPath(
        self, n: int, edges: List[List[int]], source: int, destination: int
    ) -> bool:
        uf = UF(n)
        for e in edges:
            uf.union(e[0], e[1])
        return uf.connected(source, destination)

    # 1992. Find All Groups of Farmland
    def findFarmland(self, land: List[List[int]]) -> List[List[int]]:
        m = len(land)
        n = len(land[0])
        result = []

        def find_corner(i: int, j: int, corner: List[int]) -> None:
            """Find bottom right corner of current group (rectangular farmland areas)"""
            if i >= m or j >= n or land[i][j] != 1:
                return
            # Mark cell as visited
            land[i][j] = 2
            corner[0] = max(corner[0], i)
            corner[1] = max(corner[1], j)
            find_corner(i + 1, j, corner)
            find_corner(i, j + 1, corner)

        for i, row in enumerate(land):
            for j, cell in enumerate(row):
                if cell != 1:
                    continue
                # Bottom right corner
                corner = [i, j]
                find_corner(i, j, corner)
                result.append([i, j] + corner)

        return result

    # 2092. Find All People With Secret
    def findAllPeople(
        self, n: int, meetings: List[List[int]], firstPerson: int
    ) -> List[int]:
        meetings.sort(key=lambda m: m[2])
        k = len(meetings)

        uf = UF(n)
        uf.union(0, firstPerson)
        i = 0
        while i < k:
            people = set()
            time = meetings[i][2]
            while i < k and time == meetings[i][2]:
                meeting = meetings[i]
                p, q = meeting[0], meeting[1]
                people.add(p)
                people.add(q)
                uf.union(p, q)
                i += 1
            for p in people:
                if not uf.connected(0, p):
                    uf.reset(p)

        return [i for i in range(n) if uf.connected(0, i)]

    # 2192. All Ancestors of a Node in a Directed Acyclic Graph
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        result = [[] for _ in range(n)]

        adj = collections.defaultdict(list)
        for u, v in edges:
            adj[u].append(v)

        def dfs(u: int, ancestor: int, visited: Set[int]) -> None:
            for v in adj[u]:
                if v in visited:
                    continue
                result[v].append(ancestor)
                visited.add(v)
                dfs(v, ancestor, visited)

        for i in range(n):
            dfs(i, i, set())
        return result

    # 2709. Greatest Common Divisor Traversal
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        def generate_primes() -> List[int]:
            prime_range = int(math.sqrt(10**5))
            is_prime = [True] * (prime_range + 1)
            is_prime[0], is_prime[1] = False, False

            for i in range(2, prime_range + 1):
                if is_prime[i]:
                    j = 2 * i
                    while j <= prime_range:
                        is_prime[j] = False
                        j += i

            result = []
            for i, prime in enumerate(is_prime):
                if prime:
                    result.append(i)
            return result

        def factorize(number: int) -> List[int]:
            nonlocal primes
            factors = []
            for prime in primes:
                if number % prime == 0:
                    factors.append(prime)
                    while number % prime == 0:
                        number //= prime
            # If the prime factorization of `number` includes a large prime
            # which is not included in our pre-computed prime list
            # For example: 20014 = 2 * 10007
            if number > 1:
                factors.append(number)
            return factors

        n = len(nums)
        if n == 1:
            return True
        primes = generate_primes()
        uf = UF(n)
        # maps prime to index of first element that the prime is a factor of
        prime_idx = {}

        for i, num in enumerate(nums):
            if num == 1:
                return False
            facts = factorize(num)
            for fact in facts:
                if fact in prime_idx:
                    uf.union(i, prime_idx[fact])
                else:
                    prime_idx[fact] = i

        return any(uf.size[i] == n for i in range(n))

    # 2812. Find the Safest Path in a Grid
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        self.dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        n = len(grid)
        grid_safeness = self._get_dist_to_nearest_thief(grid)

        def has_valid_path(min_safeness: int) -> bool:
            if grid_safeness[0][0] < min_safeness:
                return False

            queue = collections.deque([(0, 0)])
            visited = {(0, 0)}

            while queue:
                for _ in range(len(queue)):
                    i, j = queue.popleft()
                    if i == n - 1 and j == n - 1:
                        return True
                    for di, dj in self.dirs:
                        x = i + di
                        y = j + dj
                        if (
                            not 0 <= x < n
                            or not 0 <= y < n
                            or (x, y) in visited
                            or grid_safeness[x][y] < min_safeness
                        ):
                            continue
                        queue.append((x, y))
                        visited.add((x, y))

            return False

        return (
            bisect.bisect_left(
                range(n * 2), True, key=lambda mid: not has_valid_path(mid)
            )
            - 1
        )

    def _get_dist_to_nearest_thief(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Do a BFS starting from every thief to find out the nearest distance of every
        cell to a thief.
        """
        n = len(grid)
        result = [[0] * n for _ in range(n)]

        queue = collections.deque()
        visited = set()
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == 1:
                    queue.append((i, j))
                    visited.add((i, j))

        dist = 0
        while queue:
            for _ in range(len(queue)):
                i, j = queue.popleft()
                result[i][j] = dist
                for di, dj in self.dirs:
                    x = i + di
                    y = j + dj
                    if not 0 <= x < n or not 0 <= y < n or (x, y) in visited:
                        continue
                    queue.append((x, y))
                    visited.add((x, y))
            dist += 1

        return result

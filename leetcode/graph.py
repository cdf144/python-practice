import collections
import math
from typing import List


class UF:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

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


class Solution:
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

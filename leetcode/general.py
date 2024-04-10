import collections
from typing import List


class Solution:
    # 950. Reveal Cards In Increasing Order
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        n = len(deck)
        deck.sort()
        queue = collections.deque(range(n))
        result = [0] * n

        # Simulate card revealing process
        for i in range(n):
            # Reveal card
            result[queue.popleft()] = deck[i]
            # Move next card to bottom
            if queue:
                queue.append(queue.popleft())

        return result

    # 1291. Sequential Digits
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        # # Recursion
        # result = []

        # def dfs(s: str) -> None:
        #     if low <= int(s) <= high:
        #         result.append(int(s))
        #     elif int(s) > high:
        #         return
        #     if int(s[-1]) < 9:
        #         dfs(s + str(int(s[-1]) + 1))

        # for i in range(1, 9):
        #     dfs(str(i))

        # result.sort()
        # return result

        # Sliding Window
        s = "123456789"
        min_len = len(str(low))
        max_len = len(str(high))

        result = []
        for length in range(min_len, max_len + 1):
            for left in range(0, 10 - length):
                num = int(s[left : left + length])
                if low <= num <= high:
                    result.append(num)

        return result

    # 2073. Time Needed to Buy Tickets
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        result = 0
        k_person = tickets[k]
        for person in tickets[:k]:
            result += min(person, k_person)
        for person in tickets[k + 1 :]:
            result += min(person, k_person - 1)
        return result + k_person

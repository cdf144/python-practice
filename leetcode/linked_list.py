import heapq
from typing import List, Optional


class Node:
    def __init__(
        self, x: int = 0, n: Optional["Node"] = None, random: Optional["Node"] = None
    ):
        self.val = int(x)
        self.next = n
        self.random = random


class ListNode:
    def __init__(self, val: int = 0, n: Optional["ListNode"] = None):
        self.val = val
        self.next = n


class Solution:
    # 2. Add Two Numbers
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        head = ListNode()
        itr = head
        carry = 0

        while True:
            s = carry
            if l1:
                s += l1.val
                l1 = l1.next
            if l2:
                s += l2.val
                l2 = l2.next

            itr.val = s % 10
            carry = s // 10
            if not l1 and not l2 and carry == 0:
                break

            itr.next = ListNode()
            itr = itr.next

        return head

    # 19. Remove Nth Node From End of List
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head:
            return None
        dummy = ListNode(-1, head)
        # The distance between slow and fast == n
        slow, fast = dummy, dummy

        for _ in range(n + 1):
            if fast:
                fast = fast.next

        while slow.next and fast:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next if slow.next else None
        return dummy.next

    # 23. Merge K Sorted Lists
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # # Simple but inefficient Heapsort and create new Linked List
        # # Time: O(N*K*log(N*K))
        # # Space: O(N*K)
        # dummy = ListNode()
        # itr = dummy
        # heap = []
        #
        # for linkedlist in lists:
        #     while linkedlist:
        #         heapq.heappush(heap, linkedlist.val)
        #         linkedlist = linkedlist.next
        #
        # while heap:
        #     itr.next = ListNode(heapq.heappop(heap))
        #     itr = itr.next
        #
        # return dummy.next

        # Optimized
        # Time: O(N*K*log(K))
        # Space: O(K)
        heap = []
        # Since tuple comparison breaks for (priority, obj) if priorities are
        # equal and the objects do not have a default comparison order, we
        # have to add an entry count as a tie-breaker.
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, (head.val, i, head))

        dummy = ListNode()
        itr = dummy
        while heap:
            _, entry, node = heapq.heappop(heap)
            itr.next = node
            itr = itr.next
            if node.next:
                heapq.heappush(heap, (node.next.val, entry, node.next))

        return dummy.next

    # 25. Reverse Nodes in k-Group
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def get_kth_node(node: Optional[ListNode], k: int):
            while node and k > 0:
                node = node.next
                k -= 1
            return node

        dummy = ListNode(0, head)
        # prev_group is the last node of the previous group after being reversed
        prev_group = dummy

        while True:
            kth_node = get_kth_node(prev_group, k)
            if not kth_node:
                break
            # next_group points to first node of next group
            next_group = kth_node.next

            # After reversing, head of group points to next_group, and last node
            # in group (or the kth node) gets pointed to by prev_group
            prev = kth_node.next
            original_group_head = curr = prev_group.next
            while curr != next_group:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt

            prev_group.next = prev
            prev_group = original_group_head

        return dummy.next

    # 61. Rotate List
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head

        tail = head
        length = 0
        while tail.next:
            tail = tail.next
            length += 1

        last = length - (k % length)
        tail.next = head
        for _ in range(last):
            tail = tail.next

        new_head = tail.next
        tail.next = None
        return new_head

    # 82. Remove Duplicates from Sorted List II
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        itr = dummy

        while head:
            while head.next and head.val == head.next.val:
                head = head.next
            if itr.next != head:
                itr.next = head.next
            else:
                itr = itr.next
            head = head.next

        return dummy.next

    # 138. Copy List with Random Pointer
    # map_old_to_copy = {}
    def copyRandomList(self, head: "Optional[Node]") -> "Optional[Node]":
        # # Recursive
        # if not head:
        #     return None
        # if head in self.map_old_to_copy:
        #     return self.map_old_to_copy[head]
        #
        # new_head = Node(head.val)
        # self.map_old_to_copy[head] = new_head
        # new_head.next = self.copyRandomList(head.next)
        # new_head.random = self.copyRandomList(head.random)
        #
        # return new_head

        # Iterative
        map_old_to_copy = {}
        map_old_to_copy[None] = None
        itr = head

        while itr:
            copy = Node(itr.val)
            map_old_to_copy[itr] = copy
            itr = itr.next

        itr = head
        while itr:
            map_old_to_copy[itr].next = map_old_to_copy[itr.next]
            map_old_to_copy[itr].random = map_old_to_copy[itr.random]
            itr = itr.next

        return map_old_to_copy[head]

    # 141. Linked List Cycle
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        tortoise = hare = head

        while tortoise and hare and hare.next:
            tortoise = tortoise.next
            hare = hare.next.next
            if tortoise == hare:
                return True

        return False

    # 142. Linked List Cycle II
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        tortoise, hare = head, head

        while tortoise and hare and hare.next:
            tortoise = tortoise.next
            hare = hare.next.next
            if tortoise == hare:
                tortoise = head
                while tortoise != hare:
                    assert tortoise and hare
                    tortoise = tortoise.next
                    hare = hare.next
                return tortoise

        return None

    # 143. Reorder List
    def reorderList(self, head: Optional[ListNode]) -> None:
        slow = fast = head
        while fast and fast.next:
            assert slow
            slow = slow.next
            fast = fast.next.next
        assert slow
        if fast:
            slow = slow.next

        slow = self.reverseList(slow)

        def merge_list(head1: Optional[ListNode], head2: Optional[ListNode]) -> None:
            while head2:
                assert head1
                next_1, next_2 = head1.next, head2.next
                head1.next = head2
                head2.next = next_1
                head1, head2 = next_1, next_2

            assert head1
            head1.next = None

        merge_list(head, slow)

    # 206. Reverse Linked List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        prev, curr = None, head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        return prev

    # 234. Palindrome Linked List
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        assert head

        # slow will reach the 2nd half of linked list
        slow = fast = head
        while fast and fast.next:
            assert slow
            slow = slow.next
            fast = fast.next.next
        assert slow
        if fast:
            slow = slow.next

        slow = self.reverseList(slow)
        while slow:
            assert head
            if slow.val != head.val:
                return False
            slow = slow.next
            head = head.next

        return True

    # 237. Delete Node in a Linked List
    def deleteNode(self, node: ListNode) -> None:
        assert node and node.next
        node.val = node.next.val
        node.next = node.next.next

    # 287. Find the Duplicate Number
    def findDuplicate(self, nums: List[int]) -> int:
        start = nums[0]

        slow = nums[start]
        fast = nums[nums[start]]
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]

        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]

        return slow

    # 876. Middle of the Linked List
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next
        return slow

    # 1171. Remove Zero Sum Consecutive Nodes from Linked List
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        assert head

        # O(n^2)
        # new_head = itr = head
        # running_sum = 0
        # while itr:
        #     running_sum += itr.val
        #     itr = itr.next
        #     if running_sum == 0:
        #         new_head = itr
        #
        # if new_head and new_head.next:
        #     new_head.next = self.removeZeroSumSublists(new_head.next)
        # return new_head

        # O(n) with prefix-sum
        dummy = ListNode(0, head)
        itr = dummy
        prefix_sum_to_node = {}
        prefix_sum = 0
        while itr:
            prefix_sum += itr.val
            prefix_sum_to_node[prefix_sum] = itr
            itr = itr.next

        prefix_sum = 0
        itr = dummy
        while itr:
            # If 2 nodes have the same prefix-sum, that means
            # the nodes between them sum to zero
            prefix_sum += itr.val
            itr.next = prefix_sum_to_node[prefix_sum].next
            itr = itr.next

        return dummy.next

    # 1669. Merge In Between Linked Lists
    def mergeInBetween(
        self, list1: ListNode, a: int, b: int, list2: ListNode
    ) -> ListNode:
        itr = list1
        for _ in range(a - 1):
            assert itr
            itr = itr.next
        assert itr
        node_before_a = itr

        for _ in range(b - a + 1):
            assert itr
            itr = itr.next
        assert itr
        node_b = itr

        itr = list2
        while itr and itr.next:
            itr = itr.next
        list2_end = itr

        node_before_a.next = list2
        list2_end.next = node_b.next
        node_b.next = None
        return list1

    # 2807. Insert Greatest Common Divisors in Linked List
    def insertGreatestCommonDivisors(
        self, head: Optional[ListNode]
    ) -> Optional[ListNode]:
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        itr = head
        while itr and itr.next:
            next = itr.next
            middle = ListNode(gcd(itr.val, next.val), next)
            itr.next = middle
            itr = next

        return head

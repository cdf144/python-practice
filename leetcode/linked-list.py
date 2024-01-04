import heapq
from typing import Optional, List


class ListNode:
    def __init__(self, val=0, n=None):
        self.val = val
        self.next = n


class Node:
    def __init__(self, x: int, n: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = n
        self.random = random


class Solution:
    # 2. Add Two Numbers
    def addTwoNumbers(self, l1: Optional[ListNode],
                      l2: Optional[ListNode]) -> Optional[ListNode]:
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

    # 138. Copy List with Random Pointer
    # map_old_to_copy = {}
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        Recursive
        """
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

        """
        Iterative
        """
        map_old_to_copy = {None: None}
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

    # 206. Reverse Linked List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr, nxt = None, head, None

        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        return prev

    # 234. Palindrome Linked List
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        def reverse_llist(h: Optional[ListNode]) -> Optional[ListNode]:
            prev, curr, nxt = None, h, None

            while curr:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt

            return prev

        if not head.next:
            return True

        # Slow will reach the start of the 2nd half of llist
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if fast:
            slow = slow.next

        slow = reverse_llist(slow)
        while slow:
            if slow.val != head.val:
                return False
            slow = slow.next
            head = head.next

        return True

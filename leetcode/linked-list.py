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

    # 19. Remove Nth Node From End of List
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) \
            -> Optional[ListNode]:
        def list_length(h: Optional[ListNode]):
            result = 0
            ptr = h
            while itr is not None:
                result += 1
                ptr = ptr.next
            return result

        length = list_length(head)
        if n == length:
            return head.next
        if length == 1:
            return None

        itr = head
        n = length - n
        while n > 1:
            itr = itr.next
            n -= 1

        nxt = itr.next.next
        itr.next = nxt
        return head

    # 61. Rotate List
    def rotateRight(self, head: Optional[ListNode],
                    k: int) -> Optional[ListNode]:
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

    # 143. Reorder List
    def reorderList(self, head: Optional[ListNode]) -> None:
        # Slow is pointer to middle of Linked List
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if fast:
            slow = slow.next

        """
        Stack, O(N) space
        """
        # stack = []
        # while slow:
        #     stack.append(slow)
        #     slow = slow.next

        # slow = head
        # while stack:
        #     node = stack.pop()
        #     nxt = slow.next
        #     slow.next = node
        #     node.next = nxt
        #     slow = slow.next.next
        # slow.next = None

        """
        Reverse 2nd half of Linked List then merge, O(1) space
        """

        def reverse(old_head: Optional[ListNode]) -> Optional[ListNode]:
            prev, curr = None, old_head

            while curr:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt

            return prev

        def merge(head1: Optional[ListNode], head2: Optional[List]) -> None:
            while head2:
                nxt_1 = head1.next
                nxt_2 = head2.next

                head1.next = head2
                head2.next = nxt_1

                head1 = nxt_1
                head2 = nxt_2

            head1.next = None

        slow = reverse(slow)
        merge(head, slow)

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

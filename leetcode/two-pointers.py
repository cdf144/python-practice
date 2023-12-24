import math
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, n=None):
        self.val = val
        self.next = n


class Solution:
    # 11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        left = 0
        right = len(height) - 1

        while left < right:
            min_h = min(height[left], height[right])
            max_area = max(max_area, min_h * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area

    # 15. 3Sum
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        length = len(nums)
        if length < 3:
            return []

        nums.sort()
        result = []
        for i in range(length - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            x = nums[i]
            low = i + 1
            high = length - 1
            while low < high:
                two_sum = nums[low] + nums[high]
                if two_sum == -x:
                    result.append([x, nums[low], nums[high]])
                    low += 1
                    high -= 1
                    while low < high and nums[low] == nums[low - 1]:
                        low += 1
                    while low < high and nums[high] == nums[high + 1]:
                        high -= 1
                elif two_sum < -x:
                    low += 1
                else:
                    high -= 1

        return result

    # 16. 3Sum Closest
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        length = len(nums)
        closest_sum = math.inf
        nums.sort()

        for i in range(length - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = length - 1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if s == target:
                    return target
                elif s < target:
                    left += 1
                else:
                    right -= 1

                if abs(closest_sum - target) > abs(s - target):
                    closest_sum = s

        return closest_sum

    # 18. 4Sum
    def three_sum(self, nums: List[int], length: int,
                  low: int, target: int) -> List[List[int]]:
        result = []
        for i in range(length - low):
            if i > 0 and nums[low + i] == nums[low + i - 1]:
                continue
            x = nums[low + i]
            t = target - x
            left, right = i + 1, length - low - 1
            while left < right:
                summ = nums[low + left] + nums[low + right]
                if summ == t:
                    result.append([x, nums[low + left], nums[low + right]])
                    left += 1
                    right -= 1
                    while (left < right
                           and nums[low + left] == nums[low + left - 1]):
                        left += 1
                    while (left < right
                           and nums[low + right] == nums[low + right + 1]):
                        right -= 1
                elif summ < t:
                    left += 1
                else:
                    right -= 1

        return result

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        length = len(nums)
        if length < 4:
            return []

        nums.sort()
        result = []
        for i in range(length - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            a = nums[i]
            three_sum_results = self.three_sum(nums, length, i + 1,
                                               target - a)
            for res in three_sum_results:
                result.append([a] + res)

        return result

    # 42. Trapping Rain Water
    def trap(self, height: List[int]) -> int:
        """
        Auxiliary array
        """
        # length = len(height)
        # max_left_h, max_right_h = [0] * length, [0] * length
        #
        # max_h = 0
        # for i, h in enumerate(height):
        #     max_left_h[i] = max_h
        #     max_h = max(max_h, h)
        #
        # max_h = 0
        # for i, h in enumerate(reversed(height)):
        #     max_right_h[-i - 1] = max_h
        #     max_h = max(max_h, h)
        #
        # total_water = 0
        # for i, h in enumerate(height):
        #     water = min(max_left_h[i], max_right_h[i]) - h
        #     if water > 0:
        #         total_water += water
        #
        # return total_water

        """
        Two pointers
        """
        left, right = 0, len(height) - 1
        l_max, r_max = 0, 0
        total_water = 0

        while left < right:
            l_height, r_height = height[left], height[right]
            if l_height < r_height:
                if l_max >= l_height:
                    total_water += l_max - l_height
                else:
                    l_max = l_height
                left += 1
            else:
                if r_max >= r_height:
                    total_water += r_max - r_height
                else:
                    r_max = r_height
                right -= 1

        return total_water

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

    # 75. Sort Colors
    def sortColors(self, nums: List[int]) -> None:
        """
        Counting sort
        """
        # count = [0] * 3
        # for num in nums:
        #     count[num] += 1
        #
        # itr = 0
        # for i, cnt in enumerate(count):
        #     for _ in range(cnt):
        #         nums[itr] = i
        #         itr += 1

        """
        Two pointers
        """
        left, right = 0, len(nums) - 1
        i = 0
        while i <= right:
            if nums[i] == 0:
                nums[left], nums[i] = 0, nums[left]
                left += 1
                i += 1
            elif nums[i] == 2:
                nums[right], nums[i] = 2, nums[right]
                right -= 1
            else:
                i += 1

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

    # 142. Linked List Cycle II
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        tortoise, hare = head, head

        while hare and hare.next:
            tortoise = tortoise.next
            hare = hare.next.next
            if tortoise == hare:
                tortoise = head
                while tortoise != hare:
                    tortoise = tortoise.next
                    hare = hare.next
                return tortoise

        return None

    # 160. Intersection of Two Linked Lists
    def getIntersectionNode(self, headA: ListNode,
                            headB: ListNode) -> Optional[ListNode]:
        a, b = headA, headB
        while a != b:
            a = a.next if a else headB.next
            b = b.next if b else headA.next
        return a

    # 167. Two Sum II - Input Array Is Sorted
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        length = len(numbers)
        for i, x in enumerate(numbers):
            key = target - x
            low, high = i + 1, length - 1
            while low <= high:
                mid = low + (high - low) // 2
                number = numbers[mid]
                if number == key:
                    return [i + 1, mid + 1]
                elif number < key:
                    low = mid + 1
                else:
                    high = mid - 1
        return []

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

    # 392. Is Subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        length = len(s)
        if length == 0:
            return True

        k = 0
        for c in t:
            if s[k] == c:
                k += 1
                if k == length:
                    return True

        return False

    # 977. Squares of a Sorted Array
    def sortedSquares(self, nums: List[int]) -> List[int]:
        length = len(nums)
        result = [0] * length

        left, right = 0, length - 1
        i = length - 1
        while left <= right:
            if abs(nums[right]) >= abs(nums[left]):
                result[i] = nums[right] ** 2
                right -= 1
            else:
                result[i] = nums[left] ** 2
                left += 1
            i -= 1

        return result

    # 1913. Maximum Product Difference Between Two Pairs
    def maxProductDifference(self, nums: List[int]) -> int:
        max1 = -math.inf
        max2 = -math.inf
        min1 = math.inf
        min2 = math.inf

        for num in nums:
            if max1 < num:
                max2 = max1
                max1 = num
            elif max2 < num:
                max2 = num

            if min1 > num:
                min2 = min1
                min1 = num
            elif min2 > num:
                min2 = num

        return max1*max2 - min1*min2

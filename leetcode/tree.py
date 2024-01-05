import collections
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p or not q:
            return p == q

        """
        BFS + List
        """
        # def bfs(x: Optional[TreeNode]) -> List[int]:
        #     queue = collections.deque()
        #     result = [x.val]
        #     queue.append(x)
        #
        #     while queue:
        #         node = queue.pop()
        #
        #         if node.left:
        #             result.append(node.left.val)
        #             queue.append(node.left)
        #         else:
        #             result.append(10 ** 4 + 1)
        #
        #         if node.right:
        #             result.append(node.right.val)
        #             queue.append(node.right)
        #         else:
        #             result.append(10 ** 4 + 1)
        #
        #     return result
        #
        # return bfs(p) == bfs(q)

        """
        DFS in-place check
        """
        return (
            p.val == q.val
            and self.isSameTree(p.left, q.left)
            and self.isSameTree(p.right, q.right)
        )

import collections
from typing import Optional, List, Generator


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # 100. Same Tree
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

    # 104. Maximum Depth of Binary Tree
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        Recursive DFS
        """
        # if not root:
        #     return 0
        # return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

        """
        Iterative DFS
        """
        stack = [(root, 1)]
        max_depth = 0
        while stack:
            node, depth = stack.pop()
            if node:
                max_depth = max(max_depth, depth)
                stack.append((node.right, 1 + depth))
                stack.append((node.left, 1 + depth))

        return max_depth

    # 872. Leaf-Similar Trees
    def leafSimilar(self, root1: Optional[TreeNode],
                    root2: Optional[TreeNode]) -> bool:
        """
        Append to List
        """
        # def dfs(root: Optional[TreeNode], leaf: List[int]) -> None:
        #     if root:
        #         if not root.left and not root.right:
        #             leaf.append(root.val)
        #             return
        #         dfs(root.left, leaf)
        #         dfs(root.right, leaf)
        #
        # leaf1, leaf2 = [], []
        # dfs(root1, leaf1)
        # dfs(root2, leaf2)
        # return leaf1 == leaf2

        """
        Generator
        """
        def dfs(root: Optional[TreeNode]) -> Generator[int, TreeNode, None]:
            if root:
                if not root.left and not root.right:
                    yield root.val
                    return

                yield from dfs(root.left)
                yield from dfs(root.right)

        return list(dfs(root1)) == list(dfs(root2))

    # 938. Range Sum of BST
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        """
        BFS
        """
        # queue = collections.deque()
        # result = 0
        # queue.append(root)
        #
        # while queue:
        #     node = queue.popleft()
        #     if not node:
        #         continue
        #     if node.val > high:
        #         queue.append(node.left)
        #     elif node.val < low:
        #         queue.append(node.right)
        #     else:
        #         result += node.val
        #         queue.append(node.left)
        #         queue.append(node.right)
        #
        # return result

        """
        DFS
        """
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        return (
            root.val
            + self.rangeSumBST(root.left, low, high)
            + self.rangeSumBST(root.right, low, high)
        )

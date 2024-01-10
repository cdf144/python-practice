import collections
from typing import Optional, List, Generator, Dict


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

    # 102. Binary Tree Level Order Traversal
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        queue = collections.deque()
        result = []

        if root:
            queue.append(root)
        while queue:
            curr_level = []

            for _ in range(len(queue)):
                node = queue.popleft()
                curr_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(curr_level)

        return result

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

    # 106. Construct Binary Tree from Inorder and Postorder Traversal
    def buildTree(self, inorder: List[int],
                  postorder: List[int]) -> Optional[TreeNode]:
        # Some insights:
        # With Postorder traversal, the root node is always the last node to be
        # visited.
        # With Inorder traversal, the nodes on the left of a node (or in other
        # words, the nodes which are visited before a node) are all in the left
        # subtree of that node.

        # # Simple but inefficient O(n^2) time and extra space
        # if not inorder:
        #     # No need to check if postorder is empty, because due to how this
        #     # algorithm runs, postorder list will always run out last.
        #     return None
        #
        # root = TreeNode(postorder.pop())
        # root_index = inorder.index(root.val)
        #
        # root.right = self.buildTree(inorder[root_index + 1:], postorder)
        # root.left = self.buildTree(inorder[:root_index], postorder)
        #
        # return root

        # Optimized O(n) time and extra space using HashMap
        map_inorder_indices = {}
        for i, node_val in enumerate(inorder):
            map_inorder_indices[node_val] = i

        def build(low: int, high: int) -> Optional[TreeNode]:
            """
            Recursively rebuild a Binary Tree.
            :param low: Inclusive lower bound of the working area in
                inorder list.
            :param high: Inclusive higher bound of the working area in
                inorder list.
            :return: A built node with defined left and right subtree.
            """
            if low > high:
                return None

            root = TreeNode(postorder.pop())
            root_index = map_inorder_indices[root.val]

            root.right = build(root_index + 1, high)
            root.left = build(low, root_index - 1)

            return root

        return build(0, len(inorder) - 1)


    # 226. Invert Binary Tree
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        BFS
        """
        # queue = collections.deque()
        # if root:
        #     queue.append(root)
        #
        # while queue:
        #     node = queue.popleft()
        #     node.left, node.right = node.right, node.left
        #     if node.left:
        #         queue.append(node.left)
        #     if node.right:
        #         queue.append(node.right)
        #
        # return root

        """
        DFS
        """
        if not root:
            return

        left = root.left
        right = root.right

        root.left = self.invertTree(right)
        root.right = self.invertTree(left)

        return root

    # 235. Lowest Common Ancestor of a Binary Search Tree
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        if root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        return root

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

    # 2385. Amount of Time for Binary Tree to Be Infected
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        # Main idea: Convert to an undirected graph, then use BFS to find the
        # max distance between 'start' node and other nodes in the graph.
        def convert_to_graph(r: Optional[TreeNode]) -> Dict[int, List[int]]:
            """
            Convert from binary tree to a Map between a Node and Adjacency List
            :param r: root node
            :return: a Map (Dict) of Nodes and its Adjacency List
            """
            g = collections.defaultdict(list)
            q = collections.deque()

            q.append(r)
            while q:
                node = q.popleft()
                if node.left:
                    g[node.val].append(node.left.val)
                    g[node.left.val].append(node.val)
                    q.append(node.left)
                if node.right:
                    g[node.val].append(node.right.val)
                    g[node.right.val].append(node.val)
                    q.append(node.right)

            return g

        max_dist = -1
        graph = convert_to_graph(root)
        queue = collections.deque()
        visited = set()

        queue.append(start)
        visited.add(start)
        while queue:
            max_dist += 1
            for _ in range(len(queue)):
                # pop every node that has the previous distance to 'start'
                # and append nodes with next (+1) distance
                v = queue.popleft()
                visited.add(v)
                for adj in graph[v]:
                    if adj in visited:
                        continue
                    queue.append(adj)
                    visited.add(adj)

        return max_dist

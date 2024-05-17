import collections
import hashlib
import math
from typing import Dict, Generator, List, Optional, Tuple


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # 98. Validate Binary Search Tree
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # # 1st idea: Inorder traversal of a valid binary tree results in a
        # # strictly increasing list.
        # node_list = []
        #
        # def inorder(node: Optional[TreeNode]) -> None:
        #     if not node:
        #         return
        #     inorder(node.left)
        #     node_list.append(node.val)
        #     inorder(node.right)
        #
        # inorder(root)
        # return all(x < y for x, y in zip(node_list, node_list[1:]))

        # 2nd idea: In a valid BST, every node's left children are less than
        # itself, while its right children are greater.
        def dfs(node: Optional[TreeNode], min_node, max_node) -> bool:
            if not node:
                return True
            if not min_node < node.val < max_node:
                return False

            return dfs(node.left, min_node, node.val) and dfs(
                node.right, node.val, max_node
            )

        return dfs(root, float("-inf"), float("inf"))

    # 100. Same Tree
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p or not q:
            return p == q

        # # BFS, list check
        # def bfs(x: Optional[TreeNode]) -> List[int]:
        #     result = [x.val]
        #     queue = collections.deque()
        #     queue.append(x)
        #
        #     while queue:
        #         node = queue.popleft()
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

        # DFS in-place check
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
        # # Recursive DFS
        # if not root:
        #     return 0
        # return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

        # Iterative DFS
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
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
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

    # 110. Balanced Binary Tree
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # # Straightforward Top-Down, O(n^2)
        # def height(node: Optional[TreeNode]) -> int:
        #     if not node:
        #         return -1
        #     return 1 + max(height(node.left), height(node.right))
        #
        # if not root:
        #     return True
        # return (
        #     abs(height(root.left) - height(root.right)) <= 1
        #     and self.isBalanced(root.left)
        #     and self.isBalanced(root.right)
        # )

        # Heuristic Bottom-Up, O(n)
        def height(node: Optional[TreeNode]) -> int:
            """
            Return height of node (min height is 1 instead of traditional 0
            for the heuristic approach to work) if node is balanced, else
            return -1
            """
            if not node:
                return 0

            left_height = height(node.left)
            if left_height == -1:
                return -1

            right_height = height(node.right)
            if right_height == -1:
                return -1

            if abs(left_height - right_height) > 1:
                return -1

            return 1 + max(left_height, right_height)

        return height(root) != -1

    # 124. Binary Tree Maximum Path Sum
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def traverse(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node:
                return 0

            left = max(0, traverse(node.left))
            right = max(0, traverse(node.right))

            max_sum = max(max_sum, left + node.val + right)
            return node.val + max(left, right)

        max_sum = -1001
        traverse(root)
        return max_sum

    # 129. Sum Root to Leaf Numbers
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        res = 0

        def dfs(node: Optional[TreeNode], num: int) -> None:
            nonlocal res
            if not node:
                return
            num = num * 10 + node.val
            if not node.left and not node.right:
                res += num
                return
            dfs(node.left, num)
            dfs(node.right, num)

        dfs(root, 0)
        return res

    # 199. Binary Tree Right Side View
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # # BFS
        # if not root:
        #     return []
        #
        # queue = collections.deque([root])
        # result = []
        #
        # while queue:
        #     level_size = len(queue)
        #     for i in range(level_size):
        #         node = queue.popleft()
        #         if i == level_size - 1:
        #             result.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #
        # return result

        # DFS
        result = []

        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if not node:
                return

            if depth == len(result):
                result.append(node.val)

            dfs(node.right, depth + 1)
            dfs(node.left, depth + 1)

        dfs(root, 0)
        return result

    # 226. Invert Binary Tree
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # # BFS
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

        # DFS
        if not root:
            return

        left = root.left
        right = root.right

        root.left = self.invertTree(right)
        root.right = self.invertTree(left)

        return root

    # 230. Kth Smallest Element in a BST
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        rank = 0
        result = 0

        def inorder(node: Optional[TreeNode]) -> None:
            nonlocal rank
            nonlocal result
            if node:
                inorder(node.left)
                rank += 1
                if rank == k:
                    result = node.val
                    return
                inorder(node.right)

        inorder(root)
        return result

    # 235. Lowest Common Ancestor of a Binary Search Tree
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        if root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        return root

    # 404. Sum of Left Leaves
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        result = 0

        def dfs(node: Optional[TreeNode]) -> None:
            nonlocal result
            if not node:
                return
            if node.left:
                if not node.left.left and not node.left.right:
                    result += node.left.val
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return result

    # 513. Find Bottom Left Tree Value
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        left_side_view = []

        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if not node:
                return

            if depth == len(left_side_view):
                left_side_view.append(node.val)

            dfs(node.left, depth + 1)
            dfs(node.right, depth + 1)

        dfs(root, 0)
        return left_side_view[-1]

    # 543. Diameter of Binary Tree
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        result = 0

        def max_depth(node: Optional[TreeNode]) -> int:
            nonlocal result
            if not node:
                return 0

            left = max_depth(node.left)
            right = max_depth(node.right)

            result = max(result, left + right)
            return max(left, right) + 1

        max_depth(root)
        return result

    # 572. Subtree of Another Tree
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # # Naive, O(|root| * |subRoot|)
        # def is_same_tree(root1: Optional[TreeNode],
        #                  root2: Optional[TreeNode]) -> bool:
        #     if not root1 or not root2:
        #         return root1 == root2
        #     return (
        #         root1.val == root2.val
        #         and is_same_tree(root1.left, root2.left)
        #         and is_same_tree(root1.right, root2.right)
        #     )
        #
        # if not root:
        #     return False
        # if is_same_tree(root, subRoot):
        #     return True
        # return (
        #     self.isSubtree(root.left, subRoot)
        #     or self.isSubtree(root.right, subRoot)
        # )

        # Merkle tree (hashing), O(|root| + |subRoot|)
        def _hash(x: str) -> str:
            hash_code = hashlib.sha256()
            hash_code.update(x.encode())
            return hash_code.hexdigest()

        def merkle(node: Optional[TreeNode]) -> str:
            if not node:
                return "?"
            merkle_left = merkle(node.left)
            merkle_right = merkle(node.right)
            node.merkle = _hash(merkle_left + str(node.val) + merkle_right)
            return node.merkle

        def dfs(node: Optional[TreeNode]) -> bool:
            if not node:
                return False
            return node.merkle == subRoot.merkle or dfs(node.left) or dfs(node.right)

        merkle(root)
        merkle(subRoot)

        return dfs(root)

    # 623. Add One Row to Tree
    def addOneRow(
        self, root: Optional[TreeNode], val: int, depth: int
    ) -> Optional[TreeNode]:
        if depth == 1:
            new_root = TreeNode(val, root)
            return new_root

        queue = collections.deque()
        curr_depth = 0

        queue.append(root)
        while queue:
            curr_depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if curr_depth == depth - 1:
                    new_left = TreeNode(val, left=node.left)
                    new_right = TreeNode(val, right=node.right)
                    node.left = new_left
                    node.right = new_right
            if curr_depth == depth - 1:
                break

        return root

    # 872. Leaf-Similar Trees
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        # # Append to List
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

        # Generator
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
        # # BFS
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

        # DFS
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

    # 988. Smallest String Starting From Leaf
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        result = ""

        def dfs(node: Optional[TreeNode], path: List[str]) -> None:
            nonlocal result
            if not node:
                return

            path.append(chr(ord("a") + node.val))
            if not node.left and not node.right:
                s = "".join(reversed(path))
                if not result or result > s:
                    result = s

            dfs(node.left, path)
            dfs(node.right, path)
            path.pop()

        dfs(root, [])
        return result

    # 1026. Maximum Difference Between Node and Ancestor
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        # # Lengthy 2 HashMap 2-pass DFS, but somehow efficient
        # map_ancestor_to_min = {}
        # map_ancestor_to_max = {}
        #
        # def dfs_min(node: Optional[TreeNode]) -> int:
        #     if not node:
        #         return 100001
        #
        #     map_ancestor_to_min[node.val] = min(
        #         node.val,
        #         dfs_min(node.left),
        #         dfs_min(node.right)
        #     )
        #
        #     return map_ancestor_to_min[node.val]
        #
        # def dfs_max(node: Optional[TreeNode]) -> int:
        #     if not node:
        #         return -1
        #
        #     map_ancestor_to_max[node.val] = max(
        #         node.val,
        #         dfs_max(node.left),
        #         dfs_max(node.right)
        #     )
        #
        #     return map_ancestor_to_max[node.val]
        #
        # dfs_min(root)
        # dfs_max(root)
        # max_diff = 0
        #
        # for ancestor, min_descendant in map_ancestor_to_min.items():
        #     max_diff = max(max_diff, ancestor - min_descendant)
        #
        # for ancestor, max_descendant in map_ancestor_to_max.items():
        #     max_diff = max(max_diff, max_descendant - ancestor)
        #
        # return max_diff

        # More concise 1-pass DFS
        def max_diff(node: Optional[TreeNode], min_node: int, max_node: int) -> int:
            if not node:
                return 0
            min_node = min(min_node, node.val)
            max_node = max(max_node, node.val)
            max_diff_left = max_diff(node.left, min_node, max_node)
            max_diff_right = max_diff(node.right, min_node, max_node)
            return max(max_node - min_node, max_diff_left, max_diff_right)

        return max_diff(root, root.val, root.val)

    # 1325. Delete Leaves With a Given Value
    def removeLeafNodes(
        self, root: Optional[TreeNode], target: int
    ) -> Optional[TreeNode]:
        def is_leaf(node: Optional[TreeNode]) -> bool:
            assert node
            return not node.left and not node.right

        if not root:
            return None
        root.left = self.removeLeafNodes(root.left, target)
        root.right = self.removeLeafNodes(root.right, target)
        return None if is_leaf(root) and root.val == target else root

    # 1448. Count Good Nodes in Binary Tree
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node: Optional[TreeNode], curr_max) -> None:
            if not node:
                return

            nonlocal count
            if node.val >= curr_max:
                count += 1
                curr_max = node.val

            dfs(node.left, curr_max)
            dfs(node.right, curr_max)

        count = 0
        dfs(root, float("-inf"))
        return count

    # 1457. Pseudo-Palindromic Paths in a Binary Tree
    def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], path: collections.defaultdict) -> None:
            nonlocal count
            if not node:
                return

            path[node.val] += 1
            if not node.left and not node.right:
                count_odd = 0
                is_palindrome = True
                for cnt in path.values():
                    if cnt % 2 == 1:
                        count_odd += 1
                    if count_odd > 1:
                        is_palindrome = False
                        break

                if is_palindrome:
                    count += 1
                path[node.val] -= 1
                return

            if node.left:
                dfs(node.left, path)
            if node.right:
                dfs(node.right, path)

            path[node.val] -= 1
            return

        count = 0
        dfs(root, collections.defaultdict(int))
        return count

    # 1609. Even Odd Tree
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        queue = collections.deque()

        queue.append(root)
        is_even_level = True
        while queue:
            prev_val = -math.inf if is_even_level else math.inf
            for _ in range(len(queue)):
                node = queue.popleft()

                if is_even_level and (node.val % 2 == 0 or node.val <= prev_val):
                    return False
                if not is_even_level and (node.val % 2 != 0 or node.val >= prev_val):
                    return False

                prev_val = node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            is_even_level = not is_even_level

        return True

    # 2265. Count Nodes Equal to Average of Subtree
    def averageOfSubtree(self, root: TreeNode) -> int:
        result = 0

        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            nonlocal result
            if not node:
                return (0, 0)
            left_sum, left_count = dfs(node.left)
            right_sum, right_count = dfs(node.right)

            tree_sum = left_sum + node.val + right_sum
            tree_count = left_count + 1 + right_count
            if tree_sum // tree_count == node.val:
                result += 1
            return (tree_sum, tree_count)

        dfs(root)
        return result

    # 2331. Evaluate Boolean Binary Tree
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        def postorder(node: Optional[TreeNode]) -> bool:
            assert node
            if not node.left and not node.right:
                return bool(node.val)
            left = postorder(node.left)
            right = postorder(node.right)
            if node.val == 2:
                return left or right
            return left and right

        return postorder(root)

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

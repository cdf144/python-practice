import collections


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return '[]'

        queue = collections.deque()
        nodes = []

        queue.append(root)
        while queue:
            node = queue.popleft()
            if not node:
                nodes.append(None)
                continue

            nodes.append(node.val)
            queue.append(node.left)
            queue.append(node.right)

        while nodes[-1] is None:
            nodes.pop()

        return (
            '[' +
            ','.join(
                str(node) if node is not None else 'null' for node in nodes
            ) +
            ']'
        )

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if data == '[]':
            return None

        nodes = [None if node == 'null' else TreeNode(int(node))
                 for node in data.strip('[]').split(',')]

        children = nodes[::-1]
        root = children.pop()  # root is not a children of any node

        for node in nodes:
            if node:
                if children:
                    node.left = children.pop()
                if children:
                    node.right = children.pop()

        return root

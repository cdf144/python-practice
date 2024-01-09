import collections
from typing import Optional


class _Node:
    def __init__(self, val: int | str):
        self.val = val
        self.left = self.right = None


class BST:
    def __init__(self):
        self.__root = None

    def insert(self, val: int | str) -> None:
        self.__root = self.__insert(self.__root, val)

    def __insert(self, x: Optional[_Node], val: int | str) -> _Node:
        if not x:
            return _Node(val)

        if val < x.val:
            x.left = self.__insert(x.left, val)
        else:
            x.right = self.__insert(x.right, val)

        return x

    def height(self) -> int:
        return self.__height(self.__root)

    def __height(self, x: Optional[_Node]) -> int:
        if not x:
            return -1
        return 1 + max(self.__height(x.left), self.__height(x.right))

    def min(self) -> int | str:
        return self.__min(self.__root).val

    def __min(self, x: Optional[_Node]) -> _Node:
        if not x.left:
            return x
        return self.__min(x.left)

    def max(self) -> int | str:
        return self.__max(self.__root).val

    def __max(self, x: Optional[_Node]) -> _Node:
        if not x.right:
            return x
        return self.__max(x.right)

    def del_max(self) -> None:
        self.__root = self.__del_max(self.__root)

    def __del_max(self, x: Optional[_Node]) -> _Node:
        if not x.right:
            return x.left
        x.right = self.__del_max(x.right)
        return x

    def del_min(self) -> None:
        self.__root = self.__del_min(self.__root)

    def __del_min(self, x: Optional[_Node]) -> _Node:
        if not x.left:
            return x.right
        x.left = self.__del_min(x.left)
        return x

    def preorder(self) -> None:
        self.__preorder(self.__root)

    def __preorder(self, x: Optional[_Node]) -> None:
        if x:
            print(x.val)
            self.__preorder(x.left)
            self.__preorder(x.right)

    def inorder(self) -> None:
        self.__inorder(self.__root)

    def __inorder(self, x: Optional[_Node]) -> None:
        if x:
            self.__inorder(x.left)
            print(x.val)
            self.__inorder(x.right)

    def postorder(self) -> None:
        self.__postorder(self.__root)

    def __postorder(self, x: Optional[_Node]) -> None:
        if x:
            self.__postorder(x.left)
            self.__postorder(x.right)
            print(x.val)

    def levelorder(self) -> None:
        self.__levelorder(self.__root)

    def __levelorder(self, x: Optional[_Node]) -> None:
        queue = collections.deque()
        queue.append(x)
        while queue:
            node = queue.popleft()
            print(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

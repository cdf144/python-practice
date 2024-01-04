class Node:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity

        # Dictionary for O(1) access
        self.key_to_node = {}

        # Head and Tail are for references, they are not actually in the cache
        self.head = Node(-1, -1)
        self.tail = Node(-1, -1)
        self.join(self.head, self.tail)

    def get(self, key: int) -> int:
        if key not in self.key_to_node:
            return -1

        # If cache hit, move key to most recently used
        node = self.key_to_node[key]
        self.remove(node)
        self.move_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        # If key already in cache, update the value of key in cache, and move
        # key to most recently used
        if key in self.key_to_node:
            node = self.key_to_node[key]
            node.val = value
            self.remove(node)
            self.move_to_head(node)
            return

        # If number of keys exceed cache capacity, evict the LRU key
        if len(self.key_to_node) == self.capacity:
            least_recently_used = self.tail.prev
            del self.key_to_node[least_recently_used.key]
            self.remove(least_recently_used)

        self.move_to_head(Node(key, value))
        self.key_to_node[key] = self.head.next

    def move_to_head(self, node: Node):
        """
        This effectively makes the node the most recently used key
        """
        self.join(node, self.head.next)
        self.join(self.head, node)

    def remove(self, node: Node) -> None:
        self.join(node.prev, node.next)

    def join(self, n1: Node, n2: Node) -> None:
        """
        Make 2 nodes bound to each other
        """
        n1.next = n2
        n2.prev = n1

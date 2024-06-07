from typing import Dict, Optional


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, Optional[TrieNode]] = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            assert node
            node = node.children.setdefault(c, TrieNode())
        assert node
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self._find_prefix(word)
        return node is not None and node.is_end_of_word

    def startsWith(self, word: str) -> bool:
        return self._find_prefix(word) is not None

    def _find_prefix(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for c in prefix:
            assert node
            if c not in node.children:
                return None
            node = node.children[c]
        return node

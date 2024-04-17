from typing import List


class Codec:
    def encode(self, strs: List[str]) -> str:
        return "".join(str(len(s)) + "#" + s for s in strs)

    def decode(self, s: str) -> List[str]:
        decoded = []
        n, i = len(s), 0

        while i < n:
            mark = s.find("#", i)
            length = int(s[i:mark])
            i = mark + length + 1
            decode = s[mark + 1 : i]
            decoded.append(decode)

        return decoded

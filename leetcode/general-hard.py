class Solution:
    # 214. Shortest Palindrome
    def shortestPalindrome(self, s: str) -> str:
        # reversed string for checking
        r = s[::-1]

        # find the longest palindromic substring
        for i in range(len(s)):
            if s.startswith(r[i:]):
                return r[:i] + s

        # worst case: add the reverse with original
        # -> automatically a palindrome
        return r + s

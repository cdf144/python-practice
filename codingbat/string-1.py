def hello_name(name: str) -> str:
    return "Hello " + name + "!"


def make_abba(a: str, b: str) -> str:
    return a + 2 * b + a


def make_tags(tag: str, word: str) -> str:
    return "<{0}>{1}</{0}>".format(tag, word)


def make_out_word(out: str, word: str) -> str:
    out_1 = out[0:2]
    out_2 = out[2:4]
    return "{0}{1}{2}".format(out_1, word, out_2)


def extra_end(str) -> str:
    last_2 = str[-2:]
    s = ""
    for i in range(3):
        s += last_2

    return s


def first_two(str) -> str:
    return str[:2]


def first_half(str) -> str:
    return str[:len(str) / 2]


def without_end(str) -> str:
    return str[1:-1]


def combo_string(a: str, b: str) -> str:
    if len(a) < len(b):
        return a + b + a
    else:
        return b + a + b


def non_start(a: str, b: str) -> str:
    return a[1:] + b[1:]


def reverse_str_slicing(str) -> str:
    return str[::-1]


def left2(str) -> str:
    k = len(str) - 2
    s = reverse_str_slicing(str)
    s1 = s[:k]
    s2 = s[k:]
    return reverse_str_slicing(s1) + reverse_str_slicing(s2)


if __name__ == "__main__":
    print(hello_name("Bob"))

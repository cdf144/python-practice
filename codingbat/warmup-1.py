def sleep_in(weekday: bool, vacation: bool) -> bool:
    return not weekday or vacation


def monkey_trouble(a_smile: bool, b_smile: bool) -> bool:
    return a_smile == b_smile


def sum_double(a: int, b: int) -> int:
    double_sum = a + b
    return double_sum if a != b else double_sum * 2


def diff21(n: int) -> int:
    diff = abs(n - 21)
    return diff if n <= 21 else diff * 2


def parrot_trouble(talking: bool, hour: int) -> bool:
    return talking and not (7 <= hour <= 20)


def makes10(a: int, b: int) -> bool:
    return True if a == 10 or b == 10 or a + b == 10 else False


def near_hundred(n: int) -> bool:
    return True if abs(n - 100) <= 10 or abs(n - 200) <= 10 else False


def pos_neg(a: int, b: int, negative: bool) -> bool:
    a_neg = a < 0
    b_neg = b < 0
    if negative:
        return a_neg and b_neg
    else:
        return a_neg != b_neg


def not_string(str) -> str:
    if len(str) >= 3 and str[:3] == "not":
        return str
    return "not " + str


def missing_char(str, n: int) -> str:
    return str[:n] + str[n + 1:]


def front_back(str) -> str:
    if len(str) <= 1:
        return str

    return str[len(str) - 1] + str[1:-1] + str[0]


def front3(str) -> str:
    return 3 * str[:3]


if __name__ == "__main__":
    print(diff21(22))

def cigar_party(cigars: int, is_weekend: bool) -> bool:
    return cigars >= 40 and (is_weekend or (not is_weekend and cigars <= 60))


def date_fashion(you: int, date: int) -> int:
    if you <= 2 or date <= 2:
        return 0
    elif you >= 8 or date >= 8:
        return 2
    else:
        return 1


def squirrel_play(temp: int, is_summer: bool) -> bool:
    upper_temp_limit = 100 if is_summer else 90
    return 60 <= temp <= upper_temp_limit


def caught_speeding(speed: int, is_birthday: bool) -> int:
    tolerance = 5 if is_birthday else 0
    if speed <= 60 + tolerance:
        return 0
    elif 60 + tolerance < speed <= 80 + tolerance:
        return 1
    else:
        return 2


def sorta_sum(a: int, b: int) -> int:
    s = a + b
    return 20 if 10 <= s <= 19 else s


def alarm_clock(day: int, vacation: bool) -> str:
    if day == 0 or day == 6:
        return 'off' if vacation else '10:00'
    else:
        return '10:00' if vacation else '7:00'


def love6(a: int, b: int) -> bool:
    return a == 6 or b == 6 or a + b == 6 or abs(a - b) == 6


def in1to10(n: int, outside_mode: bool) -> bool:
    return (n <= 1 or n >= 10) if outside_mode else 1 <= n <= 10


def near_ten(num: int) -> bool:
    return num % 10 <= 2 or num % 10 >= 8

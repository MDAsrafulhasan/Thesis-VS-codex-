"""Small offline Bengali integer-to-words converter.

The word table and Bangladesh grouping follow num2words' LGPL-2.1 Bengali
converter by Mehedi Hasan Khondoker. Only the cardinal integer operation used
by the inference router is retained so Kaggle does not need a network install.
"""

from __future__ import annotations


ONES = ["", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]
TEN_TO_99 = [
    "দশ", "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনের", "ষোল", "সতের", "আঠারো", "উনিশ",
    "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "উনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁইত্রিশ", "ছত্রিশ", "সাতত্রিশ", "আটত্রিশ", "উনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চৌচল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "উনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "উনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছিষট্টি", "সাতষট্টি", "আটষট্টি", "উনসত্তর",
    "সত্তর", "একাত্তর ", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "উনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "উননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]


def _cardinal(number: int) -> str:
    if number == 0:
        return "শূন্য"
    words = ""
    for scale, name in ((10**7, " কোটি "), (10**5, " লাখ "), (1000, " হাজার ")):
        if number >= scale:
            words += _cardinal(number // scale) + name
            number %= scale
    if number >= 100:
        words += ONES[number // 100] + "শত "
        number %= 100
    if 10 <= number <= 99:
        words += TEN_TO_99[number - 10] + " "
        number = 0
    if 0 < number < 10:
        words += ONES[number] + " "
    return words.strip()


def num2words(number: int, lang: str = "bn") -> str:
    if lang != "bn":
        raise NotImplementedError("Only Bengali cardinal integers are packaged")
    value = int(number)
    if value < 0:
        value = abs(value)
    return _cardinal(value)

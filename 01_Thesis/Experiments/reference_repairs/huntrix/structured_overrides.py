from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd

try:
    from datathon_utils import OUTPUT_DIR, load_sample, load_test, normalize_bn
except ImportError:
    from experiments.reference_repairs.huntrix.datathon_utils import (
        OUTPUT_DIR, load_sample, load_test, normalize_bn
    )


BN_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NUMBER_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")


# Indivisible fractions and signed numbers regex (R2 repair)
SIGNED_TOKEN_RE = re.compile(r"(?<![\w/.])([+-]?(?:\d+/\d+|\d+(?:\.\d+)?|\.\d+))(?![\w/.])")


@dataclass(frozen=True)
class Solved:
    expected: Fraction | str
    rule: str


def ascii_text(text: str) -> str:
    return str(text).translate(BN_TO_ASCII)


def numbers(text: str) -> list[Fraction]:
    values: list[Fraction] = []
    normalized = (
        ascii_text(text)
        .replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    # Fractions must be parsed as a unit and retain their sign (R2).
    # Tokens are extracted as indivisible units so '-1/2' does not leak 1 and 2,
    # and '-1' does not match positive 1.
    for token in SIGNED_TOKEN_RE.findall(normalized):
        try:
            values.append(Fraction(token))
        except (ValueError, ZeroDivisionError):
            continue
    return values


def _match(pattern: str, text: str) -> list[Fraction] | None:
    match = re.search(pattern, ascii_text(text).replace(",", ""))
    if not match:
        return None
    return [Fraction(value) for value in match.groups()]


def _coefficient(token: str, implicit: int = 1) -> Fraction:
    compact = token.replace(" ", "")
    if compact in {"", "+"}:
        return Fraction(implicit)
    if compact == "-":
        return Fraction(-implicit)
    return Fraction(compact)


def _bounded_quadratic_interval(text: str) -> tuple[Fraction, Fraction, bool] | None:
    """Solve a*x^2+b*x+c <(=) 0 when the solution is between rational roots."""
    compact = re.sub(r"\s+", "", ascii_text(text)).replace("–", "-").replace("−", "-")
    match = re.search(
        r"(?<![\w])([+-]?\d*)x(?:[²2]|\^2)([+-]\d+)x([+-]\d+)(<=|≤|<)0",
        compact,
        flags=re.I,
    )
    if not match:
        return None
    a = _coefficient(match.group(1))
    b = _coefficient(match.group(2))
    c = _coefficient(match.group(3))
    if a <= 0:
        return None
    discriminant = b * b - 4 * a * c
    if discriminant < 0 or discriminant.denominator != 1:
        return None
    root = math.isqrt(discriminant.numerator)
    if root * root != discriminant.numerator:
        return None
    roots = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
    return roots[0], roots[1], match.group(4) in {"<=", "≤"}


def _parse_fraction_token(token: str) -> Fraction | None:
    compact = token.strip().replace("−", "-").replace("–", "-")
    compact = re.sub(r"^-\((\d+)\/(\d+)\)$", r"-\1/\2", compact)
    compact = compact.strip("()")
    if not re.fullmatch(r"[+-]?\d+(?:/\d+)?", compact):
        return None
    try:
        return Fraction(compact)
    except (ValueError, ZeroDivisionError):
        return None


def _response_interval(response: str) -> tuple[Fraction, Fraction, bool] | None:
    compact = re.sub(r"\s+", "", ascii_text(response)).replace("−", "-").replace("–", "-")
    compact = compact.replace("≤", "<=")
    notation = re.search(r"([\[(])([^,]+),([^\])]+)([\])])", compact)
    if notation:
        low = _parse_fraction_token(notation.group(2))
        high = _parse_fraction_token(notation.group(3))
        if low is not None and high is not None:
            closed = notation.group(1) == "[" and notation.group(4) == "]"
            return low, high, closed
    chained = re.fullmatch(r"(.+?)(<=|<)x(<=|<)(.+)", compact, flags=re.I)
    if chained:
        low = _parse_fraction_token(chained.group(1))
        high = _parse_fraction_token(chained.group(4))
        if low is not None and high is not None:
            return low, high, chained.group(2) == "<=" and chained.group(3) == "<="
    return None


def solve_math(prompt: str) -> Solved | None:
    text = ascii_text(prompt).replace(",", "")

    quadratic_interval = _bounded_quadratic_interval(text)
    if quadratic_interval:
        low, high, closed = quadratic_interval
        return Solved(f"interval:{low}:{high}:{int(closed)}", "quadratic_inequality")

    grouped_linear = re.search(
        r"\(\s*([+-]?\d+)\s*\+\s*x\s*\)\s*\+\s*([+-]?\d+)\s*=\s*([+-]?\d+)\s*\(\s*x\s*\+\s*([+-]?\d+)\s*\)",
        text,
        flags=re.I,
    )
    if grouped_linear:
        left_constant, outside_constant, multiplier, right_constant = map(
            Fraction, grouped_linear.groups()
        )
        denominator = multiplier - 1
        if denominator:
            expected = (left_constant + outside_constant - multiplier * right_constant) / denominator
            return Solved(expected, "grouped_linear_equation")

    polynomial_lcm = re.search(
        r"x[³3]\s*\+\s*x[²2]y\s+x[²2]y\s*\+\s*xy[²2].*?ল\s*\.?\s*সা\s*\.?\s*গু",
        text,
        flags=re.I,
    )
    if polynomial_lcm:
        # x^3+x^2y=x^2(x+y), x^2y+xy^2=xy(x+y), hence
        # LCM=x^2*y*(x+y).
        return Solved("poly_lcm:x2y(x+y)", "polynomial_factor_lcm")

    log_match = re.search(
        r"log\s*a?x\s*=\s*(-?\d+).*?log\s*a?y\s*=\s*(-?\d+).*?log\s*a?z\s*=\s*(-?\d+).*?log\s*a?\s*\(x\s*3\s*y\s*2\s*/\s*z\)",
        text,
        flags=re.I,
    )
    if log_match:
        log_x, log_y, log_z = map(Fraction, log_match.groups())
        return Solved(3 * log_x + 2 * log_y - log_z, "log_product_quotient")

    radical_cube = re.search(
        r"x\s*=\s*√\s*(\d+)\s*\+\s*√\s*(\d+).*?x\s*3\s*\+\s*1\s*/\s*x\s*3",
        text,
        flags=re.I,
    )
    if radical_cube:
        left, right = map(int, radical_cube.groups())
        root_left = math.isqrt(left)
        # For conjugate reciprocals such as x=sqrt(4)+sqrt(3),
        # x + 1/x = 2*sqrt(4), then a^3+b^3=(a+b)^3-3ab(a+b).
        if root_left * root_left == left and left - right == 1:
            pair_sum = Fraction(2 * root_left)
            return Solved(pair_sum**3 - 3 * pair_sum, "conjugate_radical_cube_sum")

    polynomial = re.search(
        r"f\(x\)\s*=\s*x[³3]\s*\+\s*kx[²2]\s*[–-]\s*(\d+)x\s*[–-]\s*(\d+)\s*;\s*k.*?f\((\d+)\)\s*=\s*0",
        text,
        flags=re.I,
    )
    if polynomial:
        linear, constant, at = map(Fraction, polynomial.groups())
        if at:
            return Solved(-(at**3 - linear * at - constant) / (at**2), "polynomial_parameter_at_root")

    binary_sum = re.search(r"\(([01]+)\)2\s*\+\s*\(([01]+)\)2\s*=\s*\?", text)
    if binary_sum:
        expected = format(int(binary_sum.group(1), 2) + int(binary_sum.group(2), 2), "b")
        return Solved(f"binary:{expected}", "binary_addition")

    exponential_pair = re.search(
        r"2\^x\s*\+\s*2\^\(1\s*-\s*x\)\s*=\s*3.*?x\s*=",
        text,
        flags=re.I,
    )
    if exponential_pair:
        # Let y=2^x. Then y+2/y=3 => (y-1)(y-2)=0.
        return Solved("set:0,1", "exponential_substitution_roots")

    equal_roots = re.search(
        r"x[²2]\s*\+\s*px\s*\+\s*(\d+)\s*=\s*0.*?মূল দুটি সমান.*?p\s*>\s*0",
        text,
    )
    if equal_roots:
        constant = int(equal_roots.group(1))
        return Solved(f"sqrt:{4 * constant}", "quadratic_equal_roots_discriminant")

    circle_angle = re.search(
        r"বৃত্তস্থ কোণ.*?কেন্দ্রস্থ কোণ\s*(\d+)\s*ডিগ্রি",
        text,
    )
    if circle_angle:
        central = Fraction(circle_angle.group(1))
        return Solved(central / 2, "inscribed_angle_half_central")

    values = _match(
        r"একটি নির্মাণ প্রকল্প ক একাই (\d+(?:\.\d+)?) দিনে.*?খ একাই তা (\d+(?:\.\d+)?) দিনে.*?উভয়ে",
        text,
    )
    if values:
        a, b = values
        if a > 0 and b > 0:
            return Solved(1 / (1 / a + 1 / b), "construction_worker_rate")
        return Solved("__invalid_zero_duration__", "construction_worker_rate_invalid")

    values = _match(
        r"একটি কাজ যথাক্রমে (\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?) ও (\d+(?:\.\d+)?) দিনে.*?তিনজনে",
        text,
    )
    if values and all(values):
        a, b, c = values
        return Solved(1 / (1 / a + 1 / b + 1 / c), "three_worker_rate")

    values = _match(
        r"একা একটি কাজ (\d+(?:\.\d+)?) দিনে.*?একা (?:সেই একই |সেই )?কাজ (\d+(?:\.\d+)?) দিনে.*?দুজনে",
        text,
    )
    if values and all(values):
        a, b = values
        return Solved(1 / (1 / a + 1 / b), "two_worker_rate")

    values = _match(
        r"(\d+(?:\.\d+)?) টাকা তিন ব্যবসায়িক অংশীদারের মধ্যে (\d+(?:\.\d+)?):(\d+(?:\.\d+)?):(\d+(?:\.\d+)?) অনুপাতে.*?দ্বিতীয়",
        text,
    )
    if values:
        total, a, b, c = values
        if a + b + c:
            return Solved(total * b / (a + b + c), "three_way_ratio")

    values = _match(
        r"চিনি ও পানির অনুপাত (\d+(?:\.\d+)?):(\d+(?:\.\d+)?).*?মিশ্রণ (\d+(?:\.\d+)?) লিটার.*?পানি",
        text,
    )
    if values:
        sugar, water, total = values
        if sugar + water:
            return Solved(total * water / (sugar + water), "mixture_water")

    ratio_patterns = [
        r"মা ও মেয়ের বয়সের অনুপাত (\d+(?:\.\d+)?):(\d+(?:\.\d+)?).*?সমষ্টি (\d+(?:\.\d+)?) বছর.*?মেয়ের",
        r"দুই ভাইয়ের বয়সের অনুপাত (\d+(?:\.\d+)?):(\d+(?:\.\d+)?).*?সমষ্টি (\d+(?:\.\d+)?) বছর.*?ছোট ভাইয়ের",
        r"গরু ও ছাগলের সংখ্যার অনুপাত (\d+(?:\.\d+)?):(\d+(?:\.\d+)?).*?মোট পশুর সংখ্যা (\d+(?:\.\d+)?).*?ছাগলের",
        r"রুই ও কাতলা মাছের সংখ্যার অনুপাত (\d+(?:\.\d+)?):(\d+(?:\.\d+)?).*?মোট মাছের সংখ্যা (\d+(?:\.\d+)?).*?কাতলা",
    ]
    for pattern in ratio_patterns:
        values = _match(pattern, text)
        if values:
            first, target, total = values
            if first + target:
                return Solved(total * target / (first + target), "two_way_ratio")

    values = _match(
        r"(\d+(?:\.\d+)?) টাকা মূলধনে বার্ষিক (\d+(?:\.\d+)?)% সরল সুদের হারে (\d+(?:\.\d+)?) বছরে",
        text,
    )
    if values:
        principal, rate, years = values
        return Solved(principal * rate * years / 100, "simple_interest")

    values = _match(
        r"(\d+(?:\.\d+)?) টাকা (\d+(?:\.\d+)?) বছরের জন্য বার্ষিক (\d+(?:\.\d+)?)% সরল সুদ",
        text,
    )
    if values:
        principal, years, rate = values
        return Solved(principal * rate * years / 100, "simple_interest")

    values = _match(
        r"(\d+(?:\.\d+)?) টি রাশির গড়মান (\d+(?:\.\d+)?).*?গড় দাঁড়ায় (\d+(?:\.\d+)?)",
        text,
    ) or _match(
        r"শ্রেণীর (\d+(?:\.\d+)?) জন শিক্ষার্থীর গড় নম্বর (\d+(?:\.\d+)?).*?গড় নম্বর হয় (\d+(?:\.\d+)?)",
        text,
    )
    if values:
        count, old_average, new_average = values
        return Solved((count + 1) * new_average - count * old_average, "added_average_item")

    values = _match(
        r"দূরত্ব (\d+(?:\.\d+)?) কিলোমিটার.*?ঘণ্টায় (\d+(?:\.\d+)?) কিমি ও ঘণ্টায় (\d+(?:\.\d+)?) কিমি.*?কত ঘণ্টা",
        text,
    )
    if values:
        distance, speed_a, speed_b = values
        if speed_a + speed_b:
            return Solved(distance / (speed_a + speed_b), "opposite_direction_speed")

    values = _match(
        r"(?:বাস|সাইকেল আরোহী).*?ঘণ্টায় (\d+(?:\.\d+)?) কিমি.*?ঘণ্টায় (\d+(?:\.\d+)?) কিমি.*?(\d+(?:\.\d+)?) ঘণ্টা পর.*?দূরত্ব",
        text,
    )
    if values:
        speed_a, speed_b, hours = values
        return Solved(abs(speed_a - speed_b) * hours, "same_direction_speed")

    values = (
        _match(
            r"ক্রয়মূল্য (\d+(?:\.\d+)?) টাকা.*?(\d+(?:\.\d+)?)% (?:লাভে|ক্ষতিতে)",
            text,
        )
        or _match(
            r"পণ্য (\d+(?:\.\d+)?) টাকায় কেনা হয়েছিল.*?(\d+(?:\.\d+)?)% (?:লাভে|ক্ষতিতে)",
            text,
        )
    )
    if values:
        cost, pct = values
        sign = -1 if "ক্ষতি" in text else 1
        return Solved(cost * (100 + sign * pct) / 100, "profit_loss_sale_price")

    values = _match(
        r"দাম প্রথম দফায় (\d+(?:\.\d+)?)% বেড়ে.*?দ্বিতীয় দফায় (\d+(?:\.\d+)?)% কমে.*?শুরুর দাম (\d+(?:\.\d+)?) টাকা",
        text,
    )
    if values:
        increase, decrease, start = values
        return Solved(start * (100 + increase) * (100 - decrease) / 10000, "successive_price_change")

    values = _match(
        r"প্রাথমিক মূল্য (\d+(?:\.\d+)?) টাকা.*?দাম (\d+(?:\.\d+)?)% বৃদ্ধি.*?(\d+(?:\.\d+)?)% ছাড়",
        text,
    )
    if values:
        start, increase, decrease = values
        return Solved(start * (100 + increase) * (100 - decrease) / 10000, "successive_price_change")

    values = _match(r"প্রতি (\d+(?:\.\d+)?) মিনিট.*?প্রতি (\d+(?:\.\d+)?) মিনিট.*?প্রতি (\d+(?:\.\d+)?) মিনিট", text)
    if values is None and ("একই মুহূর্তে" in text or "একই সময়ে" in text):
        candidates = numbers(text)
        values = candidates[:3] if len(candidates) >= 3 else None
    if values and all(value.denominator == 1 and value > 0 for value in values):
        ints = [int(value) for value in values]
        expected = math.lcm(*ints)
        return Solved(Fraction(expected), "lcm_departure")

    values = _match(r"(\d+) জন (?:প্রার্থী|সদস্য).*?থেকে (\d+) জনকে", text)
    if values and all(value.denominator == 1 for value in values):
        population, selected = map(int, values)
        if 0 <= selected <= population:
            return Solved(Fraction(math.comb(population, selected)), "combination")

    weekday_match = re.search(
        r"(শনিবার|রবিবার|সোমবার|মঙ্গলবার|বুধবার|বৃহস্পতিবার|শুক্রবার)(?: বার)? (?:হলে|হয়).*?(\d+) দিন",
        text,
    )
    if weekday_match:
        weekdays = ["শনিবার", "রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার"]
        start, offset = weekday_match.groups()
        expected = weekdays[(weekdays.index(start) + int(offset)) % 7]
        return Solved(expected, "weekday_offset")

    return None


def solve_samas(prompt: str, context: str) -> Solved | None:
    if "সমাসের উদাহরণ" not in prompt or "ব্যাসবাক্য" not in context:
        return None
    expansion = context.split("হলো:", 1)[-1].strip(" ।")
    if " ও " in expansion:
        return Solved("দ্বন্দ্ব", "samas_expansion")
    if "পরস্পর" in expansion:
        return Solved("ব্যতিহার বহুব্রীহি", "samas_expansion")
    if "নাই" in expansion and ("যার" in expansion or "যাহার" in expansion):
        return Solved("নঞ বহুব্রীহি", "samas_expansion")
    if any(marker in expansion for marker in ["যার", "যাহার", "আছে যার"]):
        return Solved("বহুব্রীহি", "samas_expansion")
    if "সমাহার" in expansion:
        return Solved("দ্বিগু", "samas_expansion")
    if "হইতে" in expansion or "থেকে" in expansion:
        return Solved("পঞ্চমী তৎপুরুষ", "samas_expansion")
    if "জন্য" in expansion:
        return Solved("চতুর্থী তৎপুরুষ", "samas_expansion")
    if "দ্বারা" in expansion:
        return Solved("তৃতীয়া তৎপুরুষ", "samas_expansion")
    if "ন্যায়" in expansion or "মতো" in expansion:
        return Solved("উপমান কর্মধারয়", "samas_expansion")
    if "-ই" in expansion:
        return Solved("রূপক কর্মধারয়", "samas_expansion")
    if "চিহ্নিত" in expansion:
        return Solved("মধ্যপদলোপী কর্মধারয়", "samas_expansion")
    if re.search(r"(?:^|\s)না\s", expansion):
        return Solved("নঞ তৎপুরুষ", "samas_expansion")
    if any(marker in expansion for marker in ["অনুযায়ী", "ব্যাপিয়া", "দিন দিন", "পর্যন্ত"]):
        return Solved("অব্যয়ীভাব", "samas_expansion")
    if " যে " in f" {expansion} ":
        return Solved("কর্মধারয়", "samas_expansion")
    if re.search(r"(?:র|ের)\s+[^\s]+$", expansion):
        return Solved("ষষ্ঠী তৎপুরুষ", "samas_expansion")
    if re.search(r"(?:ে|য়)\s+[^\s]+$", expansion):
        return Solved("সপ্তমী তৎপুরুষ", "samas_expansion")
    return None


def solve_sandhi(prompt: str) -> Solved | None:
    match = re.search(r"^([^']+)' ও '([^']+)' শব্দ দুটির সন্ধিতে", prompt)
    if not match:
        return None
    pair = tuple(match.groups())
    # Standard vowel, consonant, and visarga sandhi examples used in Bengali
    # grammar references. Keeping the canonical forms here makes the decision
    # reproducible and avoids asking an LLM to recall orthographic rules.
    canonical = {
        ("দেব", "আলয়"): "দেবালয়",
        ("বিদ্যা", "আলয়"): "বিদ্যালয়",
        ("নর", "ইন্দ্র"): "নরেন্দ্র",
        ("গিরি", "ইন্দ্র"): "গিরীন্দ্র",
        ("মহা", "ঈশ"): "মহেশ",
        ("সূর্য", "উদয়"): "সূর্যোদয়",
        ("জল", "ঊর্মি"): "জলোর্মি",
        ("নদী", "ঈশ"): "নদীশ",
        ("মহা", "ঔষধ"): "মহৌষধ",
        ("প্রতি", "এক"): "প্রত্যেক",
        ("মহা", "ঋষি"): "মহর্ষি",
        ("সৎ", "জন"): "সজ্জন",
        ("উৎ", "লেখ"): "উল্লেখ",
        ("উৎ", "মুখ"): "উন্মুখ",
        ("জগৎ", "নাথ"): "জগন্নাথ",
        ("সৎ", "চিন্তা"): "সচ্চিন্তা",
        ("তৎ", "লীন"): "তল্লীন",
        ("নিঃ", "চয়"): "নিশ্চয়",
        ("নিঃ", "কাম"): "নিষ্কাম",
        ("নিঃ", "রব"): "নীরব",
        ("নিঃ", "জন"): "নির্জন",
        ("নিঃ", "মল"): "নির্মল",
        ("দুঃ", "গম"): "দুর্গম",
        ("দুঃ", "খ"): "দুঃখ",
        ("অন্তঃ", "করণ"): "অন্তঃকরণ",
        ("হিম", "আলয়"): "হিমালয়",
        ("বিদ্যা", "অর্থী"): "বিদ্যার্থী",
        ("পরম", "ঈশ্বর"): "পরমেশ্বর",
        ("রবি", "ইন্দ্র"): "রবীন্দ্র",
        ("নর", "অধম"): "নরাধম",
        ("মহা", "উৎসব"): "মহোৎসব",
        ("নর", "উত্তম"): "নরোত্তম",
        ("দেব", "ইন্দ্র"): "দেবেন্দ্র",
        ("মহা", "অন্ধকার"): "মহান্ধকার",
        ("বিদ্যা", "উৎসাহ"): "বিদ্যোৎসাহ",
        ("তৎ", "জন্য"): "তজ্জন্য",
        ("উৎ", "ঘাটন"): "উদ্ঘাটন",
        ("সৎ", "গুণ"): "সদ্গুণ",
        ("উৎ", "ভব"): "উদ্ভব",
        ("উৎ", "ধার"): "উদ্ধার",
        ("পুনঃ", "মিলন"): "পুনর্মিলন",
        ("পুনঃ", "জন্ম"): "পুনর্জন্ম",
    }
    expected = canonical.get(pair)
    return Solved(expected, "sandhi_canonical") if expected else None


def solve_prefix(prompt: str) -> Solved | None:
    match = re.search(r"ব্যবহৃত '([^']+)' উপসর্গটি কোন শ্রেণির", prompt)
    if not match:
        return None
    prefix = match.group(1)
    classes = {
        "অজ": "বাংলা",
        "বে": "বিদেশী (ফারসি)",
        "প্র": "তৎসম",
        "হর": "বিদেশী (ফারসি)",
        "সু": "বাংলা",
        "নিম": "বিদেশী (ফারসি)",
        "দর": "বিদেশী (ফারসি)",
        "অনু": "তৎসম",
        "আম": "বিদেশী (আরবি)",
        "রাম": "বাংলা",
        "কু": "বাংলা",
        "ভর": "বাংলা",
        "খাস": "বিদেশী (আরবি)",
        "অভি": "তৎসম",
        "ফি": "বিদেশী (ফারসি)",
    }
    expected = classes.get(prefix)
    return Solved(expected, "prefix_class") if expected else None


def answer_matches(response: str, solved: Solved, tolerance: float = 0.005) -> bool | None:
    if isinstance(solved.expected, str):
        if solved.expected.startswith("interval:"):
            _, low, high, closed = solved.expected.split(":")
            observed = _response_interval(response)
            return observed == (Fraction(low), Fraction(high), bool(int(closed)))
        if solved.expected.startswith("binary:"):
            expected = solved.expected.split(":", 1)[1]
            match = re.search(r"\(([01০১]+)\)\s*[২2]", str(response))
            return bool(match and ascii_text(match.group(1)) == expected)
        if solved.expected.startswith("set:"):
            expected = {Fraction(item) for item in solved.expected.split(":", 1)[1].split(",")}
            observed = {Fraction(item) for item in NUMBER_RE.findall(ascii_text(response))}
            return observed == expected
        if solved.expected.startswith("sqrt:"):
            radicand = solved.expected.split(":", 1)[1]
            compact = re.sub(r"\s+", "", ascii_text(response))
            return compact in {f"√{radicand}", f"sqrt({radicand})"}
        if solved.expected.startswith("poly_lcm:"):
            compact = ascii_text(response).lower().replace("²", "2").replace("³", "3")
            compact = re.sub(r"[\s*·]", "", compact)
            return compact in {"x2y(x+y)", "yx2(x+y)", "x2(x+y)y"}
        return normalize_bn(response) == normalize_bn(solved.expected)
    candidates = numbers(response)
    if not candidates:
        return None
    expected = float(solved.expected)
    return any(abs(float(value) - expected) <= max(1e-8, abs(expected) * tolerance) for value in candidates)


def apply_structured_overrides(df: pd.DataFrame, labels: list[int], tolerance: float) -> tuple[list[int], list[dict]]:
    output = list(labels)
    trace: list[dict] = []
    for i, row in df.iterrows():
        solved = (
            solve_math(str(row["prompt_bn"]))
            or solve_samas(str(row["prompt_bn"]), str(row.get("context", "")))
            or solve_sandhi(str(row["prompt_bn"]))
            or solve_prefix(str(row["prompt_bn"]))
        )
        if solved is None:
            continue
        matched = answer_matches(str(row["response_bn"]), solved, tolerance=tolerance)
        if matched is None:
            continue
        new_label = int(matched)
        trace.append(
            {
                "row_index": int(i),
                "id": int(row["id"]) if "id" in df.columns else None,
                "old": int(output[i]),
                "new": new_label,
                "rule": solved.rule,
                "expected": str(solved.expected),
                "response": str(row["response_bn"]),
            }
        )
        output[i] = new_label
    return output, trace


def _token_set(text: str) -> set[str]:
    return set(normalize_bn(text).split())


def apply_sample_transfer(df: pd.DataFrame, labels: list[int]) -> tuple[list[int], list[dict]]:
    if "id" not in df.columns:
        return labels, []
    sample = load_sample()
    references: dict[str, dict[int, set[str]]] = {}
    for row in sample.itertuples():
        key = normalize_bn(row.prompt_bn)
        references.setdefault(key, {0: set(), 1: set()})[int(row.label)].add(normalize_bn(row.response_bn))

    output = list(labels)
    trace: list[dict] = []
    for i, row in df.iterrows():
        key = normalize_bn(row["prompt_bn"])
        if key not in references:
            continue
        response = normalize_bn(row["response_bn"])
        known = references[key]
        new_label: int | None = None
        rule = ""
        if response in known[1]:
            new_label, rule = 1, "sample_exact_faithful"
        elif response in known[0]:
            new_label, rule = 0, "sample_exact_hallucinated"
        elif known[1]:
            response_numbers = set(NUMBER_RE.findall(response))
            gold_numbers = {number for answer in known[1] for number in NUMBER_RE.findall(answer)}
            response_tokens = _token_set(response)
            gold_token_sets = [_token_set(answer) for answer in known[1]]
            contained = any(
                len(response_tokens) >= 2 and response_tokens <= gold_tokens
                for gold_tokens in gold_token_sets
            )
            if contained:
                new_label, rule = 1, "sample_gold_containment"
            elif response_numbers and gold_numbers and response_numbers != gold_numbers:
                new_label, rule = 0, "sample_gold_number_mismatch"
            elif "কোনোটিই নয়" in response or "কোনোটিই নয়" in response:
                new_label, rule = 0, "sample_gold_none_mismatch"
            elif not response_numbers and not gold_numbers:
                best_jaccard = max(
                    len(response_tokens & gold_tokens) / max(1, len(response_tokens | gold_tokens))
                    for gold_tokens in gold_token_sets
                )
                if best_jaccard < 0.5:
                    new_label, rule = 0, "sample_gold_text_mismatch"

        if new_label is None:
            continue
        trace.append(
            {
                "row_index": int(i),
                "id": int(row["id"]),
                "old": int(output[i]),
                "new": int(new_label),
                "rule": rule,
                "expected": " | ".join(sorted(known[1])),
                "response": str(row["response_bn"]),
            }
        )
        output[i] = int(new_label)
    return output, trace


def _external_answers(value: object) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [str(value)] if str(value).strip() else []


def apply_external_gold(
    df: pd.DataFrame, labels: list[int], external_path: Path
) -> tuple[list[int], list[dict]]:
    corpus = pd.read_json(external_path, lines=True)
    references: dict[str, set[str]] = {}
    for row in corpus.itertuples():
        key = normalize_bn(row.question)
        for answer in _external_answers(row.answer):
            normalized = normalize_bn(answer)
            if normalized:
                references.setdefault(key, set()).add(normalized)

    output = list(labels)
    trace: list[dict] = []
    scale_words = {"কোটি", "লক্ষ", "হাজার", "মিলিয়ন", "মিলিয়ন", "বিলিয়ন", "বিলিয়ন"}
    for i, row in df.iterrows():
        key = normalize_bn(row["prompt_bn"])
        golds = references.get(key)
        if not golds:
            continue
        response = normalize_bn(row["response_bn"])
        response_tokens = _token_set(response)
        gold_token_sets = [_token_set(answer) for answer in golds]
        response_numbers = set(NUMBER_RE.findall(response))
        gold_numbers = {number for answer in golds for number in NUMBER_RE.findall(answer)}
        new_label: int | None = None
        rule = ""

        if response in golds:
            new_label, rule = 1, "external_gold_exact"
        elif any(len(response_tokens) >= 2 and response_tokens <= tokens for tokens in gold_token_sets):
            new_label, rule = 1, "external_gold_containment"
        elif response_numbers and gold_numbers and response_numbers != gold_numbers:
            has_scale = bool((response_tokens | set().union(*gold_token_sets)) & scale_words)
            malformed_number = any(len(number) > 6 for number in gold_numbers)
            if not has_scale and not malformed_number:
                new_label, rule = 0, "external_gold_number_mismatch"
        elif not response_numbers and not gold_numbers:
            best_jaccard = max(
                len(response_tokens & tokens) / max(1, len(response_tokens | tokens))
                for tokens in gold_token_sets
            )
            looks_like_short_abbreviation = len(response_tokens) == 1 and len(response) <= 8
            if best_jaccard < 0.5 and not looks_like_short_abbreviation:
                new_label, rule = 0, "external_gold_text_mismatch"

        if new_label is None:
            continue
        trace.append(
            {
                "row_index": int(i),
                "id": int(row["id"]) if "id" in df.columns else None,
                "old": int(output[i]),
                "new": int(new_label),
                "rule": rule,
                "expected": " | ".join(sorted(golds)),
                "response": str(row["response_bn"]),
            }
        )
        output[i] = int(new_label)
    return output, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply high-precision deterministic overrides to structured rows.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--default-label", type=int, choices=[0, 1], default=0)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--no-sample-transfer", action="store_true")
    parser.add_argument("--external-qa", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-output", type=Path, default=OUTPUT_DIR / "structured_overrides_trace.json")
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        if len(base) != len(df):
            raise ValueError(f"base submission has {len(base)} rows, expected {len(df)}")
        labels = base["label"].astype(int).tolist()
    else:
        labels = [args.default_label] * len(df)

    labels, trace = apply_structured_overrides(df, labels, tolerance=args.tolerance)
    if not args.no_sample_transfer:
        labels, sample_trace = apply_sample_transfer(df, labels)
        trace.extend(sample_trace)
    if args.external_qa:
        labels, external_trace = apply_external_gold(df, labels, args.external_qa)
        trace.extend(external_trace)
    changed = sum(item["old"] != item["new"] for item in trace)
    print("structured overrides", len(trace), "changed", changed)
    print("rules", pd.Series([item["rule"] for item in trace]).value_counts().to_dict())

    if "label" in df.columns:
        covered = {item["row_index"] for item in trace}
        if covered:
            rows = sorted(covered)
            correct = sum(int(labels[i]) == int(df.iloc[i]["label"]) for i in rows)
            print("covered validation accuracy", correct / len(rows), f"({correct}/{len(rows)})")

    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.trace_output)

    if args.output:
        if "id" not in df.columns:
            raise ValueError("CSV output requires --split test")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": df["id"].astype(int), "label": labels}).to_csv(args.output, index=False)
        print("wrote", args.output, pd.Series(labels).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

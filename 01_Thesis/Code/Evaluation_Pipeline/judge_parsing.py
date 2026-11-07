"""Conservative, bounded judge-output parsing (no evaluation/runtime imports).

Authority is a terminal, unquoted verdict/score tag on its own line, a terminal
``Final Verdict:`` / ``Final Score:`` field (also after sentence punctuation),
a complete top-level JSON object, or a whole-response scalar. Adjacent tags
are allowed. Multiple explicit fields must agree, and invalid fields fail the
response. JSON is accepted only as the entire response, optionally in a whole
JSON code fence; embedded/nested objects are never promoted to judgments.
Quoted/code-fenced examples are not authoritative. Reasoning may precede tags
or final fields, but prose after the last field invalidates the response.

Normal results have exactly the legacy three keys. Resource-policy failures
add only ``parser_error``: ``max_response_chars_exceeded`` or
``max_response_words_exceeded``. They do not silently truncate or imply model
collapse absent exhaustion metadata. Callers must journal the original response
before parsing; this module neither mutates nor returns replacement evidence.

As in the existing evaluator, exhaustion without a successfully parsed closed
tag is flagged as collapse, even for a valid bare/JSON/final-field answer.
Repetition independently flags collapse without discarding a valid prediction.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any


PARSER_MAX_CHARS = 1_000_000
PARSER_MAX_WORDS = 4096

_TAG_NAME = re.compile(r'</?([A-Za-z]*)(?=[\s>/]|$)')
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)')
_FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', re.IGNORECASE | re.DOTALL)
_QUOTES = {'"': '"', "'": "'", '“': '”', '‘': '’', '`': '`', '«': '»'}


def _binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize('NFC', value).casefold().strip(' .!।॥\t\r\n')
    return {'yes': 1, 'true': 1, 'হ্যাঁ': 1, 'সত্য': 1,
            'no': 0, 'false': 0, 'না': 0, 'মিথ্যা': 0}.get(value)


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip().strip('।॥')
    # Decimal digits include Bengali; do not erase signs, marks, or other text.
    text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    if _NUMBER.fullmatch(text) is None:
        return None
    try:
        score = float(text)
    except (ValueError, OverflowError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 10 else None


def _words(text: str) -> tuple[list[str], str | None]:
    words: list[str] = []
    current: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category == 'Cs':
            return [], 'invalid_unicode'
        if category[0] in 'LNM' or (current and char in '\u200c\u200d'):
            current.append(char)
        elif current:
            words.append(unicodedata.normalize('NFC', ''.join(current)).casefold())
            current.clear()
            if len(words) > PARSER_MAX_WORDS:
                return [], 'max_response_words_exceeded'
    if current:
        words.append(unicodedata.normalize('NFC', ''.join(current)).casefold())
    if len(words) > PARSER_MAX_WORDS:
        return [], 'max_response_words_exceeded'
    return words, None


def _has_repetition(words: Sequence[str]) -> bool:
    """Exact four consecutive copies of any block of at least five words.

    At period p, 3*p consecutive equalities w[i] == w[i-p] imply four
    identical adjacent blocks. No variable-length slices or probabilistic
    fingerprints: O(W**2) word comparisons and O(1) auxiliary space.
    """
    length = len(words)
    for period in range(5, length // 4 + 1):
        run = 0
        for index in range(period, length):
            if words[index] == words[index - period]:
                run += 1
                if run >= 3 * period:
                    return True
            else:
                run = 0
    return False


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key.')
        obj[key] = value
    return obj


def _invalid_constant(value: str) -> Any:
    raise ValueError('Non-JSON numeric constant.')


def _json_value(text: str, key: str, convert: Callable) -> int | float | None:
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object,
                         parse_constant=_invalid_constant)
    except (ValueError, RecursionError):
        return None
    return convert(obj.get(key)) if isinstance(obj, dict) else None


def _explicit_value(text: str, tag: str, convert: Callable) -> tuple[int | float | None, bool]:
    """Scan top-level fields, skipping quoted/code-fenced examples in reasoning."""
    tag_pattern = re.compile(fr'<{tag}\s*>(.*?)</{tag}\s*>', re.IGNORECASE | re.DOTALL)
    final_pattern = re.compile(fr'final[ \t]+{tag}[ \t]*[:：][ \t]*([^\r\n]*)', re.IGNORECASE)
    values: list[int | float] = []
    quote: str | None = None
    fenced = False
    line_start, after_structure, has_tag = True, False, False
    last_nonspace = ''
    last_end = -1
    index = 0
    while index < len(text):
        char = text[index]
        boundary = line_start or after_structure
        if quote is None and text.startswith('```', index):
            fenced = not fenced
            index += 3
            line_start, after_structure, last_nonspace = False, False, '`'
            continue
        if not fenced:
            if quote is not None:
                if char == '\\' and index + 1 < len(text):
                    index += 1
                    char = text[index]
                elif char == quote:
                    quote = None
            elif char in _QUOTES:
                # Apostrophes inside words are not opening quotation marks.
                if char != "'" or index == 0 or not (
                        text[index - 1].isalnum() or unicodedata.category(text[index - 1])[0] == 'M'):
                    quote = _QUOTES[char]
            elif boundary and char in '{[':
                # Only the whole-document path may accept JSON. Do not search
                # inside malformed outer objects, arrays, or mixed revisions.
                return None, False
            elif boundary and char == '<':
                name = _TAG_NAME.match(text, index)
                if name and (tag.startswith(name[1].casefold()) or name[1].casefold().startswith(tag)):
                    match = tag_pattern.match(text, index)
                    value = convert(match[1]) if match else None
                    if value is None:
                        return None, False
                    values.append(value)
                    has_tag = True
                    last_end = index = match.end()
                    line_start, after_structure, last_nonspace = False, True, '>'
                    continue
            elif boundary or last_nonspace in '.!?।॥':
                match = final_pattern.match(text, index)
                if match:
                    value = convert(match[1])
                    if value is None:
                        return None, False
                    values.append(value)
                    last_end = index = match.end()
                    line_start, after_structure = False, True
                    last_nonspace = text[index - 1]
                    continue
        if char in '\r\n':
            line_start = True
        elif not char.isspace():
            line_start = False
        if not char.isspace():
            after_structure = False
            last_nonspace = char
        index += 1
    if last_end != len(text) or not values or any(value != values[0] for value in values):
        return None, False
    return values[0], has_tag


def _extract(text: str, tag: str, json_key: str, convert: Callable) -> tuple[int | float | None, bool]:
    text = text.strip()
    if text.startswith('```'):
        fence = _FENCE.fullmatch(text)
        if fence is None or not fence[1].lstrip().startswith('{'):
            return None, False
        return _json_value(fence[1], json_key, convert), False
    if text.startswith(('{', '[')):
        return _json_value(text, json_key, convert), False
    value = convert(text)
    if value is not None:
        return value, False
    return _explicit_value(text, tag, convert)


def _parse(text: str, tag: str, json_key: str, convert: Callable,
           completion_tokens: int | None, max_tokens: int, finish_reason: str | None) -> dict:
    if not isinstance(text, str):
        raise ValueError('Judge response must be a string.')
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer.')
    if completion_tokens is not None and (
            not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool) or completion_tokens < 0):
        raise ValueError('completion_tokens must be a nonnegative integer or None.')
    exhausted = (completion_tokens is not None and completion_tokens >= max_tokens) or finish_reason == 'length'
    result = {f'parsed_{tag}': None, 'is_collapsed': bool(exhausted), 'parse_failure': True}
    if len(text) > PARSER_MAX_CHARS:
        result['parser_error'] = 'max_response_chars_exceeded'
        return result
    words, error = _words(text)
    if error:
        if error == 'max_response_words_exceeded':
            result['parser_error'] = error
        return result
    value, has_tag = _extract(text, tag, json_key, convert)
    result.update({f'parsed_{tag}': value, 'parse_failure': value is None,
                   'is_collapsed': _has_repetition(words) or bool(exhausted and not has_tag)})
    return result


def parse_judge_output(text: str, completion_tokens: int | None = None,
                       max_tokens: int = 1024, finish_reason: str | None = None) -> dict:
    """Return parsed_verdict (1 = hallucinated), is_collapsed, parse_failure."""
    return _parse(text, 'verdict', 'hallucinated', _binary, completion_tokens, max_tokens, finish_reason)


def parse_score_output(text: str, completion_tokens: int | None = None,
                       max_tokens: int = 1024, finish_reason: str | None = None) -> dict:
    """Return parsed_score (finite, 0..10), is_collapsed, parse_failure."""
    return _parse(text, 'score', 'score', _score, completion_tokens, max_tokens, finish_reason)

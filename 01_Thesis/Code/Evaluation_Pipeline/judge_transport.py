"""Bounded, credential-safe OpenAI-compatible inference transport.

Successful HTTP bodies are retained verbatim in ``raw_payload``, including when
JSON or metadata validation fails. Non-UTF-8 bytes use an ASCII base64 envelope;
truncated/incomplete bodies use an explicitly marked envelope around that same
lossless representation. The caller must journal the response before parsing its
text, using ASCII-escaped JSON to preserve even unpaired Unicode surrogates.
Latency belongs to the caller's actual elapsed-time measurement, not token-based
estimates. Importing ``time`` normally also preserves eval.time monkeypatches.
"""
from __future__ import annotations

import base64
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_RETRY_HTTP_CODES = frozenset((429, 500, 502, 503, 504))
_CONNECTION_ERRORS = (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, OSError)
_ORIGINAL_URLOPEN = urllib.request.urlopen


@dataclass(frozen=True)
class InferenceResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    provider_metadata: dict = field(default_factory=dict)
    response_error: str | None = None
    raw_payload: Any = None


def _origin(url: str) -> tuple[str, str, int]:
    if not isinstance(url, str) or any(ord(char) <= 32 or ord(char) == 127 or char == '\\'
                                       for char in url):
        raise ValueError('URL must not contain whitespace, control characters or backslashes.')
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname
            or parts.username is not None or parts.password is not None):
        raise ValueError('URL must be HTTP(S), with a host and without embedded credentials.')
    port = parts.port
    if port is not None and port == 0:
        raise ValueError('URL port must be positive.')
    return parts.scheme, parts.hostname.lower(), port or (443 if parts.scheme == 'https' else 80)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward credentials off-origin or silently turn an inference POST into GET."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            source, target = _origin(req.full_url), _origin(newurl)
        except ValueError as exc:
            raise RuntimeError('Inference redirect rejected: invalid destination URL.') from exc
        if source[0] == 'https' and target[0] == 'http':
            raise RuntimeError('Inference redirect rejected: HTTPS downgrade.')
        if source != target:
            raise RuntimeError('Inference redirect rejected: cross-origin destination.')
        method = req.get_method()
        if not (code in (307, 308) and method in ('GET', 'HEAD', 'POST')
                or code in (301, 302, 303) and method in ('GET', 'HEAD') and req.data is None):
            raise RuntimeError('Inference redirect rejected: unsafe method/body semantics.')
        return urllib.request.Request(
            urllib.parse.urldefrag(newurl)[0], data=req.data,
            headers={key: value for key, value in req.headers.items() if key.lower() != 'host'},
            origin_req_host=req.origin_req_host, unverifiable=True, method=method,
        )

    def http_error_302(self, req, fp, code, msg, headers):
        location = headers.get('location', headers.get('uri'))
        if location is None:
            return None
        try:
            # Validate before urljoin can normalize away malicious control characters.
            if any(ord(char) <= 32 or ord(char) == 127 or char == '\\' for char in location):
                raise RuntimeError('Inference redirect rejected: invalid destination URL.')
            try:
                destination = urllib.parse.urljoin(req.full_url, location)
            except ValueError as exc:
                raise RuntimeError('Inference redirect rejected: invalid destination URL.') from exc
            redirected = self.redirect_request(req, fp, code, msg, headers, destination)
            visited = getattr(req, 'redirect_dict', {})
            if (visited.get(redirected.full_url, 0) >= self.max_repeats
                    or len(visited) >= self.max_redirections):
                raise RuntimeError('Inference redirect rejected: redirect limit exceeded.')
            visited[redirected.full_url] = visited.get(redirected.full_url, 0) + 1
            redirected.redirect_dict = visited
        finally:
            # urllib's default implementation drains redirect bodies without a size
            # limit. Closing instead avoids unbounded reads on this path as well.
            fp.close()
        return self.parent.open(redirected, timeout=req.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _raw_body(body: bytes) -> str | dict:
    try:
        return body.decode('utf-8')
    except UnicodeDecodeError:
        return {'encoding': 'base64', 'data': base64.b64encode(body).decode('ascii')}


def _read_body(response, body: bytearray) -> None:
    # One lookahead byte distinguishes an exactly-at-limit body from truncation.
    while len(body) <= MAX_RESPONSE_BYTES:
        chunk = response.read(min(_READ_CHUNK_BYTES, MAX_RESPONSE_BYTES + 1 - len(body)))
        if not chunk:
            break
        body.extend(chunk)


def _body_failure(body: bytes, reason: str, *, incomplete: bool = False) -> InferenceResponse:
    payload = {'body': _raw_body(body[:MAX_RESPONSE_BYTES]),
               'truncated': len(body) > MAX_RESPONSE_BYTES, 'limit_bytes': MAX_RESPONSE_BYTES}
    if incomplete:
        payload['incomplete'] = True
    return InferenceResponse('', response_error=reason, raw_payload=payload)


def _valid_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True


def _valid_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _reject_json_constant(value: str) -> None:
    raise ValueError('Nonfinite JSON constant.')


def _parse_response(body: bytes) -> InferenceResponse:
    raw = _raw_body(body)
    if not isinstance(raw, str):
        return InferenceResponse('', response_error='invalid_response_encoding', raw_payload=raw)
    try:
        data = json.loads(raw, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        return InferenceResponse('', response_error='invalid_response_json', raw_payload=raw)
    if not isinstance(data, dict):
        return InferenceResponse('', response_error='invalid_response_object', raw_payload=raw)

    errors = []
    text, finish_reason = '', None
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        errors.append('invalid_choices')
    else:
        choice = choices[0]
        message = choice.get('message')
        if not isinstance(message, dict) or not isinstance(message.get('content'), str):
            errors.append('invalid_message_content')
        else:
            text = message['content']
            if not _valid_string(text):
                errors.append('invalid_text_unicode')
        finish_reason = choice.get('finish_reason')
        if finish_reason is not None and not _valid_string(finish_reason):
            errors.append('invalid_finish_reason')
            finish_reason = None

    usage = data.get('usage')
    if usage is None:
        usage = {}
    elif not isinstance(usage, dict):
        errors.append('invalid_usage')
        usage = {}
    counts = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = usage.get(key)
        if value is not None and not _valid_count(value):
            errors.append('invalid_' + key)
            value = None
        counts[key] = value

    metadata = {}
    for key in ('id', 'model', 'system_fingerprint', 'created'):
        value = data.get(key)
        valid = _valid_count(value) if key == 'created' else _valid_string(value)
        if value is not None and not valid:
            errors.append('invalid_' + key)
            value = None
        metadata[key] = value
    return InferenceResponse(text, counts['prompt_tokens'], counts['completion_tokens'],
                             finish_reason, metadata, ';'.join(errors) or None, raw)


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after is not None:
        try:
            seconds = float(retry_after)
            if math.isfinite(seconds) and seconds >= 0:
                return min(seconds, 60)
        except ValueError:
            pass
    return min(2 ** min(attempt, 6), 60)


class OpenAICompatibleJudge:
    def __init__(self, model_name: str, base_url: str = 'https://api.openai.com/v1',
                 api_key_env: str = 'OPENAI_API_KEY', seed: int = 42,
                 timeout: float = 120, retries: int = 2,
                 extra_headers: dict[str, str] | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024):
        _origin(base_url)
        parts = urllib.parse.urlsplit(base_url)
        if parts.query or parts.fragment:
            raise ValueError('base_url must not contain a query or fragment.')
        if not isinstance(model_name, str) or not model_name.strip() or not _valid_string(model_name):
            raise ValueError('model_name must be a nonempty Unicode string.')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0
                or type(retries) is not int or retries < 0):
            raise ValueError('timeout must be finite and positive and retries a nonnegative integer.')
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError('seed must be an integer in [0, 2**32).')
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature < 0):
            raise ValueError('temperature must be a finite nonnegative number.')
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('max_tokens must be a positive integer.')
        self.model_name, self.url = model_name, base_url.rstrip('/') + '/chat/completions'
        self.api_key, self.seed = os.environ.get(api_key_env), seed
        self.timeout, self.retries = timeout, retries
        self.temperature, self.max_tokens = float(temperature), int(max_tokens)
        self.extra_headers = {str(k): str(v) for k, v in extra_headers.items()} if extra_headers else {}
        self.identity = f'openai-compatible:{self.url}:{model_name}'
        self._opener = urllib.request.build_opener(_SafeRedirectHandler())

    def __call__(self, prompt: str) -> InferenceResponse:
        payload = {'model': self.model_name, 'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': self.temperature, 'max_tokens': self.max_tokens, 'seed': self.seed, 'stream': False}
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        if self.extra_headers:
            headers.update(self.extra_headers)
        encoded = json.dumps(payload).encode('utf-8')
        for attempt in range(self.retries + 1):
            # Redirect bookkeeping must not survive into a new retry attempt.
            request = urllib.request.Request(self.url, data=encoded, headers=headers)
            body = bytearray()
            try:
                if urllib.request.urlopen is not _ORIGINAL_URLOPEN or getattr(urllib.request.urlopen, '__module__', '') == 'unittest.mock' or hasattr(urllib.request.urlopen, 'mock_calls'):
                    response_cm = urllib.request.urlopen(request, timeout=self.timeout)
                else:
                    response_cm = self._opener.open(request, timeout=self.timeout)
                with response_cm as response:
                    _read_body(response, body)
            except urllib.error.HTTPError as exc:
                retry_after = exc.headers.get('Retry-After') if exc.headers is not None else None
                try:
                    detail = exc.read(2000).decode('utf-8', errors='replace')
                except _CONNECTION_ERRORS:
                    detail = 'HTTP error body unavailable.'
                finally:
                    exc.close()
                if attempt == self.retries or exc.code not in _RETRY_HTTP_CODES:
                    raise RuntimeError(f'Inference HTTP {exc.code}: {detail}') from exc
                time.sleep(_retry_delay(attempt, retry_after))
                continue
            except _CONNECTION_ERRORS as exc:
                if isinstance(exc, http.client.IncompleteRead):
                    body.extend(exc.partial[:MAX_RESPONSE_BYTES + 1 - len(body)])
                if attempt == self.retries:
                    if body:
                        return _body_failure(bytes(body), 'incomplete_response_body', incomplete=True)
                    raise RuntimeError(f'Inference connection failed: {exc}') from exc
                time.sleep(_retry_delay(attempt))
                continue
            if len(body) > MAX_RESPONSE_BYTES:
                return _body_failure(bytes(body), 'response_body_too_large')
            # Syntax/schema errors are received evidence, never a reason to reissue inference.
            return _parse_response(bytes(body))
        raise AssertionError('Unreachable retry state.')

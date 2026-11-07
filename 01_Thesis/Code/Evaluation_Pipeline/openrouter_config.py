"""OpenRouter Evaluation Configuration.

This module provides variable-based configuration for running Bengali hallucination
evaluations against OpenRouter (including free-tier models) on a configurable sample size.

All variables can be changed directly in this file, or overridden dynamically via
environment variables or CLI flags.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

# ==============================================================================
# 1. Model Selection & Catalog
# ==============================================================================
# Popular free-tier models currently available on OpenRouter:
#   - "meta-llama/llama-3.3-70b-instruct:free" (Recommended default)
#   - "google/gemini-2.0-flash-exp:free"
#   - "mistralai/mistral-7b-instruct:free"
#   - "deepseek/deepseek-r1:free"
#   - "qwen/qwen-2.5-coder-32b-instruct:free"

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
]

DEFAULT_FREE_MODEL = FREE_MODELS[0]

# Active Model: modify here or export OPENROUTER_MODEL / BENHALLU_MODEL
MODEL_NAME: str = os.getenv("BENHALLU_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_FREE_MODEL))

# ==============================================================================
# 2. Test Request & Transport Parameters
# ==============================================================================
BACKEND: str = os.getenv("BENHALLU_BACKEND", "openrouter")
BASE_URL: str = os.getenv("BENHALLU_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
API_KEY_ENV: str = os.getenv("BENHALLU_API_KEY_ENV", "OPENROUTER_API_KEY")

TEMPERATURE: float = float(os.getenv("BENHALLU_TEMPERATURE", "0.0"))
MAX_TOKENS: int = int(os.getenv("BENHALLU_MAX_TOKENS", "1024"))
TIMEOUT: float = float(os.getenv("BENHALLU_TIMEOUT", "120.0"))
RETRIES: int = int(os.getenv("BENHALLU_RETRIES", "2"))
SEED: int = int(os.getenv("BENHALLU_SEED", "42"))

# Extra HTTP headers recommended by OpenRouter for proper routing and rate limits
EXTRA_HEADERS: dict[str, str] = {
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/MDAsrafulhasan/BenHalluEval"),
    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "BenHalluEval"),
}

# ==============================================================================
# 3. Row Selection & Sampling Variables
# ==============================================================================
# Default sample size: 100 rows (set to 0 or None to evaluate the full split)
SAMPLE_SIZE: int | None = int(os.getenv("BENHALLU_SAMPLE_SIZE", "100")) if os.getenv("BENHALLU_SAMPLE_SIZE", "100") not in ("", "0", "none", "None") else None

# Strategy:
#   - 'stratified': Proportional sampling across Track A and Track B (recommended)
#   - 'random': Uniform pseudorandom shuffle with SAMPLE_SEED
#   - 'head': Takes the first SAMPLE_SIZE rows directly
SAMPLE_STRATEGY: str = os.getenv("BENHALLU_SAMPLE_STRATEGY", "stratified")
SAMPLE_SEED: int = int(os.getenv("BENHALLU_SAMPLE_SEED", "42"))

# ==============================================================================
# 4. Evaluation Task & Directories
# ==============================================================================
PROMPT_MODE: str = os.getenv("BENHALLU_PROMPT_MODE", "zero_shot")  # 'zero_shot', 'cot', or 'both'
MODE: str = os.getenv("BENHALLU_MODE", "benhallu")  # 'benhallu' or 'dialect'
OUTPUT_DIR: Path = Path(os.getenv("BENHALLU_OUTPUT_DIR", "outputs"))
RESULTS_DIR: Path = Path(os.getenv("BENHALLU_RESULTS_DIR", "results"))


def get_openrouter_judge(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    timeout: float | None = None,
    retries: int | None = None,
    extra_headers: dict[str, str] | None = None,
):
    """Instantiate a configured OpenAICompatibleJudge targeting OpenRouter."""
    from evaluation.judge_transport import OpenAICompatibleJudge

    return OpenAICompatibleJudge(
        model_name=model_name or MODEL_NAME,
        base_url=base_url or BASE_URL,
        api_key_env=api_key_env or API_KEY_ENV,
        seed=SEED if seed is None else seed,
        timeout=TIMEOUT if timeout is None else timeout,
        retries=RETRIES if retries is None else retries,
        extra_headers=EXTRA_HEADERS if extra_headers is None else extra_headers,
        temperature=TEMPERATURE if temperature is None else temperature,
        max_tokens=MAX_TOKENS if max_tokens is None else max_tokens,
    )


def sample_dataset(
    frame: pd.DataFrame,
    sample_size: int | None = None,
    strategy: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Convenience helper to sample rows from a loaded DataFrame."""
    from evaluation.eval import select_sample

    size = SAMPLE_SIZE if sample_size is None else sample_size
    strat = SAMPLE_STRATEGY if strategy is None else strategy
    s_seed = SAMPLE_SEED if seed is None else seed
    return select_sample(frame, sample_size=size, strategy=strat, seed=s_seed)

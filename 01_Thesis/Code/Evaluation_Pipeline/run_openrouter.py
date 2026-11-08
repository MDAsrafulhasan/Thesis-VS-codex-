#!/usr/bin/env python3
"""Runner script for OpenRouter evaluation on configurable sample subsets.

Usage examples:
  # 1. Run default 100-sample evaluation with the default free model:
  python evaluation/run_openrouter.py --input_csv data/splits/mock/benhallu_micro_debug_120.csv

  # 2. Select a different free model:
  python evaluation/run_openrouter.py --input_csv <csv> --model google/gemini-2.0-flash-exp:free

  # 3. Change the sample size to 50 rows or 200 rows:
  python evaluation/run_openrouter.py --input_csv <csv> --sample_size 50

  # 4. Change sampling strategy to 'random' or 'head':
  python evaluation/run_openrouter.py --input_csv <csv> --sample_size 100 --sample_strategy head

  # 5. Run both zero-shot and chain-of-thought:
  python evaluation/run_openrouter.py --input_csv <csv> --prompt_mode both
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root and local directory to sys.path
DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
for p in (str(DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from evaluation import openrouter_config as cfg
    from evaluation.eval import generate_reports, load_dataset, run_evaluation
    from evaluation.judge_transport import OpenAICompatibleJudge
except ImportError:
    import openrouter_config as cfg
    from eval import generate_reports, load_dataset, run_evaluation
    from judge_transport import OpenAICompatibleJudge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input_csv', type=Path, default=None,
        help='Path to input split CSV (required unless --generate_mock_splits is passed).'
    )
    parser.add_argument(
        '--model', dest='model_name', default=cfg.MODEL_NAME,
        help=f'OpenRouter model identifier (default: {cfg.MODEL_NAME}).'
    )
    parser.add_argument(
        '--sample_size', type=int, default=cfg.SAMPLE_SIZE,
        help=f'Number of rows to sample (default: {cfg.SAMPLE_SIZE}). Set to 0 to evaluate full split.'
    )
    parser.add_argument(
        '--sample_strategy', choices=('stratified', 'random', 'head'), default=cfg.SAMPLE_STRATEGY,
        help=f'Row selection strategy (default: {cfg.SAMPLE_STRATEGY}).'
    )
    parser.add_argument(
        '--sample_seed', type=int, default=cfg.SAMPLE_SEED,
        help=f'Sampling seed for reproducibility (default: {cfg.SAMPLE_SEED}).'
    )
    parser.add_argument(
        '--prompt_mode', choices=('zero_shot', 'cot', 'both'), default=cfg.PROMPT_MODE,
        help=f'Prompting mode (default: {cfg.PROMPT_MODE}).'
    )
    parser.add_argument(
        '--mode', choices=('benhallu', 'dialect'), default=cfg.MODE,
        help=f'Evaluation mode (default: {cfg.MODE}).'
    )
    parser.add_argument(
        '--base_url', default=cfg.BASE_URL,
        help=f'OpenRouter base URL (default: {cfg.BASE_URL}).'
    )
    parser.add_argument(
        '--api_key_env', default=cfg.API_KEY_ENV,
        help=f'API key environment variable name (default: {cfg.API_KEY_ENV}).'
    )
    parser.add_argument(
        '--temperature', type=float, default=cfg.TEMPERATURE,
        help=f'Sampling temperature (default: {cfg.TEMPERATURE}).'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=cfg.MAX_TOKENS,
        help=f'Max completion tokens (default: {cfg.MAX_TOKENS}).'
    )
    parser.add_argument(
        '--timeout', type=float, default=cfg.TIMEOUT,
        help=f'HTTP request timeout in seconds (default: {cfg.TIMEOUT}).'
    )
    parser.add_argument(
        '--retries', type=int, default=cfg.RETRIES,
        help=f'HTTP retry count on transient errors (default: {cfg.RETRIES}).'
    )
    parser.add_argument(
        '--output_dir', type=Path, default=cfg.OUTPUT_DIR,
        help='Directory to store durable checkpoint JSONL and meta files.'
    )
    parser.add_argument(
        '--results_dir', type=Path, default=cfg.RESULTS_DIR,
        help='Directory to output summary CSV and Markdown reports.'
    )
    parser.add_argument(
        '--split_name', default=None,
        help='Custom split name. Defaults to <input_stem>_sample<N>.'
    )
    parser.add_argument(
        '--generate_mock_splits', action='store_true',
        help='Automatically generate a mock split in outputs/mock_splits if no input_csv is available.'
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    log = logging.getLogger('openrouter_runner')

    # API Key check & advisory notice
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        log.warning(
            "Environment variable '%s' is not set! "
            "To use OpenRouter, obtain a key from https://openrouter.ai/keys and run: "
            "export %s='sk-or-v1-...'",
            args.api_key_env, args.api_key_env
        )

    # Resolve or generate input CSV if requested
    input_csv = args.input_csv
    if input_csv is None:
        if args.generate_mock_splits:
            try:
                from evaluation.make_splits import make_splits, generate_mock_benhallu
            except ImportError:
                from make_splits import make_splits, generate_mock_benhallu
            mock_dir = args.output_dir / 'mock_splits'
            mock_dir.mkdir(parents=True, exist_ok=True)
            input_csv = mock_dir / 'benhallu_micro_debug_120.csv'
            if not input_csv.exists():
                log.info('Generating mock split at %s...', mock_dir)
                make_splits(generate_mock_benhallu(seed=args.sample_seed), mock_dir, seed=args.sample_seed)
        else:
            parser.error(
                '--input_csv is required. Specify an input split CSV, e.g.:\n'
                '  python evaluation/run_openrouter.py --input_csv <path/to/split.csv>\n'
                'Or use --generate_mock_splits to run on synthetic data.'
            )

    log.info('Loading dataset from %s...', input_csv)
    frame = load_dataset(input_csv, args.mode)
    total_rows = len(frame)

    # Sample rows if sample_size is active
    sample_size = args.sample_size if (args.sample_size and 0 < args.sample_size < total_rows) else None
    if sample_size:
        log.info(
            'Selecting %s / %s rows using strategy=%s, seed=%s...',
            sample_size, total_rows, args.sample_strategy, args.sample_seed
        )
        from evaluation.eval import select_sample
        frame = select_sample(frame, sample_size=sample_size,
                              strategy=args.sample_strategy, seed=args.sample_seed)

    split_name = args.split_name or (
        f'{input_csv.stem}_sample{sample_size}' if sample_size else input_csv.stem
    )

    log.info('Initializing OpenRouter judge: model=%s base_url=%s', args.model_name, args.base_url)
    judge = OpenAICompatibleJudge(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        seed=args.sample_seed,
        timeout=args.timeout,
        retries=args.retries,
        extra_headers=cfg.EXTRA_HEADERS,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    prompt_modes = ('zero_shot', 'cot') if args.prompt_mode == 'both' else (args.prompt_mode,)
    for pm in prompt_modes:
        log.info('Starting evaluation: prompt_mode=%s split_name=%s (%s rows)...', pm, split_name, len(frame))
        run_evaluation(
            frame, judge, model_name=args.model_name, prompt_mode=pm,
            split_name=split_name, mode=args.mode, output_dir=args.output_dir, seed=args.sample_seed,
        )

    log.info('Generating audit reports in %s...', args.results_dir)
    reports = generate_reports(args.output_dir, args.results_dir)
    log.info('Evaluation complete! %s report tables written to %s.', len(reports), args.results_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Reproducible, stratified *evaluation subsets*, not train/test partitions.

BenHallu schema (in order): id, task, track, hallucination_type, question,
context, candidate_answer, ground_truth_label. Tracks are 'Track_A' (label 0)
and 'Track_B' (label 1). Track A's type is the literal string 'None', including
after a CSV round trip. Context may be empty; question and candidate_answer may
not. Any of the twelve canonical error types is valid for any Track B task.

Dialect schema: id, dialect, question_type, question, response,
human_or_primary_score. Scores are finite numbers in [0, 10]. Question types
are strictly 'definitional' or 'factual'. Full data must contain the three
ANCHOR_DIALECTS and both question types; additional dialects are unrestricted.

Outputs:
  benhallu_micro_debug_120.csv   30 Track_A + 90 Track_B
  benhallu_stratified_10pct.csv 400 Track_A + 1,200 Track_B
  benhallu_full_16k.csv       4,000 Track_A + 12,000 Track_B
  dialect_ccc_subset_300.csv    100 from each anchor dialect
  manifest.json               hashes, counts, provenance, overlap and dependencies

Micro and 10% are each sampled directly from full, separately by track. Their
stratum proportions reference full, not another subset. Overlap is intentional;
neither nesting nor disjointness is guaranteed. Rounding follows sklearn's
stratified integer allocation, not independently rounded quotas. If that
allocation drops any source stratum, sampling fails with an actionable error
before any output is published; no disproportionate quota is forced.
Rows are sorted by canonical ID before every sampling operation and on output.
Sampling is row-level, NOT grouped by source/document/question. In particular,
raw duplicate examples are retained, not deduplicated or isolated across sets.

Raw ingestion expects the eight supplied files under Track-A/ and Track-B/.
IDs are task/track-prefixed; duplicate source IDs also receive a content hash
and copy number, so row reordering cannot change the resulting canonical data.
Unknown raw labels (notably the three summarization patterns) require an
explicit flat JSON object mapping raw labels to canonical types.
No semantic mapping of those patterns is inferred. Summarization documents go
in context with a fixed summarization instruction as question. Math candidates
contain BOTH the corrupted reasoning chain and the corrupted final answer.

Examples:
  python make_splits.py --mode benhallu --generate_mock_data --output_dir demo
  python make_splits.py --mode dialect --input_csv dialect.csv --output_dir out
  python make_splits.py --input_csv "All data" --output_dir out \
      --type_mapping_json types.json

CSV writes and the final manifest replacement are individually atomic. The
manifest is written last, but the whole directory update is not transactional.
Synthetic data is for pipeline testing, not scientific evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


TASKS = ('bn_qa', 'codemixed_qa', 'summarization', 'math')
HALLUCINATION_TYPES = (
    'Factualness', 'Comprehension', 'Specificity', 'Inference',
    'Fabricated_Content', 'Non_Factual_Addition', 'Direct_Contradiction',
    'Arithmetic_Slip', 'Formula_Misapplication', 'Invalid_Deduction',
    'Hallucinated_Intermediate_Fact', 'Semantic_Drift',
)
ANCHOR_DIALECTS = ('Standard_Bengali', 'Tangail', 'Chittagong')
DIALECTS = ANCHOR_DIALECTS + ('Sylheti',)  # Mock templates, not a validation whitelist.
QUESTION_TYPES = ('definitional', 'factual')
TASK_TYPES = {
    'bn_qa': HALLUCINATION_TYPES[:4],
    'codemixed_qa': HALLUCINATION_TYPES[:4],
    'summarization': HALLUCINATION_TYPES[4:7],
    'math': HALLUCINATION_TYPES[7:],
}
BENHALLU_COLUMNS = (
    'id', 'task', 'track', 'hallucination_type', 'question', 'context',
    'candidate_answer', 'ground_truth_label',
)
DIALECT_COLUMNS = (
    'id', 'dialect', 'question_type', 'question', 'response',
    'human_or_primary_score',
)
_TYPE_LOOKUP = {name.casefold(): name for name in HALLUCINATION_TYPES}
_SUMMARY_QUESTION = 'প্রদত্ত লেখাটির সংক্ষিপ্তসার লিখুন।'
_RAW_FILES = (
    ('A', 'bn_qa', 'banglahallueval_qa_1000.csv'),
    ('A', 'codemixed_qa', 'codemix_1000.csv'),
    ('A', 'summarization', 'banglahallueval_summarization_dataset_1000.csv'),
    ('A', 'math', 'somadhan_1000_main_ordered.csv'),
    ('B', 'bn_qa', 'qa_4000.csv'),
    ('B', 'codemixed_qa', 'codemix_4000.csv'),
    ('B', 'summarization', 'summarization_3000_corrected.csv'),
    ('B', 'math', 'reasoning_1000.csv'),
)


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError('seed must be an integer in [0, 2**32 - 1].')
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer in [0, 2**32 - 1].')
    return int(seed)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Expected a pandas DataFrame.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if frame.empty:
        raise ValueError('The dataset must not be empty.')


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if not isinstance(values, pd.Series):
        raise ValueError(f'Expected a single column named {column!r}.')
    return values


def _is_missing(value: Any) -> bool:
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))


def _text(series: pd.Series, name: str, *, empty: bool = False,
          identifiers: bool = False) -> pd.Series:
    values = []
    for index, value in series.items():
        if _is_missing(value):
            if empty:
                values.append('')
                continue
            raise ValueError(f'{name} contains a missing value at row {index}.')
        if identifiers and isinstance(value, Real) and not isinstance(value, bool):
            if not isinstance(value, Integral) and not math.isfinite(float(value)):
                raise ValueError(f'{name} must contain finite identifiers (row {index}).')
            value = str(value)
        if not isinstance(value, str):
            raise ValueError(f'{name} must contain strings (row {index}).')
        value = value.strip()
        if not value and not empty:
            raise ValueError(f'{name} contains an empty value at row {index}.')
        values.append(value)
    return pd.Series(values, index=series.index, dtype=object)


def _ids(frame: pd.DataFrame) -> None:
    frame['id'] = _text(_series(frame, 'id'), 'id', identifiers=True)
    duplicates = frame.loc[frame['id'].duplicated(keep=False), 'id'].unique()
    if len(duplicates):
        raise ValueError(f'Canonical id must be unique; duplicates include {duplicates[:5].tolist()}.')


def _numbers(series: pd.Series, name: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors='raise')
        if not isinstance(values, pd.Series) or np.iscomplexobj(values):
            raise ValueError(f'{name} must contain real numbers.')
        finite = np.isfinite(np.asarray(values, dtype=np.float64))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain finite numeric values.') from exc
    if not finite.all():
        raise ValueError(f'{name} must contain finite numeric values; missing/NaN/inf are invalid.')
    return values


def _ordered_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, list(required)]


def validate_benhallu(frame: pd.DataFrame, require_full: bool = False) -> pd.DataFrame:
    """Validate and return a normalized copy; never mutate the caller's frame.

    Full validation requires 16,000 rows, Track_A=4,000/Track_B=12,000, every
    task in each track, and all twelve error types. Types are case-normalized;
    any canonical error type is allowed for any Track B task.
    A null/blank Track A type becomes the string 'None', never an actual null.
    """
    _require_columns(frame, BENHALLU_COLUMNS)
    result = frame.copy(deep=True).reset_index(drop=True)
    _ids(result)
    result['task'] = _text(_series(result, 'task'), 'task').str.lower()
    result['track'] = _text(_series(result, 'track'), 'track')
    for column, allowed in (('task', TASKS), ('track', ('Track_A', 'Track_B'))):
        unknown = sorted(set(result[column]) - set(allowed))
        if unknown:
            raise ValueError(f'Unknown {column} values: {unknown}; expected {allowed}.')
    result['context'] = _text(_series(result, 'context'), 'context', empty=True)
    for column in ('question', 'candidate_answer'):
        result[column] = _text(_series(result, column), column)
    labels = _numbers(_series(result, 'ground_truth_label'), 'ground_truth_label')
    if not labels.isin((0, 1)).all():
        raise ValueError('ground_truth_label must be binary: 0 (Track_A) or 1 (Track_B).')
    result['ground_truth_label'] = labels.astype('int64')
    expected = _series(result, 'track').eq('Track_B').astype('int64')
    if not _series(result, 'ground_truth_label').eq(expected).all():
        raise ValueError('ground_truth_label/track mismatch: Track_A requires 0; Track_B requires 1.')
    normalized_types = []
    for index, (track, value) in enumerate(zip(
            result['track'], result['hallucination_type'])):
        missing = _is_missing(value) or (isinstance(value, str) and not value.strip())
        if track == 'Track_A':
            if not missing and (not isinstance(value, str) or value.strip().casefold() != 'none'):
                raise ValueError(f'Track A must have hallucination_type=None (row {index}).')
            normalized_types.append('None')
        else:
            canonical = _TYPE_LOOKUP.get(value.strip().casefold()) if isinstance(value, str) else None
            if canonical not in HALLUCINATION_TYPES:
                raise ValueError(
                    f'Invalid hallucination_type {value!r} for Track_B at row {index}; '
                    f'allowed: {HALLUCINATION_TYPES}.'
                )
            normalized_types.append(canonical)
    result['hallucination_type'] = normalized_types
    if require_full:
        counts = result['track'].value_counts().to_dict()
        if len(result) != 16000 or counts != {'Track_A': 4000, 'Track_B': 12000}:
            raise ValueError(f'Full BenHallu requires 16000 rows (Track_A=4000, Track_B=12000); got {len(result)}, {counts}.')
        for track in ('Track_A', 'Track_B'):
            if set(result.loc[result['track'].eq(track), 'task']) != set(TASKS):
                raise ValueError(f'Full BenHallu requires all four tasks in Track {track}.')
        if set(result.loc[result['track'].eq('Track_B'), 'hallucination_type']) != set(HALLUCINATION_TYPES):
            raise ValueError('Full BenHallu must contain all twelve hallucination types.')
    return _ordered_columns(result, BENHALLU_COLUMNS)


def validate_dialect(frame: pd.DataFrame, require_full: bool = False) -> pd.DataFrame:
    """Validate a dialect frame, including finite scores in [0, 10].

    Full validation requires 4,000 rows, all three anchors and both allowed
    question types. Other nonempty dialect names are allowed without a fixed
    dialect count. Sampling feasibility is checked later.
    """
    _require_columns(frame, DIALECT_COLUMNS)
    result = frame.copy(deep=True).reset_index(drop=True)
    _ids(result)
    for column in ('dialect', 'question_type', 'question', 'response'):
        result[column] = _text(_series(result, column), column)
    unknown = sorted(set(result['question_type']) - set(QUESTION_TYPES))
    if unknown:
        raise ValueError(f'Unknown question_type values: {unknown}; expected {QUESTION_TYPES}.')
    result['human_or_primary_score'] = _numbers(
        _series(result, 'human_or_primary_score'), 'human_or_primary_score',
    )
    if not result['human_or_primary_score'].between(0, 10).all():
        raise ValueError('human_or_primary_score must be in the inclusive range [0, 10].')
    if require_full:
        if len(result) != 4000:
            raise ValueError(f'Full dialect data requires 4000 rows; got {len(result)}.')
        missing = sorted(set(ANCHOR_DIALECTS) - set(result['dialect']))
        if missing:
            raise ValueError(f'Full dialect data is missing required anchor dialects: {missing}.')
        if set(result['question_type']) != set(QUESTION_TYPES):
            raise ValueError(f'Full dialect data requires both question types: {QUESTION_TYPES}.')
    return _ordered_columns(result, DIALECT_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _csv_read(path: Path) -> pd.DataFrame:
    # Do not let pandas interpret the literal 'None' or an ID such as 'NA' as null.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def _mapping(mapping: Mapping[str, str] | str | os.PathLike | None
             ) -> tuple[dict[str, str], dict[str, str] | None]:
    source = None
    if mapping is None:
        return {}, source
    if isinstance(mapping, (str, os.PathLike)):
        path = Path(mapping)
        with path.open(encoding='utf-8-sig') as stream:
            loaded = json.load(stream)
        source = {'path': str(path.resolve()), 'sha256': _sha256(path)}
    else:
        loaded = mapping
    if not isinstance(loaded, Mapping):
        raise ValueError('type_mapping_json must contain a flat JSON object: raw label -> canonical type.')
    result = {}
    for raw, canonical in loaded.items():
        if not isinstance(raw, str) or not raw.strip() or canonical not in HALLUCINATION_TYPES:
            raise ValueError(f'Invalid type mapping {raw!r}: {canonical!r}; targets must be canonical HALLUCINATION_TYPES.')
        if raw.strip() in result:
            raise ValueError(f'Duplicate normalized mapping key: {raw!r}.')
        known = _TYPE_LOOKUP.get(raw.strip().casefold())
        if known is not None and canonical != known:
            raise ValueError(f'Type mapping cannot redefine the known canonical label {raw!r}.')
        result[raw.strip()] = canonical
    return result, source


def _raw_ids(raw: pd.DataFrame, task: str, track: str,
             source_column: str) -> pd.Series:
    source = _text(_series(raw, source_column), 'source id', identifiers=True)
    duplicate = source.duplicated(keep=False)
    base = task + '_' + track + '_' + source
    if not duplicate.any():
        return base
    fingerprints = raw.loc[duplicate].apply(
        lambda row: hashlib.sha256(json.dumps(
            row.to_dict(), sort_keys=True, ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8')).hexdigest(), axis=1,
    )
    duplicate_base = base.loc[duplicate] + '::sha256=' + fingerprints
    copies = duplicate_base.groupby(duplicate_base, sort=False).cumcount() + 1
    base.loc[duplicate] = duplicate_base + '::copy=' + copies.astype(str)
    return base


def _load_raw(root: Path, type_mapping: Mapping[str, str] | str | os.PathLike | None
              ) -> pd.DataFrame:
    mapping, mapping_source = _mapping(type_mapping)
    frames, sources = [], []
    for track, task, filename in _RAW_FILES:
        path = root / f'Track-{track}' / filename
        if not path.is_file():
            raise ValueError(f'Missing raw dataset file: {path}')
        raw = _csv_read(path)
        source_column = 'question_id' if track == 'A' and task == 'math' else 'id'
        context_column, question_column, answer_column = 'context', 'question', 'answer'
        if task in ('bn_qa', 'codemixed_qa'):
            context_column = 'codemix_context' if task == 'codemixed_qa' else 'context'
            question_column = 'codemix_question' if task == 'codemixed_qa' else 'question'
            answer_column = ('codemix_answer' if task == 'codemixed_qa' else 'correct_answer') if track == 'A' else 'hallucinated_answer'
            required = (source_column, context_column, question_column, answer_column)
        elif task == 'summarization':
            context_column = 'question' if track == 'A' else 'document'
            answer_column = 'summary' if track == 'A' else 'hallucinated_summary'
            required = (source_column, context_column, answer_column)
        else:
            required = (source_column, 'question', 'answer') if track == 'A' else (
                source_column, 'question', 'hallucinated_chain', 'hallucinated_answer',
            )
        type_column = 'hallucination_type' if task == 'math' else 'pattern'
        if track == 'B':
            required += (type_column,)
        try:
            _require_columns(raw, required)
        except ValueError as exc:
            raise ValueError(f'{path}: {exc}') from exc
        canonical = pd.DataFrame(index=raw.index)
        canonical['id'] = _raw_ids(raw, task, f'Track_{track}', source_column)
        canonical['task'], canonical['track'] = task, f'Track_{track}'
        canonical['ground_truth_label'] = int(track == 'B')
        if task in ('bn_qa', 'codemixed_qa'):
            canonical['context'] = raw[context_column]
            canonical['question'] = raw[question_column]
            canonical['candidate_answer'] = raw[answer_column]
        elif task == 'summarization':
            canonical['context'] = _text(_series(raw, context_column), f'{path}: document')
            canonical['question'] = _SUMMARY_QUESTION
            canonical['candidate_answer'] = raw[answer_column]
        else:
            canonical['context'] = ''
            canonical['question'] = raw['question']
            if track == 'A':
                canonical['candidate_answer'] = raw['answer']
            else:
                chain = _text(_series(raw, 'hallucinated_chain'), f'{path}: hallucinated_chain')
                answer = _text(_series(raw, 'hallucinated_answer'), f'{path}: hallucinated_answer')
                canonical['candidate_answer'] = chain + '\n\nচূড়ান্ত উত্তর: ' + answer
        if track == 'A':
            canonical['hallucination_type'] = 'None'
        else:
            labels = _text(_series(raw, type_column), f'{path}: {type_column}')
            translated = labels.map(lambda value: mapping.get(value, _TYPE_LOOKUP.get(value.casefold())))
            unknown = sorted({value for value in labels if value not in mapping and value.casefold() not in _TYPE_LOOKUP})
            if unknown:
                raise ValueError(
                    f'{path}: unknown raw hallucination labels {unknown}. Supply '
                    '--type_mapping_json PATH containing an explicit flat JSON object '
                    'mapping each raw label to a canonical type. No semantic equivalence '
                    f'is assumed. Canonical types: {HALLUCINATION_TYPES}.'
                )
            canonical['hallucination_type'] = translated
        sources.append({
            'path': str(path.resolve()), 'sha256': _sha256(path), 'rows': len(raw),
            'duplicate_source_id_rows': int(raw[source_column].duplicated(keep=False).sum()),
            'exact_duplicate_rows_beyond_first': int(raw.duplicated().sum()),
        })
        frames.append(canonical)
    result = validate_benhallu(pd.concat(frames, ignore_index=True), require_full=True)
    result.attrs['mock'] = False
    result.attrs['provenance'] = {
        'kind': 'raw_directory', 'sources': sources, 'type_mapping': mapping,
        'type_mapping_source': mapping_source,
        'id_policy': 'task_track_source-id; duplicate IDs add SHA256(raw record) and copy ordinal; all rows retained',
        'summarization_policy': 'document in context; fixed summarization instruction in question',
        'math_candidate_policy': 'hallucinated_chain + final-answer delimiter + hallucinated_answer; never original answer',
    }
    return result


def load_dataset(path: str | os.PathLike, mode: str,
                 type_mapping: Mapping[str, str] | str | os.PathLike | None = None
                 ) -> pd.DataFrame:
    """Load a canonical CSV or, for BenHallu, the supplied raw directory.

    ``type_mapping`` may be a flat dict or a path to its JSON representation.
    Full-size requirements are enforced by the split-making functions, allowing
    this loader and the validators to inspect smaller canonical datasets too.
    """
    if mode not in ('benhallu', 'dialect'):
        raise ValueError("mode must be 'benhallu' or 'dialect'.")
    path = Path(path)
    if path.is_dir():
        if mode != 'benhallu':
            raise ValueError('Raw directory ingestion is supported only in benhallu mode.')
        return _load_raw(path, type_mapping)
    if type_mapping is not None:
        raise ValueError('type_mapping is only applicable to raw BenHallu directory ingestion.')
    if not path.is_file():
        raise ValueError(f'Input does not exist or is not a regular file/directory: {path}')
    frame = _csv_read(path)
    frame = validate_benhallu(frame) if mode == 'benhallu' else validate_dialect(frame)
    frame.attrs['mock'] = False
    frame.attrs['provenance'] = {
        'kind': 'canonical_csv',
        'sources': [{'path': str(path.resolve()), 'sha256': _sha256(path), 'rows': len(frame)}],
    }
    return frame


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values('id', kind='mergesort').reset_index(drop=True)


def _composite(frame: pd.DataFrame) -> pd.Series:
    return frame['task'] + '_' + frame['track'] + '_' + frame['hallucination_type']


def _sampling_problem(frame: pd.DataFrame, n: int, strata: pd.Series) -> str | None:
    counts = strata.value_counts()
    if n <= 0 or n >= len(frame):
        return f'need 0 < selected size {n} < available rows {len(frame)}'
    if counts.min() < 2:
        return f'each stratum needs at least 2 rows; counts={counts.to_dict()}'
    if min(n, len(frame) - n) < len(counts):
        return f'both selected and remaining sets need at least {len(counts)} rows (one per stratum)'
    return None


def _sample(frame: pd.DataFrame, n: int, seed: int, *, composite: bool,
            description: str) -> pd.DataFrame:
    ordered = _sort(frame)
    strata = _composite(ordered) if composite else _series(ordered, 'question_type')
    problem = _sampling_problem(ordered, n, strata)
    if problem:
        raise ValueError(f'Cannot stratify {description}: {problem}. Do not merge or discard rare strata silently.')
    _, selected = train_test_split(
        ordered, test_size=n, random_state=seed, stratify=strata,
    )
    selected = cast(pd.DataFrame, selected)
    selected_strata = _composite(selected) if composite else _series(selected, 'question_type')
    source_keys, selected_keys = set(strata), set(selected_strata)
    if source_keys != selected_keys:
        counts = strata.value_counts()
        missing = {
            key: {'source_rows': int(counts[key]),
                  'proportional_quota': n * int(counts[key]) / len(ordered)}
            for key in sorted(source_keys - selected_keys)
        }
        raise ValueError(
            f'Cannot stratify {description}: selecting {n} of {len(ordered)} rows '
            f'loses source-stratum coverage; missing={missing}, '
            f'unexpected={sorted(selected_keys - source_keys)}. '
            'Insufficient rare-stratum proportions can receive zero rows under '
            "sklearn's integer proportional allocation. Increase the subset size "
            'or collect more examples for rare strata. No quotas are adjusted '
            'and no incomplete split will be published.'
        )
    return _sort(selected)


def _atomic_write(path: Path, *, frame: pd.DataFrame | None = None,
                  payload: dict[str, Any] | None = None) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if frame is not None:
                frame.to_csv(stream, index=False, lineterminator='\n')
            else:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(series.value_counts().items())}


def _frame_counts(frame: pd.DataFrame, mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {'rows': len(frame)}
    columns = ('track', 'task', 'ground_truth_label', 'hallucination_type') if mode == 'benhallu' else ('dialect', 'question_type')
    for column in columns:
        result[column] = _counts(_series(frame, column))
    if mode == 'benhallu':
        result['composite'] = _counts(_composite(frame))
    else:
        result['dialect_question_type'] = {
            dialect: _counts(_series(group, 'question_type'))
            for dialect, group in frame.groupby('dialect', sort=True)
        }
    return result


def _publish(frame: pd.DataFrame, output_dir: str | os.PathLike, seed: int,
             mode: str, outputs: dict[str, tuple[str, pd.DataFrame]],
             selection: dict[str, Any]) -> None:
    directory = Path(output_dir)
    provenance = frame.attrs.get('provenance', {'kind': 'in_memory', 'sources': []})
    artifacts = dict(outputs)
    mock_filename = frame.attrs.get('mock_filename')
    if mock_filename:
        artifacts['mock_dataset'] = (mock_filename, frame)
    targets = [directory / name for name, _ in artifacts.values()] + [directory / 'manifest.json']
    sources = {Path(item['path']).resolve() for item in provenance.get('sources', [])}
    mapping_source = provenance.get('type_mapping_source')
    if mapping_source:
        sources.add(Path(mapping_source['path']).resolve())
    for target in targets:
        if target.resolve() in sources or any(
                target.exists() and source.exists() and os.path.samefile(target, source)
                for source in sources):
            raise ValueError(f'Refusing to overwrite an input/provenance file: {target}')
    # Compute every selection and check all destinations before creating output.
    directory.mkdir(parents=True, exist_ok=True)
    output_metadata = {}
    for key, (filename, subset) in artifacts.items():
        target = directory / filename
        _atomic_write(target, frame=subset)
        output_metadata[key] = {
            'filename': filename, 'sha256': _sha256(target),
            **_frame_counts(subset, mode),
        }
    ids = {key: set(subset['id']) for key, (_, subset) in outputs.items()}
    overlaps = {
        f'{left}__{right}': len(ids[left] & ids[right])
        for index, left in enumerate(ids) for right in list(ids)[index + 1:]
    }
    canonical_bytes = frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
    manifest = {
        'schema_version': 2, 'mode': mode, 'seed': seed,
        'mock': bool(frame.attrs.get('mock', False)), 'evaluation_only': True,
        'overlap_policy': 'Intentional evaluation subset overlap; NOT train/test partitions. Row-level, not source-grouped.',
        'dependencies': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
        },
        'input': {
            **_frame_counts(frame, mode),
            'canonical_csv_sha256': hashlib.sha256(canonical_bytes).hexdigest(),
            'provenance': provenance,
        },
        'selection': selection, 'outputs': output_metadata,
        'pairwise_id_overlap': overlaps,
    }
    _atomic_write(directory / 'manifest.json', payload=manifest)


def make_benhallu_splits(frame: pd.DataFrame, output_dir: str | os.PathLike,
                         seed: int = 42) -> dict[str, pd.DataFrame]:
    """Write full, 10% and micro evaluation sets; return keys micro/10pct/full.

    Each track is sampled separately with task_track_hallucination_type strata.
    Both micro and 10% are sampled directly from full to preserve its proportions;
    overlap is allowed, but neither nesting nor disjointness is enforced.
    """
    seed = _seed(seed)
    full = _sort(validate_benhallu(frame, require_full=True))
    ten_parts, micro_parts = [], []
    for track, ten_size, micro_size in (('Track_A', 400, 30), ('Track_B', 1200, 90)):
        source = full.loc[full['track'].eq(track)]
        ten_parts.append(_sample(source, ten_size, seed, composite=True, description=f'{track} 10%'))
        micro_parts.append(_sample(source, micro_size, seed, composite=True, description=f'{track} micro'))
    ten = _sort(pd.concat(ten_parts, ignore_index=True))
    micro = _sort(pd.concat(micro_parts, ignore_index=True))
    outputs = {
        'micro': ('benhallu_micro_debug_120.csv', micro),
        '10pct': ('benhallu_stratified_10pct.csv', ten),
        'full': ('benhallu_full_16k.csv', full),
    }
    _publish(full, output_dir, seed, 'benhallu', outputs, {
        'algorithm': 'sklearn.model_selection.train_test_split',
        'stratify': "f'{task}_{track}_{hallucination_type}'", 'separate_tracks': True,
        'input_order': 'canonical id, ascending',
        'rounding': 'sklearn stratified integer allocation; both subsets reference full proportions',
        'sampling_pool': 'full dataset, separately by track, for both micro and 10pct',
        'track_targets': {
            'micro': {'Track_A': 30, 'Track_B': 90},
            '10pct': {'Track_A': 400, 'Track_B': 1200},
        },
    })
    return {key: subset for key, (_, subset) in outputs.items()}


def make_dialect_subset(frame: pd.DataFrame, output_dir: str | os.PathLike,
                         seed: int = 42) -> pd.DataFrame:
    """Write and return 300 rows: 100 per anchor, stratified by question_type."""
    seed = _seed(seed)
    full = _sort(validate_dialect(frame, require_full=True))
    pieces = []
    for dialect in ANCHOR_DIALECTS:
        source = full.loc[full['dialect'].eq(dialect)]
        if source['question_type'].nunique() != 2:
            raise ValueError(f'Anchor dialect {dialect} must contain both question types.')
        pieces.append(_sample(source, 100, seed, composite=False, description=f'{dialect} anchor'))
    subset = _sort(pd.concat(pieces, ignore_index=True))
    _publish(full, output_dir, seed, 'dialect', {
        'subset': ('dialect_ccc_subset_300.csv', subset),
    }, {
        'algorithm': 'sklearn.model_selection.train_test_split', 'stratify': 'question_type',
        'input_order': 'canonical id, ascending', 'rounding': 'sklearn stratified integer allocation',
        'anchor_targets': dict.fromkeys(ANCHOR_DIALECTS, 100),
        'excluded_dialects': sorted(set(full['dialect']) - set(ANCHOR_DIALECTS)),
    })
    return subset


def _mock_example(task: str, kind: str, rng: np.random.Generator
                  ) -> tuple[str, str, str]:
    """Small Bengali/code-mixed templates with type-specific synthetic errors."""
    if task in ('bn_qa', 'codemixed_qa'):
        topics = (
            ('বাংলাদেশের রাজধানী ঢাকা।', 'বাংলাদেশের রাজধানী কী?', 'ঢাকা', 'কলকাতা'),
            ('কাজী নজরুল ইসলাম বাংলাদেশের জাতীয় কবি।', 'বাংলাদেশের জাতীয় কবি কে?', 'কাজী নজরুল ইসলাম', 'জীবনানন্দ দাশ'),
            ('পদ্মা সেতু ২০২২ সালে উদ্বোধন করা হয়।', 'পদ্মা সেতু কত সালে উদ্বোধন করা হয়?', '২০২২ সালে', '২০১২ সালে'),
            ('সূর্য একটি নক্ষত্র। পৃথিবী সূর্যের চারদিকে ঘোরে।', 'পৃথিবী কিসের চারদিকে ঘোরে?', 'সূর্যের চারদিকে', 'চাঁদের চারদিকে'),
        )
        context, question, correct, wrong = topics[int(rng.integers(len(topics)))]
        if task == 'codemixed_qa':
            context = 'Read this তথ্য: ' + context
            question = 'Context অনুযায়ী answer দিন: ' + question
            correct = 'The answer is ' + correct
            wrong = 'The answer is ' + wrong
        errors = {
            'Factualness': wrong,
            'Comprehension': 'প্রশ্নটি আসলে নদীর দৈর্ঘ্য সম্পর্কে; উত্তর ৫০০ কিলোমিটার।',
            'Specificity': correct + ', নির্দিষ্টভাবে সকাল ৭টা ১৩ মিনিটে প্রতিষ্ঠিত।',
            'Inference': correct + '; তাই নিশ্চিতভাবে দেশের সবাই একই ভাষায় কথা বলে।',
        }
        return context, question, correct if kind == 'None' else errors[kind]
    if task == 'summarization':
        city = str(rng.choice(['ঢাকা', 'খুলনা', 'রাজশাহী', 'বরিশাল']))
        count = int(rng.integers(20, 200))
        context = (
            f'{city} শহরে একটি পাঠাগার চালু হয়েছে। প্রথম দিনে {count} জন পাঠক আসেন। '
            'পাঠাগারটি শিক্ষার্থীদের বিনা মূল্যে বই পড়ার সুযোগ দেয়।'
        )
        correct = f'{city} শহরের নতুন পাঠাগারে প্রথম দিনে {count} জন পাঠক আসেন; শিক্ষার্থীরা বিনা মূল্যে বই পড়তে পারেন।'
        errors = {
            'Fabricated_Content': f'{city} শহরের নতুন পাঠাগারটি আন্তর্জাতিক মহাকাশ গবেষণা কেন্দ্রও চালু করেছে।',
            'Non_Factual_Addition': correct + ' এটি বিশ্বের সবচেয়ে প্রাচীন পাঠাগার।',
            'Direct_Contradiction': f'{city} শহরের পাঠাগারে কোনো পাঠক আসেননি এবং শিক্ষার্থীদের বই পড়তে দেওয়া হয় না।',
        }
        return context, _SUMMARY_QUESTION, correct if kind == 'None' else errors[kind]
    a, b = (int(value) for value in rng.integers(5, 90, size=2))
    question = f'রহিমের কাছে {a}টি আম ছিল। সে আরও {b}টি আম পেল। এখন মোট কতটি আম আছে? ধাপে ধাপে দেখাও।'
    correct = f'মোট আম = {a} + {b} = {a + b}। উত্তর: {a + b}টি।'
    errors = {
        'Arithmetic_Slip': f'{a} + {b} = {a + b + 3}। উত্তর: {a + b + 3}টি।',
        'Formula_Misapplication': f'মোট নির্ণয়ের সূত্র গুণ: {a} × {b} = {a * b}। উত্তর: {a * b}টি।',
        'Invalid_Deduction': f'আরও আম পাওয়ায় সংখ্যা কমবে: {a} - {b} = {a - b}। উত্তর: {a - b}টি।',
        'Hallucinated_Intermediate_Fact': f'রহিম গোপনে আরও ১০টি আম পেয়েছে। মোট {a} + {b} + ১০ = {a + b + 10}। উত্তর: {a + b + 10}টি।',
        'Semantic_Drift': 'আমের পরিবর্তে ঝুড়ির সংখ্যা হিসাব করি। দুটি ঝুড়ি আছে। চূড়ান্ত উত্তর: ২টি ঝুড়ি।',
    }
    return '', question, correct if kind == 'None' else errors[kind]


def generate_mock_benhallu(seed: int = 42) -> pd.DataFrame:
    """Generate exactly the canonical schema and realistic raw task/type counts."""
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    records = []
    b_counts = {'bn_qa': 4000, 'codemixed_qa': 4000, 'summarization': 3000, 'math': 1000}
    for track in ('Track_A', 'Track_B'):
        for task in TASKS:
            size = 1000 if track == 'Track_A' else b_counts[task]
            kinds = ('None',) if track == 'Track_A' else TASK_TYPES[task]
            for i in range(size):
                kind = kinds[i % len(kinds)]
                context, question, answer = _mock_example(task, kind, rng)
                records.append({
                    'id': f'{task}_{track}_{i:05d}', 'task': task, 'track': track,
                    'hallucination_type': kind, 'question': question, 'context': context,
                    'candidate_answer': answer, 'ground_truth_label': int(track == 'Track_B'),
                })
    frame = pd.DataFrame.from_records(records, columns=BENHALLU_COLUMNS)
    frame = validate_benhallu(frame.iloc[rng.permutation(len(frame))], require_full=True)
    frame.attrs['mock'] = True
    frame.attrs['provenance'] = {'kind': 'synthetic', 'generator': 'generate_mock_benhallu', 'seed': seed, 'sources': []}
    return frame


def generate_mock_dialect(seed: int = 42) -> pd.DataFrame:
    """Generate 4,000 rows: 1,000 per dialect, two question types, scores 0..10.

    Scores illustrate correctness/completeness: 0 is wholly incorrect, 10 fully
    correct, and intermediate responses use partial answers or hedging. These
    are synthetic examples, not calibrated human judgments or linguistically
    validated dialect resources.
    """
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    factual_questions = {
        'Standard_Bengali': 'বাংলাদেশের রাজধানীর নাম কী?',
        'Tangail': 'বাংলাদেশের রাজধানীর নাম কী কও তো?',
        'Chittagong': 'বাংলাদেশর রাজধানীর নাম কিতা?',
        'Sylheti': 'বাংলাদেশর রাজধানীর নাম কিতা অয়?',
    }
    definitional_questions = {
        'Standard_Bengali': 'পাঠাগার কাকে বলে?',
        'Tangail': 'পাঠাগার কইতে কী বুঝায় কও তো?',
        'Chittagong': 'পাঠাগার কইলে কিতা বুঝায়?',
        'Sylheti': 'পাঠাগার কইতে কিতা বুঝায়?',
    }
    factual_responses = (
        'বাংলাদেশের রাজধানী কলকাতা।',
        'বাংলাদেশের রাজধানী সম্ভবত কলকাতা, তবে আমি নিশ্চিত নই।',
        'নামটি মনে নেই; শহরটি বাংলাদেশের ভেতরে অবস্থিত।',
        'রাজধানীর নাম নিশ্চিত নই; সম্ভবত দেশের মধ্যাঞ্চলের কোনো শহর।',
        'ঢাকা অথবা চট্টগ্রাম হতে পারে; কোনটি রাজধানী তা নিশ্চিত নই।',
        'ঢাকা হতে পারে, কিন্তু আমি মূলত অনুমান করছি।',
        'আমার মনে হয় রাজধানী ঢাকা, তবে নিশ্চিত হতে যাচাই করা দরকার।',
        'সম্ভবত বাংলাদেশের রাজধানী ঢাকা।',
        'আমার জানা অনুযায়ী বাংলাদেশের রাজধানী ঢাকা।',
        'ঢাকাই বাংলাদেশের রাজধানী বলে আমি প্রায় নিশ্চিত।',
        'বাংলাদেশের রাজধানী ঢাকা।',
    )
    definitional_responses = (
        'পাঠাগার হলো শুধু গাড়ি মেরামতের কারখানা।',
        'পাঠাগার হলো বই বিক্রির দোকান।',
        'পাঠাগার বইয়ের সঙ্গে সম্পর্কিত কোনো জায়গা; এর কাজটি বলতে পারছি না।',
        'পাঠাগার হলো বই রাখা হয় এমন একটি ঘর।',
        'পাঠাগার হলো বই জমা রাখা ও পড়ার জায়গা।',
        'পাঠাগারে বিভিন্ন বই রাখা থাকে এবং মানুষ সেখানে পড়তে পারেন।',
        'পাঠাগার হলো সুশৃঙ্খলভাবে বই সংগ্রহ ও পড়ার সুযোগ দেওয়ার স্থান।',
        'পাঠাগার হলো বই ও অন্যান্য পাঠসামগ্রী সংগ্রহ, সংরক্ষণ ও পাঠের স্থান।',
        'পাঠাগারে বই ও অন্যান্য পাঠসামগ্রী সুশৃঙ্খলভাবে রাখা হয় এবং পাঠকেরা '
        'সেগুলো পড়তে পারেন; অনেক ক্ষেত্রে ধারও নিতে পারেন।',
        'পাঠাগার হলো পাঠকের ব্যবহারের জন্য বই ও অন্যান্য তথ্যসামগ্রী সংগ্রহ, '
        'সংগঠিত ও সংরক্ষণ করার প্রতিষ্ঠান, যেখানে পড়া এবং নিয়ম অনুযায়ী '
        'ধার নেওয়ার সুযোগ থাকতে পারে।',
        'পাঠাগার হলো এমন একটি প্রতিষ্ঠান বা সংগ্রহ, যেখানে বই ও অন্যান্য '
        'তথ্যসামগ্রী সুশৃঙ্খলভাবে সংগ্রহ, সংগঠিত ও সংরক্ষণ করা হয় এবং পাঠকদের '
        'পড়া, তথ্য অনুসন্ধান ও প্রযোজ্য নিয়মে ধার নেওয়ার সুবিধা দেওয়া হয়।',
    )
    records = []
    for dialect in DIALECTS:
        for i in range(1000):
            question_type = 'factual' if i % 2 == 0 else 'definitional'
            score = int(rng.integers(0, 11))
            if question_type == 'factual':
                question = factual_questions[dialect]
                answer = factual_responses[score]
            else:
                question = definitional_questions[dialect]
                answer = definitional_responses[score]
            records.append({
                'id': f'{dialect}_{i:04d}', 'dialect': dialect,
                'question_type': question_type, 'question': question,
                'response': answer, 'human_or_primary_score': score,
            })
    frame = pd.DataFrame.from_records(records, columns=DIALECT_COLUMNS)
    frame = validate_dialect(frame.iloc[rng.permutation(len(frame))], require_full=True)
    frame.attrs['mock'] = True
    frame.attrs['provenance'] = {'kind': 'synthetic', 'generator': 'generate_mock_dialect', 'seed': seed, 'sources': []}
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Create reproducible stratified evaluation subsets (NOT train/test partitions).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input_csv', type=Path, help='Canonical CSV, or the raw All data directory for benhallu.')
    source.add_argument('--generate_mock_data', action='store_true', help='Generate synthetic data and the requested subsets.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Destination directory; input files are never overwritten.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed in [0, 2**32 - 1].')
    parser.add_argument('--mode', choices=('benhallu', 'dialect'), default='benhallu', help='Dataset mode.')
    parser.add_argument('--type_mapping_json', type=Path, help='Explicit raw-label -> canonical-type JSON object; raw BenHallu only.')
    args = parser.parse_args(argv)
    try:
        seed = _seed(args.seed)
        if args.generate_mock_data:
            if args.type_mapping_json is not None:
                raise ValueError('--type_mapping_json is only applicable to raw directory ingestion, not mock data.')
            frame = generate_mock_benhallu(seed) if args.mode == 'benhallu' else generate_mock_dialect(seed)
            frame.attrs['mock_filename'] = f'mock_{args.mode}_{16000 if args.mode == "benhallu" else 4000}.csv'
        else:
            frame = load_dataset(args.input_csv, args.mode, args.type_mapping_json)
        if args.mode == 'benhallu':
            make_benhallu_splits(frame, args.output_dir, seed)
        else:
            make_dialect_subset(frame, args.output_dir, seed)
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f'Wrote {args.mode} evaluation subsets and manifest.json to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

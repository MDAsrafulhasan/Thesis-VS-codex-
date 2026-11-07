"""Contract tests for the evaluation pipeline; all artifacts live under tmp_path.

Run everything with ``python test_pipeline.py`` or ``python -m pytest -q``.
While eval.py is being implemented, run ``python test_pipeline.py -k splits``.
Mock fixtures exercise plumbing only; they are not scientific evaluation data.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from email.message import Message
from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import make_splits


ROOT = Path(__file__).resolve().parent
BENHALLU_COLUMNS = [
    'id', 'task', 'track', 'hallucination_type', 'question', 'context',
    'candidate_answer', 'ground_truth_label',
]
DIALECT_COLUMNS = [
    'id', 'dialect', 'question_type', 'question', 'response',
    'human_or_primary_score',
]
TASKS = {'bn_qa', 'codemixed_qa', 'summarization', 'math'}
TYPES = {
    'Factualness', 'Comprehension', 'Specificity', 'Inference',
    'Fabricated_Content', 'Non_Factual_Addition', 'Direct_Contradiction',
    'Arithmetic_Slip', 'Formula_Misapplication', 'Invalid_Deduction',
    'Hallucinated_Intermediate_Fact', 'Semantic_Drift',
}
ANCHORS = {'Standard_Bengali', 'Tangail', 'Chittagong'}
QUESTION_TYPES = {'definitional', 'factual'}
SPLIT_FILES = {
    'micro': 'benhallu_micro_debug_120.csv',
    '10pct': 'benhallu_stratified_10pct.csv',
    'full': 'benhallu_full_16k.csv',
}
REPORT_KEYS = {
    'table1_overall_and_per_task_bhs',
    'table2_granular_hallucination_types',
    'table3_cot_delta_ablation',
}


@pytest.fixture(scope='session')
def benhallu_full():
    return make_splits.generate_mock_benhallu(seed=42)


@pytest.fixture(scope='session')
def dialect_full():
    return make_splits.generate_mock_dialect(seed=42)


@pytest.fixture
def benhallu_small(benhallu_full):
    # Preserve both tracks for every task, and every task/type stratum.
    return benhallu_full.drop_duplicates(
        ['task', 'track', 'hallucination_type'],
    ).reset_index(drop=True).copy(deep=True)


@pytest.fixture
def dialect_small(dialect_full):
    return dialect_full.groupby(
        ['dialect', 'question_type'], sort=True, group_keys=False,
    ).head(2).reset_index(drop=True).copy(deep=True)


@pytest.fixture
def report_benhallu():
    records = []
    for track, label in [('Track_A', 0), ('Track_B', 1)]:
        for index in range(4):
            records.append({
                'id': f'report_{track}_{index}', 'task': 'bn_qa', 'track': track,
                'hallucination_type': 'None' if label == 0 else 'Factualness',
                'question': f'REPORT_QUESTION_{track}_{index}_END',
                'context': 'বাংলাদেশের রাজধানী ঢাকা।', 'candidate_answer': 'ঢাকা।',
                'ground_truth_label': label,
            })
    return pd.DataFrame.from_records(records, columns=BENHALLU_COLUMNS)


@pytest.fixture
def evaluation():
    # Do not import eval at collection time: split-only work must be runnable.
    return importlib.import_module('eval')


@pytest.fixture
def offline_chat_transport(evaluation, monkeypatch):
    transport, sleep = Mock(name='urlopen'), Mock(name='sleep')
    monkeypatch.setattr(urllib.request, 'urlopen', transport)
    monkeypatch.setattr(evaluation.time, 'sleep', sleep)
    monkeypatch.delenv('PYTEST_PIPELINE_API_KEY', raising=False)
    payload = {
        'id': 'offline-completion-123', 'model': 'provider-model-revision',
        'system_fingerprint': 'offline-fingerprint', 'created': 1700000000,
        'choices': [{
            'message': {'content': '<verdict>হ্যাঁ</verdict>'}, 'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': 31, 'completion_tokens': 9, 'total_tokens': 40},
    }
    return transport, sleep, payload


def _transport_http_error(status, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return urllib.error.HTTPError(
        'https://example.invalid/v1/chat/completions', status, 'Synthetic HTTP failure',
        headers, BytesIO(b'offline provider error detail'),
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _write_jsonl(path, rows):
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows),
        encoding='utf-8',
    )


def _prediction_judge(evaluation, frame, predictions):
    answers = {
        row['question']: predictions[row['id']]
        for row in frame.to_dict(orient='records')
    }

    def judge(prompt):
        matches = [value for question, value in answers.items() if question in prompt]
        assert len(matches) == 1, 'Each test question must uniquely identify a scripted response'
        prediction = matches[0]
        text = 'Unparseable response.' if prediction is None else (
            '<verdict>yes</verdict>' if prediction == 1 else '<verdict>no</verdict>'
        )
        return evaluation.InferenceResponse(text, finish_reason='stop')

    return judge


def _checkpoint(directory):
    paths = list(Path(directory).rglob('*.jsonl'))
    assert len(paths) == 1, f'Expected one checkpoint, found {paths}'
    return paths[0]


def _run_cli(script, arguments, cwd):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, arguments)],
        cwd=cwd, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return result


class RecordingJudge:
    def __init__(self, evaluation, text='<verdict>yes</verdict>'):
        self.evaluation = evaluation
        self.text = text
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.evaluation.InferenceResponse(
            text=self.text, prompt_tokens=13, completion_tokens=7,
            finish_reason='stop',
        )


def _evaluate(evaluation, frame, judge, output_dir, **overrides):
    options = {
        'model_name': 'pytest_model', 'prompt_mode': 'zero_shot',
        'split_name': 'micro', 'mode': 'benhallu', 'output_dir': output_dir,
        'seed': 42, 'backend_identity': 'pytest-recording-v1',
    }
    options.update(overrides)
    return evaluation.run_evaluation(frame, judge, **options)


# Split generation and input validation ---------------------------------------


def test_splits_mock_benhallu_schema_counts_and_determinism(benhallu_full):
    frame = benhallu_full
    assert list(frame.columns) == BENHALLU_COLUMNS
    assert len(frame) == 16000
    assert frame['id'].is_unique
    assert frame['track'].value_counts().to_dict() == {
        'Track_A': 4000, 'Track_B': 12000,
    }
    for track, label in [('Track_A', 0), ('Track_B', 1)]:
        rows = frame.loc[frame['track'].eq(track)]
        assert set(rows['task']) == TASKS
        assert rows['ground_truth_label'].eq(label).all()
    assert frame.loc[frame['track'].eq('Track_A'), 'hallucination_type'].eq('None').all()
    assert set(frame.loc[frame['track'].eq('Track_B'), 'hallucination_type']) == TYPES
    assert_frame_equal(frame, make_splits.generate_mock_benhallu(seed=42))
    assert_frame_equal(frame, make_splits.validate_benhallu(frame, require_full=True))


def test_splits_mock_dialect_schema_counts_and_determinism(dialect_full):
    frame = dialect_full
    assert list(frame.columns) == DIALECT_COLUMNS
    assert len(frame) == 4000
    assert frame['id'].is_unique
    assert ANCHORS <= set(frame['dialect'])
    assert set(frame['question_type']) == QUESTION_TYPES
    assert np.isfinite(frame['human_or_primary_score']).all()
    assert frame['human_or_primary_score'].between(0, 10).all()
    assert_frame_equal(frame, make_splits.generate_mock_dialect(seed=42))
    assert_frame_equal(frame, make_splits.validate_dialect(frame, require_full=True))


def test_splits_benhallu_artifacts_strata_and_round_trip(benhallu_full, tmp_path):
    before = benhallu_full.copy(deep=True)
    outputs = make_splits.make_benhallu_splits(benhallu_full, tmp_path, seed=42)
    assert set(outputs) == {'micro', '10pct', 'full'}
    source_strata = set(map(tuple, benhallu_full[
        ['task', 'track', 'hallucination_type']
    ].drop_duplicates().to_numpy()))
    expected_counts = {
        'micro': {'Track_A': 30, 'Track_B': 90},
        '10pct': {'Track_A': 400, 'Track_B': 1200},
        'full': {'Track_A': 4000, 'Track_B': 12000},
    }
    for name, subset in outputs.items():
        assert list(subset.columns) == BENHALLU_COLUMNS
        assert subset['track'].value_counts().to_dict() == expected_counts[name]
        assert subset['id'].is_unique
        assert subset['id'].tolist() == sorted(subset['id'])
        assert set(subset['id']) <= set(benhallu_full['id'])
        assert set(map(tuple, subset[
            ['task', 'track', 'hallucination_type']
        ].drop_duplicates().to_numpy())) == source_strata
        stored = pd.read_csv(tmp_path / SPLIT_FILES[name], keep_default_na=False)
        assert_frame_equal(subset, stored)
        assert stored.loc[stored['track'].eq('Track_A'), 'hallucination_type'].eq('None').all()
        for track, size in expected_counts[name].items():
            full_track = benhallu_full.loc[benhallu_full['track'].eq(track)]
            selected = subset.loc[subset['track'].eq(track)]
            keys = ['task', 'hallucination_type']
            expected = full_track.groupby(keys).size() * size / len(full_track)
            actual = selected.groupby(keys).size().reindex(expected.index, fill_value=0)
            assert (actual.sub(expected).abs() <= 1 + 1e-12).all()
    assert_frame_equal(benhallu_full, before)
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['seed'] == 42
    assert manifest['mode'] == 'benhallu'
    for key, filename in SPLIT_FILES.items():
        assert manifest['outputs'][key]['filename'] == filename
        assert manifest['outputs'][key]['sha256'] == hashlib.sha256(
            (tmp_path / filename).read_bytes(),
        ).hexdigest()


def test_splits_benhallu_reproducible_under_row_reordering(benhallu_full, tmp_path):
    original = make_splits.make_benhallu_splits(benhallu_full, tmp_path / 'first', seed=42)
    reordered = make_splits.make_benhallu_splits(
        benhallu_full.sample(frac=1, random_state=11), tmp_path / 'reordered', seed=42,
    )
    changed_seed = make_splits.make_benhallu_splits(benhallu_full, tmp_path / 'other', seed=43)
    for key in original:
        assert_frame_equal(original[key], reordered[key])
    assert set(original['micro']['id']) != set(changed_seed['micro']['id'])
    assert_frame_equal(original['full'], changed_seed['full'])


def test_splits_dialect_anchor_and_question_type_stratification(dialect_full, tmp_path):
    before = dialect_full.copy(deep=True)
    subset = make_splits.make_dialect_subset(dialect_full, tmp_path / 'first', seed=42)
    assert list(subset.columns) == DIALECT_COLUMNS
    assert len(subset) == 300
    assert subset['id'].is_unique
    assert subset['dialect'].value_counts().to_dict() == dict.fromkeys(ANCHORS, 100)
    assert set(subset['id']) <= set(dialect_full['id'])
    assert subset['id'].tolist() == sorted(subset['id'])
    for dialect, group in subset.groupby('dialect'):
        assert set(group['question_type']) == QUESTION_TYPES
        source = dialect_full.loc[dialect_full['dialect'].eq(dialect)]
        expected = source['question_type'].value_counts(normalize=True) * 100
        actual = group['question_type'].value_counts().reindex(expected.index, fill_value=0)
        assert (actual.sub(expected).abs() <= 1 + 1e-12).all()
    stored = pd.read_csv(tmp_path / 'first' / 'dialect_ccc_subset_300.csv', keep_default_na=False)
    assert_frame_equal(subset, stored)
    repeated = make_splits.make_dialect_subset(
        dialect_full.sample(frac=1, random_state=11), tmp_path / 'reordered', seed=42,
    )
    assert_frame_equal(subset, repeated)
    assert_frame_equal(dialect_full, before)


@pytest.mark.parametrize('column,value', [
    ('id', ''), ('task', 'unknown'), ('track', 'A'),
    ('ground_truth_label', 2), ('ground_truth_label', float('nan')),
    ('ground_truth_label', float('inf')), ('question', ''),
    ('candidate_answer', None), ('hallucination_type', 'Not_A_Type'),
])
def test_splits_benhallu_rejects_invalid_values(benhallu_small, column, value):
    frame = benhallu_small.copy(deep=True).astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        make_splits.validate_benhallu(frame)


@pytest.mark.parametrize('column,value', [
    ('id', ''), ('dialect', ''), ('question_type', 'opinion'),
    ('question', None), ('response', ''),
    ('human_or_primary_score', -0.1), ('human_or_primary_score', 10.1),
    ('human_or_primary_score', float('nan')),
    ('human_or_primary_score', float('inf')),
])
def test_splits_dialect_rejects_invalid_values(dialect_small, column, value):
    frame = dialect_small.copy(deep=True).astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        make_splits.validate_dialect(frame)


@pytest.mark.parametrize('mode', ['benhallu', 'dialect'])
def test_splits_rejects_missing_columns_duplicate_ids_and_incomplete_full(
    mode, benhallu_small, dialect_small,
):
    frame = benhallu_small if mode == 'benhallu' else dialect_small
    validator = getattr(make_splits, f'validate_{mode}')
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    for invalid in [frame.drop(columns=['question']), duplicate, frame.iloc[:0]]:
        with pytest.raises(ValueError):
            validator(invalid)
    with pytest.raises(ValueError):
        validator(frame, require_full=True)


def test_splits_benhallu_rejects_track_label_and_type_mismatch(benhallu_small):
    for track, column, value in [
        ('Track_A', 'ground_truth_label', 1),
        ('Track_B', 'ground_truth_label', 0),
        ('Track_A', 'hallucination_type', 'Factualness'),
        ('Track_B', 'hallucination_type', 'None'),
    ]:
        frame = benhallu_small.copy(deep=True)
        index = frame.index[frame['track'].eq(track)][0]
        frame.loc[index, column] = value
        with pytest.raises(ValueError):
            make_splits.validate_benhallu(frame)


def test_splits_validation_normalizes_without_mutating(benhallu_small, dialect_small):
    frame = benhallu_small.copy(deep=True)
    frame['task'] = frame['task'].str.upper()
    frame.loc[frame['track'].eq('Track_A'), 'hallucination_type'] = None
    frame['context'] = None
    before = frame.copy(deep=True)
    result = make_splits.validate_benhallu(frame)
    assert set(result['task']) == TASKS
    assert all(value == '' for value in result['context'])
    assert result.loc[result['track'].eq('Track_A'), 'hallucination_type'].eq('None').all()
    assert_frame_equal(frame, before)
    dialect_before = dialect_small.copy(deep=True)
    make_splits.validate_dialect(dialect_small)
    assert_frame_equal(dialect_small, dialect_before)


@pytest.mark.parametrize('mode,mock_name,filenames', [
    ('benhallu', 'mock_benhallu_16000.csv', list(SPLIT_FILES.values())),
    ('dialect', 'mock_dialect_4000.csv', ['dialect_ccc_subset_300.csv']),
])
def test_splits_mock_cli_subprocess(mode, mock_name, filenames, tmp_path):
    destination = tmp_path / mode
    args = [
        '--mode', mode, '--generate_mock_data', '--seed', '42',
        '--output_dir', destination,
    ]
    _run_cli('make_splits.py', args, tmp_path)
    expected = [mock_name, *filenames, 'manifest.json']
    snapshots = {name: (destination / name).read_bytes() for name in expected}
    _run_cli('make_splits.py', args, tmp_path)
    assert snapshots == {name: (destination / name).read_bytes() for name in expected}
    mock = pd.read_csv(destination / mock_name, keep_default_na=False)
    assert list(mock.columns) == (BENHALLU_COLUMNS if mode == 'benhallu' else DIALECT_COLUMNS)
    assert len(mock) == (16000 if mode == 'benhallu' else 4000)
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['mock'] is True


# Exact metrics --------------------------------------------------------------


@pytest.mark.parametrize('preds_a,preds_b,fpr,fnr,bhs', [
    ([0, 0], [1, 1, 1], 0, 0, 0),
    ([1, 1], [0, 0, 0], 1, 1, 100),
    ([1], [1, 1, 1], 1, 0, 50),
    ([0], [0, 0, 0], 0, 1, 50),
    ([0, 1, 0, 0], [1, 0], 0.25, 0.5, 37.5),
])
def test_benhallu_score_exact_rates(evaluation, preds_a, preds_b, fpr, fnr, bhs):
    result = evaluation.compute_benhallu_score(preds_a, preds_b)
    assert result['FPR_A'] == pytest.approx(fpr)
    assert result['FNR_B'] == pytest.approx(fnr)
    assert result['BHS'] == pytest.approx(bhs)


def test_benhallu_score_custom_weights(evaluation):
    result = evaluation.compute_benhallu_score([0, 0, 0, 1], [0, 1], w_a=0.25, w_b=0.75)
    assert result['FPR_A'] == pytest.approx(0.25)
    assert result['FNR_B'] == pytest.approx(0.5)
    assert result['BHS'] == pytest.approx(43.75)


@pytest.mark.parametrize('task', sorted(TASKS))
@pytest.mark.parametrize('prediction', [0, 1])
def test_benhallu_degenerate_judge_is_50_percent_for_each_task(
    evaluation, benhallu_full, task, prediction,
):
    rows = benhallu_full.loc[benhallu_full['task'].eq(task)]
    preds_a = [prediction] * rows['track'].eq('Track_A').sum()
    preds_b = [prediction] * rows['track'].eq('Track_B').sum()
    assert evaluation.compute_benhallu_score(preds_a, preds_b)['BHS'] == pytest.approx(50)


@pytest.mark.parametrize('a,b', [
    ([], [1]), ([0], []), ([], []), ([2], [1]), ([0], [-1]),
    ([0.5], [1]), ([0], [float('nan')]), ([float('inf')], [1]),
    ([0], [float('-inf')]),
])
def test_benhallu_score_rejects_invalid_predictions(evaluation, a, b):
    with pytest.raises(ValueError):
        evaluation.compute_benhallu_score(a, b)


@pytest.mark.parametrize('w_a,w_b', [
    (-0.1, 1.1), (1.1, -0.1), (0, 0), (0.2, 0.2),
    (float('nan'), 0.5), (0.5, float('inf')),
])
def test_benhallu_score_rejects_bad_weights(evaluation, w_a, w_b):
    with pytest.raises(ValueError):
        evaluation.compute_benhallu_score([0], [1], w_a=w_a, w_b=w_b)


def test_weight_sensitivity_has_nine_exact_weight_pairs(evaluation):
    rows = evaluation.compute_weight_sensitivity([0, 1, 0, 0], [0, 1])
    assert isinstance(rows, list)
    assert len(rows) == 9
    assert sorted(row['w_a'] for row in rows) == pytest.approx(np.arange(1, 10) / 10)
    for row in rows:
        assert row['w_a'] + row['w_b'] == pytest.approx(1)
        assert row['BHS'] == pytest.approx(
            100 * (row['w_a'] * 0.25 + row['w_b'] * 0.5),
        )


@pytest.mark.parametrize('x,y,expected', [
    ([1, 2, 3], [1, 2, 3], 1),
    ([-1, 0, 1], [1, 0, -1], -1),
    ([-1, -1, 1, 1], [-1, 1, -1, 1], 0),
    ([0, 1, 2], [1, 2, 3], 4 / 7),
    ([0, 1, 2], [0, 2, 4], 8 / 13),
    ([1, 1, 1], [2, 2, 2], 0),
    ([1, 1, 1], [0, 1, 2], 0),
])
def test_lins_ccc_exact_population_formula(evaluation, x, y, expected):
    assert evaluation.compute_lins_ccc(x, y) == pytest.approx(expected, abs=1e-12)


def test_lins_ccc_plus_two_offset_penalizes_perfectly_correlated_vectors(evaluation):
    reference = np.asarray([0, 1, 2, 3], dtype=float)
    shifted = reference + 2
    variance = float(np.var(reference, ddof=0))
    expected = 2 * variance / (2 * variance + 4)
    result = evaluation.compute_lins_ccc(reference, shifted)
    assert result == pytest.approx(expected, rel=0, abs=1e-12)
    assert result == pytest.approx(5 / 13, rel=0, abs=1e-12)
    assert result < 1


@pytest.mark.parametrize('x,y', [([2, 2, 2], [2, 2, 2]), ([0], [0])])
def test_lins_ccc_zero_denominator_is_undefined(evaluation, x, y):
    result = evaluation.compute_lins_ccc(x, y)
    assert isinstance(result, float)
    assert math.isnan(result)


@pytest.mark.parametrize('x,y', [
    ([], []), ([], [1]), ([1, 2], [1]),
    ([0, float('nan')], [0, 1]), ([0, 1], [0, float('inf')]),
])
def test_lins_ccc_rejects_invalid_inputs(evaluation, x, y):
    with pytest.raises(ValueError):
        evaluation.compute_lins_ccc(x, y)


def test_critical_bias_sensitivity_is_anchor_minus_chittagong_in_ccc_units(evaluation):
    performance = {'Tangail': 0.8, 'Standard_Bengali': 0.9, 'Chittagong': 0.3}
    assert evaluation.compute_critical_bias_sensitivity(performance) == pytest.approx(0.5)
    assert evaluation.compute_critical_bias_sensitivity(
        performance, anchor='Standard_Bengali',
    ) == pytest.approx(0.6)
    performance['Tangail'] = 0.1
    assert evaluation.compute_critical_bias_sensitivity(performance) == pytest.approx(-0.2)


# Conservative bilingual parsing and collapse detection -----------------------


@pytest.mark.parametrize('text,expected', [
    ('yes', 1), (' NO ', 0), ('true', 1), ('FALSE', 0), ('হ্যাঁ', 1), ('না', 0),
    ('<verdict>Yes</verdict>', 1), ('<verdict>no.</verdict>', 0),
    ('<verdict>হ্যাঁ।</verdict>', 1), ('Final Verdict: Yes.', 1),
    ('Final Verdict: না।', 0), ('{"hallucinated": true}', 1),
    ('{"hallucinated": false}', 0), ('{"hallucinated": "হ্যাঁ"}', 1),
    ('{"hallucinated": "না"}', 0),
    ('Reasoning: the candidate contradicts the context.\n<verdict>yes</verdict>', 1),
    ('<verdict>yes</verdict>\nFinal Verdict: yes', 1),
    ('Reasoning complete. Final Verdict: Yes', 1),
])
def test_parse_judge_accepts_supported_unambiguous_formats(evaluation, text, expected):
    result = evaluation.parse_judge_output(text)
    assert result['parsed_verdict'] == expected
    assert result['parse_failure'] is False
    assert result['is_collapsed'] is False


@pytest.mark.parametrize('normalization', ['NFC', 'NFD'])
def test_parse_judge_bengali_unicode_normalization(evaluation, normalization):
    for word, expected in [('হ্যাঁ', 1), ('না', 0)]:
        text = unicodedata.normalize(normalization, f'<verdict>{word}</verdict>')
        assert evaluation.parse_judge_output(text)['parsed_verdict'] == expected


@pytest.mark.parametrize('text', [
    '', '   ', 'maybe', 'The answer is yes, but I am not certain.',
    'No hallucination is present in this answer.', 'yes or no',
    '<verdict>maybe</verdict>', '<verdict>yes</verdict><verdict>no</verdict>',
    '<verdict>yes</verdict>\nFinal Verdict: no',
    '<verdict>no</verdict>\n{"hallucinated": true}',
    'Final Verdict: yes\nFinal Verdict: no',
    '{"hallucinated": true}\n{"hallucinated": false}',
])
def test_parse_judge_rejects_prose_and_conflicts(evaluation, text):
    result = evaluation.parse_judge_output(text)
    assert result['parsed_verdict'] is None
    assert result['parse_failure'] is True


@pytest.mark.parametrize('phrase,separator', [
    ('the same five words repeat', ' '),
    ('this answer repeats without making progress', '. '),
    ('এই উত্তর শুধু একই কথা বলে', '। '),
])
def test_parse_judge_detects_four_consecutive_long_ngrams(evaluation, phrase, separator):
    text = separator.join([phrase] * 4) + '\n<verdict>yes</verdict>'
    assert evaluation.parse_judge_output(text)['is_collapsed'] is True


@pytest.mark.parametrize('text', [
    ' '.join(['these five words repeat here'] * 3) + '\n<verdict>no</verdict>',
    ' '.join(['only four words here'] * 4) + '\n<verdict>no</verdict>',
])
def test_parse_judge_does_not_overflag_short_repetition(evaluation, text):
    assert evaluation.parse_judge_output(text)['is_collapsed'] is False


@pytest.mark.parametrize('text,metadata,collapsed', [
    ('yes', {'completion_tokens': 1023}, False),
    ('yes', {'completion_tokens': 1024}, True),
    ('yes', {'completion_tokens': 1025}, True),
    ('<verdict>yes', {'finish_reason': 'length'}, True),
    ('yes', {'finish_reason': 'length'}, True),
    ('<verdict>yes</verdict>', {'completion_tokens': 1024}, False),
    ('<verdict>no</verdict>', {'finish_reason': 'length'}, False),
    ('yes', {'completion_tokens': 20, 'max_tokens': 20}, True),
])
def test_parse_judge_token_budget_and_closing_tag(evaluation, text, metadata, collapsed):
    assert evaluation.parse_judge_output(text, **metadata)['is_collapsed'] is collapsed


@pytest.mark.parametrize('text,expected', [
    ('<score>7.5</score>', 7.5), ('{"score": 7.5}', 7.5),
    ('Final Score: 7.5', 7.5), ('7.5', 7.5), ('0', 0), ('10', 10),
    ('<score>৭.৫</score>', 7.5), ('৭.৫', 7.5), ('১০', 10),
    ('Final Score: ৭.৫', 7.5),
    ('<score>7.5</score>\nFinal Score: 7.5', 7.5),
    ('Reasoning complete. Final Score: 7.5', 7.5),
])
def test_parse_score_supported_formats_and_bengali_digits(evaluation, text, expected):
    result = evaluation.parse_score_output(text)
    assert result['parsed_score'] == pytest.approx(expected)
    assert result['parse_failure'] is False
    assert result['is_collapsed'] is False


@pytest.mark.parametrize('text', [
    '', '-0.1', '10.1', 'nan', 'inf', 'I would give this answer 7.5 out of 10.',
    '<score>11</score>', '{"score": true}',
    '<score>7</score><score>8</score>', '<score>7</score>\nFinal Score: 8',
    '<score>7</score>\n{"score": 8}', 'Final Score: 7\nFinal Score: 8',
])
def test_parse_score_rejects_invalid_prose_and_conflicts(evaluation, text):
    result = evaluation.parse_score_output(text)
    assert result['parsed_score'] is None
    assert result['parse_failure'] is True


@pytest.mark.parametrize('text,metadata,collapsed', [
    ('7.5', {'completion_tokens': 1023}, False),
    ('7.5', {'completion_tokens': 1024}, True),
    ('<score>7.5', {'finish_reason': 'length'}, True),
    ('<score>7.5</score>', {'completion_tokens': 1024}, False),
    ('<score>7.5</score>', {'finish_reason': 'length'}, False),
])
def test_parse_score_collapse_metadata(evaluation, text, metadata, collapsed):
    assert evaluation.parse_score_output(text, **metadata)['is_collapsed'] is collapsed


def test_parse_score_detects_bengali_repetition(evaluation):
    text = ' '.join(['এই উত্তর শুধু একই কথা বলে'] * 4) + '\n<score>7</score>'
    assert evaluation.parse_score_output(text)['is_collapsed'] is True


@pytest.mark.parametrize('parser_name,tag,first,revision,parsed_key', [
    ('parse_judge_output', 'verdict', 'yes', 'no', 'parsed_verdict'),
    ('parse_score_output', 'score', '7.5', '2', 'parsed_score'),
])
@pytest.mark.parametrize('metadata', [
    {'finish_reason': 'length'}, {'completion_tokens': 1024},
])
def test_parse_unfinished_revision_invalidates_earlier_closed_tag(
    evaluation, parser_name, tag, first, revision, parsed_key, metadata,
):
    text = f'<{tag}>{first}</{tag}>\n<{tag}>{revision}'
    result = getattr(evaluation, parser_name)(text, **metadata)
    assert result['is_collapsed'] is True
    assert result['parse_failure'] is True
    assert result[parsed_key] is None


@pytest.mark.parametrize('parser_name,parsed_key,text', [
    ('parse_judge_output', 'parsed_verdict', '{"hallucinated": true, "hallucinated": false}'),
    ('parse_judge_output', 'parsed_verdict', '{"hallucinated": "হ্যাঁ", "hallucinated": "না"}'),
    ('parse_judge_output', 'parsed_verdict', '{"hallucinated": true, "hallucinated": false'),
    ('parse_score_output', 'parsed_score', '{"score": 7.5, "score": 2}'),
    ('parse_score_output', 'parsed_score', '{"score": "৭.৫", "score": "২"}'),
    ('parse_score_output', 'parsed_score', '{"score": 7.5, "score": 2'),
])
def test_parse_rejects_conflicting_duplicate_json_keys(evaluation, parser_name, parsed_key, text):
    result = getattr(evaluation, parser_name)(text)
    assert result['parse_failure'] is True
    assert result[parsed_key] is None


# Prompt isolation and durable/resumable evaluation ---------------------------


@pytest.mark.parametrize('mode', ['benhallu', 'dialect'])
@pytest.mark.parametrize('prompt_mode', ['zero_shot', 'cot'])
def test_build_prompt_contains_inputs_but_not_evaluation_metadata(evaluation, mode, prompt_mode):
    if mode == 'benhallu':
        row = {
            'id': 'SECRET_RECORD_ID', 'task': 'SECRET_TASK', 'track': 'SECRET_TRACK',
            'hallucination_type': 'SECRET_TYPE', 'ground_truth_label': 'SECRET_LABEL',
            'question': 'QUESTION_SENTINEL', 'context': 'CONTEXT_SENTINEL',
            'candidate_answer': 'CANDIDATE_SENTINEL',
        }
        visible = ['question', 'context', 'candidate_answer']
    else:
        row = {
            'id': 'SECRET_RECORD_ID', 'dialect': 'SECRET_DIALECT',
            'question_type': 'SECRET_QUESTION_TYPE',
            'human_or_primary_score': 'SECRET_HUMAN_SCORE',
            'question': 'QUESTION_SENTINEL', 'response': 'RESPONSE_SENTINEL',
        }
        visible = ['question', 'response']
    before = row.copy()
    prompt = evaluation.build_prompt(row, mode=mode, prompt_mode=prompt_mode)
    assert isinstance(prompt, str) and prompt.strip()
    for key in visible:
        assert row[key] in prompt
    for key in row.keys() - set(visible):
        assert row[key] not in prompt
    assert row == before
    # Changing hidden metadata must not influence the prompt at all.
    changed = {key: (value if key in visible else 'OTHER_' + value) for key, value in row.items()}
    assert evaluation.build_prompt(changed, mode=mode, prompt_mode=prompt_mode) == prompt


def test_run_evaluation_checkpoint_resume_and_record_metadata(evaluation, benhallu_small, tmp_path):
    judge = RecordingJudge(evaluation)
    rows = _evaluate(evaluation, benhallu_small, judge, tmp_path)
    assert isinstance(rows, list)
    assert len(judge.prompts) == len(benhallu_small)
    stored = _read_jsonl(_checkpoint(tmp_path))
    assert len(rows) == len(stored) == len(benhallu_small)
    assert {row['id'] for row in stored} == set(benhallu_small['id'])
    expected = benhallu_small.set_index('id')
    for row in stored:
        assert row['parsed_verdict'] == 1
        assert row['parse_failure'] is False
        assert row['is_collapsed'] is False
        for column in ['task', 'track', 'hallucination_type', 'ground_truth_label']:
            assert row[column] == expected.loc[row['id'], column]
        assert row['prompt_mode'] == 'zero_shot'
        assert row['split_name'] == 'micro'
        assert row['prompt_tokens'] == 13
        assert row['completion_tokens'] == 7
    resumed = RecordingJudge(evaluation)
    _evaluate(evaluation, benhallu_small, resumed, tmp_path)
    assert resumed.prompts == []
    assert _read_jsonl(_checkpoint(tmp_path)) == stored


def test_run_evaluation_fsyncs_before_requesting_next_prediction(
    evaluation, benhallu_small, tmp_path, monkeypatch,
):
    fsync_calls = []
    actual_fsync = os.fsync

    def spy_fsync(fd):
        actual_fsync(fd)
        fsync_calls.append(fd)

    monkeypatch.setattr(os, 'fsync', spy_fsync)
    if getattr(evaluation, 'fsync', None) is actual_fsync:
        monkeypatch.setattr(evaluation, 'fsync', spy_fsync)
    calls = []

    def judge(prompt):
        if calls:
            assert len(_read_jsonl(_checkpoint(tmp_path))) == len(calls)
            assert len(fsync_calls) >= len(calls)
        calls.append(prompt)
        return evaluation.InferenceResponse('<verdict>no</verdict>')

    _evaluate(evaluation, benhallu_small, judge, tmp_path)
    assert len(fsync_calls) >= len(benhallu_small)


def test_run_evaluation_parse_failures_and_collapses_are_saved_and_resumed(
    evaluation, benhallu_small, tmp_path,
):
    frame = benhallu_small.iloc[:2].copy()
    responses = iter([
        evaluation.InferenceResponse('This is not a verdict.'),
        evaluation.InferenceResponse('<verdict>yes', completion_tokens=1024, finish_reason='length'),
    ])
    _evaluate(evaluation, frame, lambda prompt: next(responses), tmp_path)
    stored = _read_jsonl(_checkpoint(tmp_path))
    assert len(stored) == 2
    assert stored[0]['parsed_verdict'] is None
    assert stored[0]['parse_failure'] is True
    assert stored[1]['is_collapsed'] is True
    assert all('ground_truth_label' in row for row in stored)
    resumed = RecordingJudge(evaluation)
    _evaluate(evaluation, frame, resumed, tmp_path)
    assert resumed.prompts == []
    assert len(_read_jsonl(_checkpoint(tmp_path))) == 2


def test_run_evaluation_interrupted_run_resumes_only_remaining_ids(
    evaluation, benhallu_small, tmp_path,
):
    frame = benhallu_small.iloc[:4].copy()
    calls = []

    def interrupted(prompt):
        if len(calls) == 2:
            raise KeyboardInterrupt('Simulated interruption after two durable rows')
        calls.append(prompt)
        return evaluation.InferenceResponse('<verdict>yes</verdict>')

    with pytest.raises(KeyboardInterrupt):
        _evaluate(evaluation, frame, interrupted, tmp_path)
    checkpoint = _checkpoint(tmp_path)
    prefix = _read_jsonl(checkpoint)
    assert len(prefix) == 2
    resumed = RecordingJudge(evaluation)
    _evaluate(evaluation, frame, resumed, tmp_path)
    assert len(resumed.prompts) == 2
    stored = _read_jsonl(checkpoint)
    assert stored[:2] == prefix
    assert len(stored) == 4
    assert {row['id'] for row in stored} == set(frame['id'])


@pytest.mark.parametrize('suffix', [b'{"id": "torn', b'{"id": "\xe0\xa6'])
def test_run_evaluation_repairs_torn_trailing_json_without_repeating_ids(
    evaluation, benhallu_small, tmp_path, suffix,
):
    _evaluate(evaluation, benhallu_small, RecordingJudge(evaluation), tmp_path)
    checkpoint = _checkpoint(tmp_path)
    prefix = checkpoint.read_bytes()
    assert prefix.endswith(b'\n')
    with checkpoint.open('ab') as stream:
        stream.write(suffix)
    resumed = RecordingJudge(evaluation)
    _evaluate(evaluation, benhallu_small, resumed, tmp_path)
    assert resumed.prompts == []
    assert checkpoint.read_bytes() == prefix


def test_run_evaluation_keeps_valid_final_json_without_newline(
    evaluation, benhallu_small, tmp_path,
):
    _evaluate(evaluation, benhallu_small, RecordingJudge(evaluation), tmp_path)
    checkpoint = _checkpoint(tmp_path)
    expected = _read_jsonl(checkpoint)
    checkpoint.write_bytes(checkpoint.read_bytes().rstrip(b'\r\n'))
    resumed = RecordingJudge(evaluation)
    _evaluate(evaluation, benhallu_small, resumed, tmp_path)
    assert resumed.prompts == []
    assert _read_jsonl(checkpoint) == expected


@pytest.mark.parametrize('corruption', ['internal_json', 'duplicate_id'])
def test_run_evaluation_rejects_internal_corruption_and_duplicate_existing_ids(
    evaluation, benhallu_small, tmp_path, corruption,
):
    _evaluate(evaluation, benhallu_small, RecordingJudge(evaluation), tmp_path)
    checkpoint = _checkpoint(tmp_path)
    lines = checkpoint.read_bytes().splitlines(keepends=True)
    if corruption == 'internal_json':
        lines[1] = b'{malformed internal record}\n'
    else:
        lines.append(lines[0])
    corrupted = b''.join(lines)
    checkpoint.write_bytes(corrupted)
    judge = RecordingJudge(evaluation)
    with pytest.raises(ValueError):
        _evaluate(evaluation, benhallu_small, judge, tmp_path)
    assert judge.prompts == []
    assert checkpoint.read_bytes() == corrupted


@pytest.mark.parametrize('change', ['seed', 'backend', 'question', 'ids', 'removed_row'])
def test_run_evaluation_rejects_config_and_dataset_mismatch(
    evaluation, benhallu_small, tmp_path, change,
):
    _evaluate(evaluation, benhallu_small, RecordingJudge(evaluation), tmp_path)
    checkpoint = _checkpoint(tmp_path)
    before = checkpoint.read_bytes()
    frame = benhallu_small.copy(deep=True)
    options = {}
    if change == 'seed':
        options['seed'] = 43
    elif change == 'backend':
        options['backend_identity'] = 'different-backend'
    elif change == 'question':
        frame.loc[0, 'question'] += ' Changed dataset content.'
    elif change == 'ids':
        frame.loc[0, 'id'] += '_changed'
    else:
        frame = frame.iloc[:-1].copy()
    judge = RecordingJudge(evaluation)
    with pytest.raises(ValueError):
        _evaluate(evaluation, frame, judge, tmp_path, **options)
    assert judge.prompts == []
    assert checkpoint.read_bytes() == before


def test_run_evaluation_unsafe_model_names_are_safe_and_collision_resistant(
    evaluation, benhallu_small, tmp_path,
):
    frame = benhallu_small.iloc[:2].copy()
    for name in ['organization/model:revision', 'organization:model/revision']:
        _evaluate(evaluation, frame, RecordingJudge(evaluation), tmp_path, model_name=name)
    paths = list(tmp_path.rglob('*.jsonl'))
    assert len(paths) == 2
    assert paths[0].name != paths[1].name
    for path in paths:
        assert path.parent == tmp_path
        assert re.fullmatch(r'[A-Za-z0-9_.-]+\.jsonl', path.name)
        assert 'zero_shot' in path.name
        assert 'micro' in path.name


def test_run_evaluation_lock_prevents_concurrent_checkpoint_writers(
    evaluation, benhallu_small, tmp_path,
):
    frame = benhallu_small.iloc[:2].copy()
    source = tmp_path / 'lock_input.csv'
    frame.to_csv(source, index=False)
    outputs = tmp_path / 'outputs'
    attempted, inferred = tmp_path / 'attempted', tmp_path / 'inferred'
    child_code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pandas as pd
import eval as evaluation
frame = pd.read_csv(sys.argv[2], keep_default_na=False)
def judge(prompt):
    Path(sys.argv[5]).write_text('inference attempted', encoding='utf-8')
    return evaluation.InferenceResponse('<verdict>yes</verdict>')
Path(sys.argv[4]).write_text('evaluation attempted', encoding='utf-8')
try:
    evaluation.run_evaluation(
        frame, judge, model_name='pytest_model', prompt_mode='zero_shot',
        split_name='micro', mode='benhallu', output_dir=sys.argv[3], seed=42,
        backend_identity='pytest-recording-v1',
    )
except (ValueError, RuntimeError, OSError) as exc:
    print(str(exc))
    raise SystemExit(23)
'''
    calls = []

    def judge(prompt):
        if not calls:
            # The outer run holds its lock while waiting for this prediction.
            # A competing writer may fail fast or wait, but must not infer.
            try:
                child = subprocess.run(
                    [sys.executable, '-c', child_code, str(ROOT), str(source),
                     str(outputs), str(attempted), str(inferred)],
                    cwd=tmp_path, capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=15, check=False,
                )
            except subprocess.TimeoutExpired:
                pass
            else:
                assert child.returncode == 23, child.stdout + '\n' + child.stderr
                assert re.search(r'lock|concurren|already|in use', child.stdout, re.I)
            assert attempted.is_file(), 'Competing process did not reach evaluation'
            assert not inferred.exists(), 'A second writer invoked inference under the lock'
        calls.append(prompt)
        return evaluation.InferenceResponse('<verdict>yes</verdict>')

    _evaluate(evaluation, frame, judge, outputs)
    assert len(calls) == len(frame)
    assert len(_read_jsonl(_checkpoint(outputs))) == len(frame)


def test_run_evaluation_dialect_saves_scores_and_ground_truth(evaluation, dialect_small, tmp_path):
    judge = RecordingJudge(evaluation, '<score>৭.৫</score>')
    rows = _evaluate(evaluation, dialect_small, judge, tmp_path, mode='dialect')
    stored = _read_jsonl(_checkpoint(tmp_path))
    assert len(rows) == len(stored) == len(dialect_small)
    expected = dialect_small.set_index('id')
    for row in stored:
        assert row['parsed_score'] == pytest.approx(7.5)
        assert row['parse_failure'] is False
        for column in ['dialect', 'question_type', 'human_or_primary_score']:
            assert row[column] == expected.loc[row['id'], column]


def test_generate_reports_exports_required_tables_and_type_coverage(
    evaluation, benhallu_small, tmp_path,
):
    outputs, results = tmp_path / 'outputs', tmp_path / 'results'
    for prompt_mode in ['zero_shot', 'cot']:
        _evaluate(
            evaluation, benhallu_small, RecordingJudge(evaluation), outputs,
            prompt_mode=prompt_mode,
        )
    reports = evaluation.generate_reports(outputs, results)
    assert isinstance(reports, dict)
    assert REPORT_KEYS <= reports.keys()
    for name in REPORT_KEYS:
        assert isinstance(reports[name], pd.DataFrame)
        assert not reports[name].empty
        assert (results / f'{name}.csv').is_file()
        assert (results / f'{name}.md').is_file()
    granular = reports['table2_granular_hallucination_types']
    values = set(granular.astype(str).to_numpy().ravel())
    assert TYPES <= values
    overall = reports['table1_overall_and_per_task_bhs']
    assert 'BHS' in overall.columns
    assert np.all(np.asarray(overall['BHS'], dtype=float) == 50)
    assert TASKS <= set(overall.astype(str).to_numpy().ravel())


def test_generate_reports_rejects_tampered_track_truth_metadata(
    evaluation, benhallu_small, tmp_path,
):
    outputs, results = tmp_path / 'outputs', tmp_path / 'results'
    _evaluate(evaluation, benhallu_small, RecordingJudge(evaluation), outputs)
    checkpoint = _checkpoint(outputs)
    rows = _read_jsonl(checkpoint)
    tampered = next(row for row in rows if row['track'] == 'Track_B')
    assert tampered['ground_truth_label'] == 1
    tampered['track'] = 'Track_A'
    # Keep the type consistent with the new track, isolating the truth mismatch.
    tampered['hallucination_type'] = 'None'
    _write_jsonl(checkpoint, rows)
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError):
        evaluation.generate_reports(outputs, results)
    assert checkpoint.read_bytes() == before
    assert not list(results.glob('*.csv'))


def test_run_evaluation_rejects_openai_judge_seed_mismatch_before_call(
    evaluation, benhallu_small, tmp_path, monkeypatch,
):
    judge_calls, network_calls = [], []

    def forbidden_judge_call(self, prompt):
        judge_calls.append(prompt)
        raise AssertionError('Seed mismatch must be rejected before calling the judge')

    def forbidden_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError('This test must never access the network')

    monkeypatch.setattr(urllib.request, 'urlopen', forbidden_network)
    monkeypatch.setattr(evaluation.OpenAICompatibleJudge, '__call__', forbidden_judge_call)
    monkeypatch.delenv('PYTEST_PIPELINE_API_KEY', raising=False)
    judge = evaluation.OpenAICompatibleJudge(
        model_name='pytest_seed_model', base_url='https://example.invalid/v1',
        api_key_env='PYTEST_PIPELINE_API_KEY', seed=7, retries=0,
    )
    with pytest.raises(ValueError, match='(?i)seed'):
        evaluation.run_evaluation(
            benhallu_small, judge, model_name='pytest_seed_model',
            prompt_mode='zero_shot', split_name='micro', seed=42,
            output_dir=tmp_path / 'outputs',
        )
    assert judge_calls == []
    assert network_calls == []


@pytest.mark.parametrize('a,b,fpr,fnr,valid_bhs', [
    ([0, None, 0, 0], [1, None, 0, 1], 25, 50, 25),
    ([None] * 4, [None] * 4, 100, 100, None),
])
def test_generate_reports_penalizes_failed_parses_without_dropping_coverage(
    evaluation, report_benhallu, tmp_path, a, b, fpr, fnr, valid_bhs,
):
    outputs = tmp_path / 'outputs'
    predictions = dict(zip(report_benhallu['id'], a + b))
    _evaluate(
        evaluation, report_benhallu,
        _prediction_judge(evaluation, report_benhallu, predictions), outputs,
    )
    reports = evaluation.generate_reports(outputs, tmp_path / 'results', w_a=0.25, w_b=0.75)
    table = reports['table1_overall_and_per_task_bhs']
    overall = table.loc[table['task'].eq('overall')].iloc[0]
    failed = sum(value is None for value in a + b)
    assert overall['N'] == overall['Completed_N'] == overall['Expected_N'] == 8
    assert overall['Track_A_N'] == overall['Track_B_N'] == 4
    assert overall['Parse_Failure_N'] == failed
    assert overall['Parse_Failure_Rate (%)'] == pytest.approx(100 * failed / 8)
    assert overall['Track_A_FPR (%)'] == pytest.approx(fpr)
    assert overall['Track_B_FNR (%)'] == pytest.approx(fnr)
    assert overall['BHS'] == pytest.approx(0.25 * fpr + 0.75 * fnr)
    assert overall['BenHalluScore (%)'] == pytest.approx(overall['BHS'])
    if valid_bhs is None:
        assert pd.isna(overall['Valid_Only_BHS (%)'])
    else:
        assert overall['Valid_Only_BHS (%)'] == pytest.approx(valid_bhs)
    granular = reports['table2_granular_hallucination_types']
    factual = granular.loc[granular['hallucination_type'].eq('Factualness')].iloc[0]
    assert factual['Track_B_N'] == 4
    assert factual['Parse_Failure_N'] == sum(value is None for value in b)
    assert factual['Track_B_FNR (%)'] == pytest.approx(fnr)
    if valid_bhs is None:
        assert pd.isna(factual['Valid_Only_FNR (%)'])
    else:
        assert factual['Valid_Only_FNR (%)'] == pytest.approx(100 / 3)


@pytest.mark.parametrize('baseline_correct,cot_correct,expected_delta', [
    (False, True, -100), (True, False, 100),
])
def test_generate_reports_cot_delta_sign_is_change_in_error_not_accuracy(
    evaluation, report_benhallu, tmp_path, baseline_correct, cot_correct, expected_delta,
):
    outputs = tmp_path / 'outputs'
    for prompt_mode, correct in [('zero_shot', baseline_correct), ('cot', cot_correct)]:
        predictions = {
            row['id']: row['ground_truth_label'] if correct else 1 - row['ground_truth_label']
            for row in report_benhallu.to_dict(orient='records')
        }
        _evaluate(
            evaluation, report_benhallu,
            _prediction_judge(evaluation, report_benhallu, predictions), outputs,
            prompt_mode=prompt_mode,
        )
    reports = evaluation.generate_reports(outputs, tmp_path / 'results')
    delta = reports['table3_cot_delta_ablation']
    for task in ['overall', 'bn_qa']:
        row = delta.loc[delta['task'].eq(task)].iloc[0]
        assert row['Paired_N'] == 8
        assert bool(row['Both_Runs_Complete'])
        assert row['BHS_ZeroShot'] == pytest.approx(0 if baseline_correct else 100)
        assert row['BHS_CoT'] == pytest.approx(0 if cot_correct else 100)
        assert row['Delta_BHS'] == pytest.approx(expected_delta)


def test_generate_reports_pairs_shared_ids_with_unequal_incomplete_coverage(
    evaluation, report_benhallu, tmp_path,
):
    outputs = tmp_path / 'outputs'
    a_ids = report_benhallu.loc[report_benhallu['track'].eq('Track_A'), 'id'].tolist()
    b_ids = report_benhallu.loc[report_benhallu['track'].eq('Track_B'), 'id'].tolist()
    paired_ids = {a_ids[0], b_ids[0]}
    retained = {
        'zero_shot': {a_ids[0], a_ids[1], a_ids[2], b_ids[0], b_ids[1]},
        'cot': {a_ids[0], b_ids[0], b_ids[2], b_ids[3]},
    }
    for prompt_mode in ['zero_shot', 'cot']:
        predictions = {}
        for row in report_benhallu.to_dict(orient='records'):
            # Shared IDs improve from wholly wrong to perfect; unpaired IDs do
            # the reverse, so comparing unpaired averages cannot pass this test.
            correct = (row['id'] in paired_ids) == (prompt_mode == 'cot')
            predictions[row['id']] = (
                row['ground_truth_label'] if correct else 1 - row['ground_truth_label']
            )
        _evaluate(
            evaluation, report_benhallu,
            _prediction_judge(evaluation, report_benhallu, predictions), outputs,
            prompt_mode=prompt_mode,
        )
    for checkpoint in outputs.glob('*.jsonl'):
        rows = _read_jsonl(checkpoint)
        keep = retained[rows[0]['prompt_mode']]
        # Truncating to durable whole records models independently interrupted runs.
        _write_jsonl(checkpoint, [row for row in rows if row['id'] in keep])
    reports = evaluation.generate_reports(outputs, tmp_path / 'results')
    overall = reports['table1_overall_and_per_task_bhs']
    overall = overall.loc[overall['task'].eq('overall')].set_index('prompt_mode')
    assert overall.loc['zero_shot', 'Completed_N'] == 5
    assert overall.loc['cot', 'Completed_N'] == 4
    assert np.all(np.asarray(overall['Expected_N']) == 8)
    assert not np.asarray(overall['Complete'], dtype=bool).any()
    assert overall.loc['zero_shot', 'BHS'] == pytest.approx(100 * (1 / 3 + 1 / 2) / 2)
    assert overall.loc['cot', 'BHS'] == pytest.approx(100 / 3)
    delta = reports['table3_cot_delta_ablation']
    row = delta.loc[delta['task'].eq('overall')].iloc[0]
    assert row['Paired_N'] == 2
    assert not bool(row['Both_Runs_Complete'])
    assert row['BHS_ZeroShot'] == pytest.approx(100)
    assert row['BHS_CoT'] == pytest.approx(0)
    assert row['Delta_BHS'] == pytest.approx(-100)
    sweep = reports['appendix_g_weight_sensitivity']
    assert sweep['Rank'].isna().all()
    assert sweep['Ranking_Stable'].isna().all()


@pytest.mark.parametrize('stable', [True, False])
def test_generate_reports_all_nine_sweep_weights_measure_rank_stability(
    evaluation, report_benhallu, tmp_path, stable,
):
    outputs = tmp_path / 'outputs'
    if stable:
        rates = {'better': (0, 0), 'worse': (1, 1)}
    else:
        rates = {'a_specialist': (0, 1), 'b_specialist': (1, 0)}
    for model, (fpr, fnr) in rates.items():
        predictions = {
            row['id']: fpr if row['track'] == 'Track_A' else 1 - fnr
            for row in report_benhallu.to_dict(orient='records')
        }
        _evaluate(
            evaluation, report_benhallu,
            _prediction_judge(evaluation, report_benhallu, predictions), outputs,
            model_name=model,
        )
    reports = evaluation.generate_reports(outputs, tmp_path / 'results')
    sweep = reports['appendix_g_weight_sensitivity']
    overall = sweep.loc[sweep['task'].eq('overall')]
    assert len(overall) == 18
    assert overall['Ranking_Stable'].notna().all()
    assert all(bool(value) == stable for value in overall['Ranking_Stable'])
    for model in rates:
        rows = overall.loc[overall['model_name'].eq(model)]
        assert sorted(rows['w_a']) == pytest.approx(np.arange(1, 10) / 10)
    for row in overall.to_dict(orient='records'):
        wa, wb = row['w_a'], row['w_b']
        scores = {model: 100 * (wa * fpr + wb * fnr) for model, (fpr, fnr) in rates.items()}
        score = scores[row['model_name']]
        assert wa + wb == pytest.approx(1)
        assert row['BHS'] == pytest.approx(score)
        assert row['Rank'] == 1 + sum(value < score - 1e-12 for value in scores.values())
    if not stable:
        tied = overall.loc[np.isclose(np.asarray(overall['w_a'], dtype=float), 0.5)]
        assert len(tied) == 2
        assert np.all(np.asarray(tied['Rank'], dtype=float) == 1)


def test_cli_offline_mock_evaluation_both_reports_only_and_dialect(tmp_path):
    splits, outputs, results = tmp_path / 'splits', tmp_path / 'outputs', tmp_path / 'results'
    _run_cli('make_splits.py', [
        '--generate_mock_data', '--mode', 'benhallu', '--seed', 42,
        '--output_dir', splits / 'benhallu',
    ], tmp_path)
    _run_cli('eval.py', [
        '--input_csv', splits / 'benhallu' / SPLIT_FILES['micro'],
        '--mode', 'benhallu', '--backend', 'always_yes', '--model_name', 'offline_yes',
        '--prompt_mode', 'both', '--output_dir', outputs, '--results_dir', results,
    ], tmp_path)
    checkpoints = list(outputs.glob('*.jsonl'))
    assert len(checkpoints) == 2
    for checkpoint in checkpoints:
        rows = _read_jsonl(checkpoint)
        assert len(rows) == 120
        assert all(row['parsed_verdict'] == 1 and not row['parse_failure'] for row in rows)
    assert {_read_jsonl(path)[0]['prompt_mode'] for path in checkpoints} == {'zero_shot', 'cot'}
    before = {path.name: path.read_bytes() for path in checkpoints}
    _run_cli('eval.py', [
        '--reports_only', '--output_dir', outputs, '--results_dir', results,
    ], tmp_path)
    assert {path.name: path.read_bytes() for path in outputs.glob('*.jsonl')} == before
    _run_cli('make_splits.py', [
        '--generate_mock_data', '--mode', 'dialect', '--seed', 42,
        '--output_dir', splits / 'dialect',
    ], tmp_path)
    _run_cli('eval.py', [
        '--input_csv', splits / 'dialect' / 'dialect_ccc_subset_300.csv',
        '--mode', 'dialect', '--backend', 'mock_score', '--model_name', 'offline_score',
        '--output_dir', outputs, '--results_dir', results,
    ], tmp_path)
    paths = list(outputs.glob('*.jsonl'))
    assert len(paths) == 3
    dialect_path = next(path for path in paths if _read_jsonl(path)[0]['mode'] == 'dialect')
    rows = _read_jsonl(dialect_path)
    assert len(rows) == 300
    assert {row['dialect'] for row in rows} == ANCHORS
    assert all(row['parsed_score'] == 7.5 and not row['parse_failure'] for row in rows)
    for name in REPORT_KEYS | {
        'appendix_g_weight_sensitivity', 'table4_dialect_ccc', 'table5_dialect_cbs',
    }:
        assert not pd.read_csv(results / f'{name}.csv').empty
        assert (results / f'{name}.md').stat().st_size > 0
    overall = pd.read_csv(results / 'table1_overall_and_per_task_bhs.csv')
    assert np.all(np.asarray(overall['BHS'], dtype=float) == 50)
    delta = pd.read_csv(results / 'table3_cot_delta_ablation.csv')
    assert np.all(np.asarray(delta['Delta_BHS'], dtype=float) == 0)
    dialect = pd.read_csv(results / 'table4_dialect_ccc.csv')
    assert set(dialect['dialect']) == ANCHORS
    assert np.allclose(np.asarray(dialect['CCC'], dtype=float), 0)
    cbs = pd.read_csv(results / 'table5_dialect_cbs.csv')
    assert np.allclose(np.asarray(cbs['CBS'], dtype=float), 0)
    assert (results / 'report_manifest.json').is_file()


def test_openai_transport_locked_payload_and_response_metadata(
    evaluation, offline_chat_transport, monkeypatch,
):
    transport, sleep, payload = offline_chat_transport
    transport.return_value = BytesIO(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
    monkeypatch.setenv('PYTEST_PIPELINE_API_KEY', 'synthetic-test-key')
    judge = evaluation.OpenAICompatibleJudge(
        model_name='offline-request-model', base_url='https://example.invalid/v1/',
        api_key_env='PYTEST_PIPELINE_API_KEY', seed=42, timeout=3, retries=2,
    )
    prompt = 'শুধু পরীক্ষার জন্য এই উত্তর যাচাই করুন।'
    response = judge(prompt)
    transport.assert_called_once()
    request = transport.call_args.args[0]
    assert transport.call_args.kwargs == {'timeout': 3}
    assert request.full_url == 'https://example.invalid/v1/chat/completions'
    assert request.get_method() == 'POST'
    assert request.get_header('Authorization') == 'Bearer synthetic-test-key'
    assert request.get_header('Content-type') == 'application/json'
    assert json.loads(request.data.decode('utf-8')) == {
        'model': 'offline-request-model', 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'max_tokens': 1024, 'seed': 42, 'stream': False,
    }
    assert isinstance(response, evaluation.InferenceResponse)
    assert response.text == '<verdict>হ্যাঁ</verdict>'
    assert response.prompt_tokens == 31
    assert response.completion_tokens == 9
    assert response.finish_reason == 'stop'
    assert response.provider_metadata == {
        key: payload[key] for key in ['id', 'model', 'system_fingerprint', 'created']
    }
    sleep.assert_not_called()


@pytest.mark.parametrize('status', [400, 403])
def test_openai_transport_nonretryable_4xx_fails_without_retry(
    evaluation, offline_chat_transport, status,
):
    transport, sleep, _ = offline_chat_transport
    transport.side_effect = _transport_http_error(status)
    judge = evaluation.OpenAICompatibleJudge(
        model_name='offline-request-model', base_url='https://example.invalid/v1',
        api_key_env='PYTEST_PIPELINE_API_KEY', seed=42, retries=3,
    )
    with pytest.raises(RuntimeError, match=f'HTTP {status}') as error:
        judge('Offline request')
    assert 'offline provider error detail' in str(error.value)
    transport.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize('status,retry_after,expected_delay', [
    (429, '120', 60), (503, None, 1),
])
def test_openai_transport_transient_http_retry_recovers_with_bounded_delay(
    evaluation, offline_chat_transport, status, retry_after, expected_delay,
):
    transport, sleep, payload = offline_chat_transport
    transport.side_effect = [
        _transport_http_error(status, retry_after),
        BytesIO(json.dumps(payload).encode('utf-8')),
    ]
    judge = evaluation.OpenAICompatibleJudge(
        model_name='offline-request-model', base_url='https://example.invalid/v1',
        api_key_env='PYTEST_PIPELINE_API_KEY', seed=42, retries=2,
    )
    response = judge('Offline request')
    assert response.text == payload['choices'][0]['message']['content']
    assert transport.call_count == 2
    sleep.assert_called_once_with(expected_delay)
    requests = [invocation.args[0] for invocation in transport.call_args_list]
    assert requests[0].data == requests[1].data
    assert all(json.loads(request.data)['seed'] == 42 for request in requests)


@pytest.mark.parametrize('failure_kind', ['http_503', 'connection'])
def test_openai_transport_transient_retry_exhaustion_is_bounded(
    evaluation, offline_chat_transport, failure_kind,
):
    transport, sleep, _ = offline_chat_transport
    retries = 2
    if failure_kind == 'http_503':
        transport.side_effect = [_transport_http_error(503) for _ in range(retries + 1)]
        message = 'HTTP 503'
    else:
        transport.side_effect = [
            urllib.error.URLError('offline connection reset') for _ in range(retries + 1)
        ]
        message = 'connection failed'
    judge = evaluation.OpenAICompatibleJudge(
        model_name='offline-request-model', base_url='https://example.invalid/v1',
        api_key_env='PYTEST_PIPELINE_API_KEY', seed=42, retries=retries,
    )
    with pytest.raises(RuntimeError, match=message):
        judge('Offline request')
    assert transport.call_count == retries + 1
    assert sleep.call_args_list == [call(1), call(2)]


def test_generate_reports_different_seeds_withhold_rank_and_stability(
    evaluation, report_benhallu, tmp_path,
):
    outputs = tmp_path / 'outputs'
    for seed in [42, 43]:
        predictions = {
            row['id']: row['ground_truth_label'] if seed == 42 else 1 - row['ground_truth_label']
            for row in report_benhallu.to_dict(orient='records')
        }
        _evaluate(
            evaluation, report_benhallu,
            _prediction_judge(evaluation, report_benhallu, predictions), outputs,
            model_name=f'model_seed_{seed}', seed=seed,
        )
    configs = [
        json.loads(path.read_text(encoding='utf-8')) for path in outputs.glob('*.meta.json')
    ]
    assert len(configs) == 2
    assert {config['seed'] for config in configs} == {42, 43}
    assert len({config['dataset_sha256'] for config in configs}) == 1
    assert len({config['split_name'] for config in configs}) == 1
    reports = evaluation.generate_reports(outputs, tmp_path / 'results')
    sweep = reports['appendix_g_weight_sensitivity']
    overall = sweep.loc[sweep['task'].eq('overall')]
    assert len(overall) == 18
    assert set(overall['model_name']) == {'model_seed_42', 'model_seed_43'}
    assert np.asarray(overall['Complete'], dtype=bool).all()
    assert np.all(np.asarray(overall['Completed_N']) == len(report_benhallu))
    assert set(overall['BHS']) == {0, 100}
    for model in ['model_seed_42', 'model_seed_43']:
        assert sorted(overall.loc[overall['model_name'].eq(model), 'w_a']) == pytest.approx(
            np.arange(1, 10) / 10,
        )
    assert sweep['Rank'].isna().all()
    assert sweep['Ranking_Stable'].isna().all()


@pytest.mark.parametrize('mode', ['benhallu', 'dialect'])
def test_splits_rare_stratum_omission_is_actionable_error(
    mode, benhallu_full, dialect_full, tmp_path,
):
    if mode == 'benhallu':
        frame = benhallu_full.copy(deep=True)
        rare_indices = frame.index[frame['track'].eq('Track_A') & frame['task'].eq('math')]
        frame.loc[rare_indices[2:], 'task'] = 'bn_qa'
        rare_rows = frame.loc[frame['track'].eq('Track_A') & frame['task'].eq('math')]
        assert len(rare_rows) == 2
        make_splits.validate_benhallu(frame, require_full=True)
        splitter = make_splits.make_benhallu_splits
        identifiers = ['Track_A', 'math']
    else:
        frame = dialect_full.copy(deep=True)
        rare_indices = frame.index[
            frame['dialect'].eq('Standard_Bengali') & frame['question_type'].eq('definitional')
        ]
        frame.loc[rare_indices[2:], 'question_type'] = 'factual'
        rare_rows = frame.loc[
            frame['dialect'].eq('Standard_Bengali') & frame['question_type'].eq('definitional')
        ]
        assert len(rare_rows) == 2
        make_splits.validate_dialect(frame, require_full=True)
        splitter = make_splits.make_dialect_subset
        identifiers = ['Standard_Bengali', 'definitional']
    # Two source rows satisfy sklearn's minimum count but have a quota of 0.2;
    # integer allocation omits them unless the splitter explicitly refuses it.
    destination = tmp_path / mode
    with pytest.raises(ValueError) as error:
        splitter(frame, destination, seed=42)
    message = str(error.value)
    assert all(identifier in message for identifier in identifiers)
    assert re.search(r'coverage|omitt|missing', message, re.I)
    assert re.search(r'increase|collect|more examples', message, re.I)
    assert not list(destination.rglob('*.csv'))
    assert not (destination / 'manifest.json').exists()


if __name__ == '__main__':
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), *sys.argv[1:]]))

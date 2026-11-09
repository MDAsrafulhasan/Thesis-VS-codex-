"""Offline contract tests for the v2 detector; mocks are not thesis evidence.

Run from the workspace root with::

    python -m pytest -q 01_Thesis/Code/Evaluation_Pipeline/test_pipeline_v2.py
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import urllib.request
from unittest.mock import Mock

import numpy as np
import pytest


PIPELINE_DIR = Path(__file__).resolve().parent
# Root pytest configuration still names the pre-organization evaluation directory.
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


@pytest.fixture(scope="module")
def v2_math():
    if not (PIPELINE_DIR / "pipeline_v2_math.py").is_file():
        pytest.skip("pipeline_v2_math.py has not been created yet")
    return importlib.import_module("pipeline_v2_math")


def test_math_contract_exports(v2_math):
    for name in (
        "canonicalize", "dual_views", "hash13", "sigmoid_score",
        "js_divergence", "smooth_extrema", "route", "polarity_rrf",
        "operational_bhs", "conformal_quantile", "prediction_set", "cad_logits",
    ):
        assert callable(getattr(v2_math, name))


@pytest.mark.parametrize("raw, expected", [
    ("  বাংলা\t+৩\n−২  /  -5  ", "বাংলা +৩ −২ / -5"),
    ("Ａ\u00a0Ｂ", "A B"),
    ("", ""),
])
def test_canonicalize_preserves_signs_and_collapses_whitespace(v2_math, raw, expected):
    assert v2_math.canonicalize(raw) == expected
    assert v2_math.canonicalize(expected) == expected


def test_dual_views_identity_protection_and_whole_token_mapping(v2_math):
    raw = "  ami, Python amir বাংলা -৩ +2  "
    assert v2_math.dual_views(raw) == {
        "native": raw.strip(), "code": raw.strip(),
    }
    views = v2_math.dual_views(
        raw, reviewed_map={"ami": "আমি", "Python": "বদল", "বাংলা": "bangla"},
        protected_terms={"Python"},
    )
    assert views["native"] == raw.strip()
    assert views["code"] == "আমি, Python amir bangla -৩ +2"
    assert raw == "  ami, Python amir বাংলা -৩ +2  "


def test_hash13_protocol_boundaries_and_whitespace(v2_math):
    tokens = [f"শব্দ{i}" for i in range(14)]
    assert v2_math.hash13(" ".join(tokens[:12])) == set()
    digest = lambda words: hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()
    assert v2_math.hash13(" ".join(tokens[:13])) == {digest(tokens[:13])}
    assert v2_math.hash13("\t\n".join(tokens)) == {
        digest(tokens[:13]), digest(tokens[1:]),
    }
    positive = " ".join(["+৩", *tokens[:12]])
    negative = " ".join(["-৩", *tokens[:12]])
    assert v2_math.hash13(positive).isdisjoint(v2_math.hash13(negative))
    assert len(v2_math.hash13(" ".join(["repeat"] * 20))) == 1


@pytest.mark.parametrize("faith, hall, expected", [
    (0, 0, .5), (1000, 1001, 1 / (1 + math.exp(-1))),
    (-1000, -1001, 1 / (1 + math.exp(1))),
    (-1e308, 1e308, 1), (1e308, -1e308, 0),
])
def test_sigmoid_score_stable_hallucination_orientation(v2_math, faith, hall, expected):
    assert v2_math.sigmoid_score(faith, hall) == pytest.approx(expected)


@pytest.mark.parametrize("p, q", [(0, 0), (1, 1), (.3, .3), (0, 1), (.1, .8)])
def test_js_divergence_bounds_symmetry_and_formula(v2_math, p, q):
    left, right = [p, 1 - p], [q, 1 - q]
    middle = [(a + b) / 2 for a, b in zip(left, right)]
    expected = sum(
        .5 * value * math.log(value / mean)
        for distribution in (left, right)
        for value, mean in zip(distribution, middle) if value
    )
    actual = v2_math.js_divergence(p, q)
    assert 0 <= actual <= math.log(2)
    assert actual == pytest.approx(expected)
    assert actual == pytest.approx(v2_math.js_divergence(q, p))


def test_smooth_extrema_matches_weighted_formula_and_hard_guards(v2_math):
    entails, contrads, weights, tau = [.2, .9], [.1, .8], [.25, .75], .2
    actual = v2_math.smooth_extrema(entails, contrads, tau=tau, weights=weights)
    expected_min = -tau * math.log(sum(w * math.exp(-e / tau) for w, e in zip(weights, entails)))
    expected_max = tau * math.log(sum(w * math.exp(c / tau) for w, c in zip(weights, contrads)))
    assert actual["smin_ent"] == pytest.approx(expected_min)
    assert actual["smax_con"] == pytest.approx(expected_max)
    assert actual["emin"] == min(entails)
    assert actual["cmax"] == max(contrads)
    assert actual["emin"] <= actual["smin_ent"] <= max(entails)
    assert min(contrads) <= actual["smax_con"] <= actual["cmax"]
    assert v2_math.smooth_extrema(entails, contrads, tau=tau, weights=[1, 3]) == pytest.approx(actual)


def test_smooth_extrema_empty_tiny_temperature_and_zero_weight_guards(v2_math):
    assert all(value is None for value in v2_math.smooth_extrema([], []).values())
    tiny = v2_math.smooth_extrema([.1, .9], [.2, .8], tau=1e-300)
    assert tiny["smin_ent"] == pytest.approx(.1)
    assert tiny["smax_con"] == pytest.approx(.8)
    guarded = v2_math.smooth_extrema([.95, .05], [.02, .99], weights=[1, 0])
    assert guarded["smin_ent"] == pytest.approx(.95)
    assert guarded["emin"] == .05
    assert guarded["cmax"] == .99


@pytest.mark.parametrize("context, is_math, entails, contrads, expected", [
    (True, False, [.95], [.05], "fast-faithful"),
    (True, False, [.05], [.95], "fast-hallucinated"),
    (True, False, [.95], [.95], "full-verification"),
    (True, False, [.5], [.5], "full-verification"),
    (True, False, [], [], "full-verification"),
    (False, False, [.95], [.05], "full-verification"),
    (False, False, [.05], [.95], "full-verification"),
    (True, True, [.95], [.05], "full-verification"),
    (True, True, [.05], [.95], "full-verification"),
    (True, False, [.85], [.15], "fast-faithful"),
    (True, False, [.85], [.85], "full-verification"),
])
def test_route_context_math_conflict_and_threshold_guards(
    v2_math, context, is_math, entails, contrads, expected,
):
    assert v2_math.route(context, is_math, entails, contrads) == expected


def test_polarity_rrf_never_turns_counterevidence_into_support(v2_math):
    rankings = {"+": [["shared", "support"], ["support"]],
                "-": [["counter", "shared"], ["counter"]]}
    actual = v2_math.polarity_rrf(rankings)
    assert actual["+"] == pytest.approx({"shared": 1 / 61, "support": 1 / 62 + 1 / 61})
    assert actual["-"] == pytest.approx({"counter": 2 / 61, "shared": 1 / 62})
    assert "counter" not in actual["+"]
    assert "support" not in actual["-"]
    assert v2_math.polarity_rrf({"+": [["a", "a", "b"]]})["+"] == pytest.approx({
        "a": 1 / 61, "b": 1 / 63,
    })


@pytest.mark.parametrize("predictions, expected", [([0, 1], 0), ([1, 0], 100), ([None, None], 100)])
def test_operational_bhs_is_error_percentage(v2_math, predictions, expected):
    assert v2_math.operational_bhs([0, 1], predictions)["op"] == expected


def test_operational_bhs_failures_both_tracks_and_valid_only(v2_math):
    actual = v2_math.operational_bhs([0, 0, 0, 1, 1, 1], [0, 1, None, 1, 0, None])
    assert actual["op"] == pytest.approx(200 / 3)
    assert actual["BHS"] == actual["op"]
    assert actual["valid_only"] == 50
    assert actual["coverage"] == pytest.approx(2 / 3)
    assert actual["wrong_a"] == actual["wrong_b"] == 2
    assert actual["failures_a"] == actual["failures_b"] == 1
    assert actual["fpr"] == actual["fnr"] == pytest.approx(2 / 3)
    assert v2_math.operational_bhs([0, 1], [None, 1])["valid_only"] is None
    weighted = v2_math.operational_bhs([0, 1], [None, 1], w_a=.8, w_b=.2)
    assert weighted["op"] == 80


@pytest.mark.parametrize("labels, predictions", [([], []), ([0], [0]), ([1], [0])])
def test_operational_bhs_absent_track_undefined(v2_math, labels, predictions):
    actual = v2_math.operational_bhs(labels, predictions)
    assert actual["op"] is None
    assert actual["valid_only"] is None


@pytest.mark.parametrize("scores, alpha, expected", [
    ([.4, .1, .3, .2], .4, .3),
    ([.4, .1, .3, .2], .2, .4),
    ([.4, .1, .3, .2], .1, math.inf),
    ([.2], .5, .2), ([.2], .1, math.inf), ([], .1, math.inf),
    ([.2, .2, .8], .5, .2),
])
def test_conformal_quantile_finite_sample_order_without_interpolation(v2_math, scores, alpha, expected):
    assert v2_math.conformal_quantile(scores, alpha) == expected


@pytest.mark.parametrize("prob, quantile, expected", [
    (.1, .2, {0}), (.9, .2, {1}), (.5, .5, {0, 1}),
    (.5, .4, set()), (0, 0, {0}), (1, 0, {1}), (.3, math.inf, {0, 1}),
])
def test_prediction_set_hallucinated_probability_and_inclusive_boundary(v2_math, prob, quantile, expected):
    assert set(v2_math.prediction_set(prob, quantile)) == expected


@pytest.mark.parametrize("alpha, temp", [(0, 1), (.5, 1), (2, .5)])
def test_cad_logits_uses_aligned_logits_then_normalizes(v2_math, alpha, temp):
    context, prior = np.array([[2., -1., 0.], [-3., 4., 1.]]), np.array([[0., 1., 2.], [1., 0., 2.]])
    logits = ((1 + alpha) * context - alpha * prior) / temp
    expected = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    actual = v2_math.cad_logits(context, prior, alpha=alpha, temp=temp)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=-1), 1)
    assert np.all(actual >= 0)
    shifted = v2_math.cad_logits(context + 10000, prior - 10000, alpha=alpha, temp=temp)
    np.testing.assert_allclose(shifted, actual, atol=1e-12)


@pytest.mark.parametrize("name, args, kwargs", [
    ("sigmoid_score", (math.nan, 0), {}),
    ("sigmoid_score", (0, math.inf), {}),
    ("js_divergence", (-.1, .5), {}),
    ("js_divergence", (.5, 1.1), {}),
    ("smooth_extrema", ([.5], [.5]), {"tau": 0}),
    ("smooth_extrema", ([.5], []), {}),
    ("smooth_extrema", ([.5], [.5]), {"weights": [0]}),
    ("smooth_extrema", ([.5], [.5]), {"weights": [-1]}),
    ("operational_bhs", ([0, 1], [0]), {}),
    ("operational_bhs", ([0, 2], [0, 1]), {}),
    ("operational_bhs", ([0, 1], [0, "1"]), {}),
    ("operational_bhs", ([0, 1], [0, math.nan]), {}),
    ("operational_bhs", ([0, 1], [0, 1]), {"w_a": .2, "w_b": .2}),
    ("conformal_quantile", ([.1],), {"alpha": 0}),
    ("conformal_quantile", ([.1],), {"alpha": 1}),
    ("conformal_quantile", ([math.nan],), {}),
    ("prediction_set", (1.1, .2), {}),
    ("prediction_set", (.5, math.nan), {}),
    ("cad_logits", ([0, 1], [0]), {}),
    ("cad_logits", ([0, 1], [0, 1]), {"alpha": -1}),
    ("cad_logits", ([0, 1], [0, 1]), {"temp": 0}),
    ("cad_logits", ([math.inf, 1], [0, 1]), {}),
])
def test_math_rejects_invalid_inputs(v2_math, name, args, kwargs):
    with pytest.raises(ValueError):
        getattr(v2_math, name)(*args, **kwargs)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Offline contract tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)


@pytest.fixture(scope="module")
def runner():
    return importlib.import_module("update_pipeline_evaluation")


@pytest.fixture
def canonical_row():
    return {
        "id": "SECRET_ID_Track_A", "task": "bn_qa", "track": "Track_A",
        "hallucination_type": "SECRET_TYPE", "ground_truth_label": 0,
        "question": "বাংলাদেশের রাজধানী কী?", "context": "বাংলাদেশের রাজধানী ঢাকা।",
        "candidate_answer": "ঢাকা।", "correct_answer": "SECRET_GOLD_ANSWER",
        "saved_predictions": "SECRET_SAVED_PREDICTION",
    }


def _token_response(faith=-2., hall=-.2, decoded="1"):
    return {
        "response": decoded, "eval_count": 1,
        "logprobs": [{"token": "1", "logprob": hall, "top_logprobs": [
            {"token": "0", "logprob": faith}, {"token": "1", "logprob": hall},
        ]}],
    }


def test_judge_prompt_allowlists_only_scientific_input(runner, canonical_row):
    original = copy.deepcopy(canonical_row)
    prompt = runner.judge_prompt(canonical_row)
    payload, suffix = json.JSONDecoder().raw_decode(prompt[len(runner.JUDGE_INSTRUCTION):])
    assert payload == {k: canonical_row[k] for k in ("question", "context", "candidate_answer")}
    assert suffix == len(json.dumps(payload, ensure_ascii=False))
    assert prompt.endswith("\nVerdict:")
    for forbidden in ("id", "track", "ground_truth_label", "hallucination_type", "correct_answer", "saved_predictions", "task"):
        assert forbidden not in payload
    for sentinel in ("SECRET_ID_Track_A", "SECRET_TYPE", "SECRET_GOLD_ANSWER", "SECRET_SAVED_PREDICTION"):
        assert sentinel not in prompt
    changed = {**canonical_row, "id": "OTHER", "track": "Track_B", "ground_truth_label": 1,
               "hallucination_type": "OTHER", "correct_answer": "OTHER"}
    assert runner.judge_prompt(changed) == prompt
    assert canonical_row == original


def test_parse_token_score_exact_labels_and_orientation(runner, v2_math):
    score, error = runner.parse_token_score(_token_response())
    assert error is None
    assert score == pytest.approx(v2_math.sigmoid_score(-2., -.2))
    assert runner.parse_token_score(_token_response(-.2, -2.))[0] < .5


@pytest.mark.parametrize("response, reason", [
    ({}, "missing_logprobs"),
    ({"logprobs": []}, "missing_logprobs"),
    ({"logprobs": [{"token": "1", "logprob": -.1}]}, "both_labels_not_in_first_token_top20"),
    ({"logprobs": [{"token": " 0", "logprob": -1., "top_logprobs": [
        {"token": "1", "logprob": -.1}]}]}, "both_labels_not_in_first_token_top20"),
    ({"logprobs": [{"token": "0", "logprob": -1., "top_logprobs": [
        {"token": "1\n", "logprob": -.1}, {"token": "10", "logprob": -.1}]}]},
     "both_labels_not_in_first_token_top20"),
    ({"logprobs": [{"token": "0", "logprob": -1.}, {"token": "1", "logprob": -.1}]},
     "both_labels_not_in_first_token_top20"),
    (_token_response(math.nan, -.1), "nonfinite_logprob"),
    (_token_response(-1., math.inf), "nonfinite_logprob"),
    (_token_response(-math.inf, -.1), "nonfinite_logprob"),
])
def test_parse_token_score_missing_censored_or_nonfinite_is_not_probability(runner, response, reason):
    assert runner.parse_token_score(response) == (None, reason)


def test_evaluate_judge_injected_generate_two_label_success(runner, canonical_row, v2_math):
    client = Mock(spec=runner.LocalOllama)
    client.generate.return_value = _token_response(decoded="no")
    result = runner.evaluate_judge(canonical_row, client, "fake-local-model")
    client.generate.assert_called_once_with("fake-local-model", runner.judge_prompt(canonical_row))
    assert result["prediction"] == 1
    assert result["parsed_verdict"] == 0
    assert result["p_hall"] == pytest.approx(v2_math.sigmoid_score(-2, -.2))
    assert result["decision_source"] == "two_label_score_fixed_threshold"
    assert result["failure"] is None
    assert result["score_error"] is None
    assert result["raw_response"] == client.generate.return_value
    assert result["jsd"] is None
    assert result["latency_seconds"] >= 0


@pytest.mark.parametrize("label, decoded", [(0, "no"), (1, "yes"), (0, "0"), (1, "1")])
def test_evaluate_judge_decoded_fallback_without_fabricated_probability(runner, canonical_row, label, decoded):
    client = Mock(spec=runner.LocalOllama)
    client.generate.return_value = {
        "response": decoded, "eval_count": 1,
        "logprobs": [{"token": str(label), "logprob": -.1}],
    }
    result = runner.evaluate_judge(canonical_row, client, "fake-local-model")
    assert result["prediction"] == label
    assert result["p_hall"] is None
    assert result["score_error"] == "both_labels_not_in_first_token_top20"
    assert result["decision_source"] == "decoded_label_fallback"
    assert result["failure"] is None


@pytest.mark.parametrize("response", [
    {"response": "unparseable", "eval_count": 1},
    {"response": "", "logprobs": []},
    {"response": "1", "logprobs": [{"token": "1", "logprob": "bad"}]},
    {"response": "1", "logprobs": [{"token": "1"}]},
    None,
])
def test_evaluate_judge_malformed_response_remains_failed(runner, canonical_row, response):
    client = Mock(spec=runner.LocalOllama)
    client.generate.return_value = response
    result = runner.evaluate_judge(canonical_row, client, "fake-local-model")
    assert result["prediction"] is None
    assert result["p_hall"] is None
    assert result["failure"]
    assert result["latency_seconds"] >= 0


def test_evaluate_judge_transport_error_is_recorded_not_faithful(runner, canonical_row):
    client = Mock(spec=runner.LocalOllama)
    client.generate.side_effect = RuntimeError("synthetic local transport failure")
    result = runner.evaluate_judge(canonical_row, client, "fake-local-model")
    assert result["failure"] == "RuntimeError: synthetic local transport failure"
    assert result["prediction"] is None
    assert result["p_hall"] is None


@pytest.mark.parametrize("tail", [b'{"id":', b'{"text":"\xe0\xa6'])
def test_read_journal_repairs_only_torn_tail(runner, tmp_path, tail):
    path = tmp_path / "journal.jsonl"
    prefix = (json.dumps({"id": "বাংলা"}, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(prefix + tail)
    assert runner.read_journal(path) == [{"id": "বাংলা"}]
    assert path.read_bytes() == prefix
    assert runner.read_journal(path) == [{"id": "বাংলা"}]


def test_read_journal_missing_and_complete_unterminated_line(runner, tmp_path):
    path = tmp_path / "journal.jsonl"
    assert runner.read_journal(path) == []
    assert not path.exists()
    path.write_bytes(b'{"id":"complete"}')
    assert runner.read_journal(path) == [{"id": "complete"}]
    assert path.read_bytes() == b'{"id":"complete"}\n'


@pytest.mark.parametrize("data", [
    b'{"id":"first"}\nnot-json\n{"id":"last"}\n',
    b'{"id":"first"}\nnot-json\n',
    b'[]\n{"id":"last"}\n',
])
def test_read_journal_rejects_interior_or_terminated_corruption(runner, tmp_path, data):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        runner.read_journal(path)
    assert path.read_bytes() == data


def _write_rows_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_validates_thesis_binary_track_semantics(runner, canonical_row, tmp_path):
    rows = [canonical_row, {**canonical_row, "id": "second", "track": "Track_B", "ground_truth_label": 1}]
    path = tmp_path / "rows.csv"
    _write_rows_csv(path, rows)
    loaded = runner.load_rows(path)
    assert [r["ground_truth_label"] for r in loaded] == [0, 1]
    assert all(type(r["ground_truth_label"]) is int for r in loaded)
    assert loaded[0]["candidate_answer"] == canonical_row["candidate_answer"]


@pytest.mark.parametrize("updates", [
    {"ground_truth_label": "0.0"}, {"ground_truth_label": "2"},
    {"ground_truth_label": ""}, {"ground_truth_label": "True"},
    {"track": "Track_B"}, {"track": "unknown"}, {"id": ""}, {"task": "unknown"},
])
def test_load_rows_rejects_nonbinary_or_inconsistent_metadata(runner, canonical_row, tmp_path, updates):
    path = tmp_path / "rows.csv"
    _write_rows_csv(path, [{**canonical_row, **updates}])
    with pytest.raises(ValueError):
        runner.load_rows(path)


def test_load_rows_rejects_duplicate_ids(runner, canonical_row, tmp_path):
    path = tmp_path / "rows.csv"
    _write_rows_csv(path, [canonical_row, canonical_row])
    with pytest.raises(ValueError, match="duplicate"):
        runner.load_rows(path)


def test_summary_complete_counts_failures_and_partial_withholds_final_bhs(runner):
    rows = [{"id": f"row-{i}", "ground_truth_label": label} for i, label in enumerate([0, 1, 0, 1])]
    records = [
        {"id": row["id"], "model": "complete", "prediction": prediction, "latency_seconds": .25,
         "p_hall": .1 if i == 0 else None,
         "decision_source": "decoded_label_fallback" if i == 1 else "two_label_score_fixed_threshold"}
        for i, (row, prediction) in enumerate(zip(rows, [0, 1, None, None]))
    ]
    partial = [{**r, "model": "partial"} for r in records[:2]]
    stats = {s["model"]: s for s in runner.summary(rows, records + partial)}
    complete = stats["complete"]
    assert complete["complete"] is True
    assert complete["attempted"] == complete["expected"] == 4
    assert complete["BHS"] == complete["op"] == 50
    assert complete["coverage"] == .5
    assert complete["two_label_scores"] == complete["decoded_fallbacks"] == 1
    assert complete["latency_seconds"] == 1
    assert "partial_recorded_rows_BHS" not in complete
    assert stats["partial"]["complete"] is False
    assert stats["partial"]["BHS"] is None
    assert stats["partial"]["partial_recorded_rows_BHS"] == 0
    assert stats["partial"]["attempted"] == 2
    assert stats["partial"]["expected"] == 4


@pytest.fixture(scope="module")
def v2_features():
    return importlib.import_module("pipeline_v2_features")


def test_feature_vector_synthetic_order_and_missing_indicators(v2_features):
    semantic = {name: i / 20 for i, name in enumerate(v2_features.SEMANTIC_FEATURES)}
    surface = {name: i / 20 for i, name in enumerate(v2_features.SURFACE_FEATURES)}
    record = {"semantic_features": semantic, "surface_features": surface}
    vector = v2_features.feature_vector(record, {"p_hall": .8})
    assert len(vector) == 30
    assert vector[:16] == list(semantic.values())
    assert vector[16:20] == [.8, None, None, None]
    assert vector[20:] == list(surface.values())
    assert record["missing_flags"]["jsd"] is True
    assert record["missing_flags"]["p_hall"] is False


def test_no_context_enrichment_never_loads_neural_models(v2_features, v2_math, tmp_path, monkeypatch):
    stages = v2_features.NeuralStages(tmp_path)
    resolve, load = Mock(side_effect=AssertionError("must not resolve models")), Mock(side_effect=AssertionError("must not load models"))
    monkeypatch.setattr(stages, "_resolve", resolve)
    monkeypatch.setattr(stages, "_load", load)
    monkeypatch.setattr(stages, "_cleanup", Mock())
    record = {"id": "synthetic", "task": "bn_qa", "question": "কত?", "context": "",
              "candidate_answer": "-৩", "claims": [{"claim": "মান -৩", "positive_query": "মান -৩",
              "negative_query": "মান -৩ নয়", "negative_claim": "মান -৩ নয়"}]}
    original = copy.deepcopy(record)
    result = stages.enrich([record])
    assert result[0] is record
    resolve.assert_not_called()
    load.assert_not_called()
    for key, value in original.items():
        if key != "claims":
            assert record[key] == value
    assert record["evidence_status"] == "no_evidence"
    assert record["nli_coverage"]["complete"] is False
    assert record["route_inputs"]["entails"] == []
    assert v2_math.route(**record["route_inputs"]) == "full-verification"
    assert record["semantic_features"]["claim_entailment_mean"] is None
    assert record["surface_features"]["numeric_match_ratio"] is None
    assert record["claims"][0]["evidence"] == []
    assert len(v2_features.feature_vector(record, {})) == 30


def test_synthetic_feature_aggregation_retains_incomplete_claim_guard(v2_features, v2_math, tmp_path):
    stages = v2_features.NeuralStages(tmp_path)
    record = {"task": "bn_qa", "context": "প্রমাণ", "question": "কী?", "candidate_answer": "উত্তর",
              "claims": [{"evidence": [], "nli_maxima": {"entailment": .99, "contradiction": .01, "neutral": 0}},
                         {"evidence": [], "nli_maxima": {"entailment": None, "contradiction": None, "neutral": None}}]}
    stages._features(record)
    assert record["nli_coverage"] == {"complete": False, "claims_total": 2, "claims_scored": 1}
    assert record["semantic_features"]["claim_entailment_mean"] is None
    assert record["nli_extrema"]["smin_ent"] is None
    assert v2_math.route(**record["route_inputs"]) == "full-verification"


def test_grouped_meta_nested_missing_feature_fit_and_conformal_quantile(v2_math):
    pytest.importorskip("sklearn")
    from inspect import signature
    from sklearn.impute import SimpleImputer

    if "keep_empty_features" not in signature(SimpleImputer).parameters:
        pytest.skip("Missing-feature fitting requires scikit-learn >= 1.2")
    rng = np.random.default_rng(42)
    train_y = np.tile([0, 1], 16)
    groups = np.repeat(np.arange(16), 2)
    train_x = rng.normal(size=(32, 30))
    train_x[:, 0] = np.nan
    train_x[::3, 1] = np.nan
    train_x[:, 2] = train_y
    train_texts = [f"synthetic source {i // 2} {'faithful' if y == 0 else 'hallucinated'}" for i, y in enumerate(train_y)]
    dev_x, cal_x = rng.normal(size=(4, 30)), rng.normal(size=(4, 30))
    dev_x[:, 0] = cal_x[:, 0] = np.nan
    dev_y, cal_y = np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])
    dev_texts = [f"development unique {i}" for i in range(4)]
    cal_texts = [f"calibration unique {i}" for i in range(4)]
    model = v2_math.GroupedMetaModel(n_splits=2, random_state=42)
    assert model.fit(train_x, train_texts, train_y, groups,
                     dev_x, dev_texts, dev_y, ["open", "open", "closed", "closed"],
                     cal_x, cal_texts, cal_y, alpha=.4) is model
    assert len(model.feature_order_) == 31
    assert len(model.internal_feature_order_) == 61
    assert np.isfinite(model.oof_probabilities_).all()
    heldout_counts = np.zeros(32, dtype=int)
    for fold in model.fold_manifest_:
        train, holdout = fold["train_indices"], fold["holdout_indices"]
        assert not set(groups[train]) & set(groups[holdout])
        heldout_counts[holdout] += 1
        assert "numeric_00" in fold["all_missing_features"]
        assert fold["imputation_statistics"]["numeric_01"] == pytest.approx(np.nanmedian(train_x[train, 1]))
        inner_counts = {i: 0 for i in train}
        for inner in fold["inner_folds"]:
            inner_train, inner_holdout = inner["train_indices"], inner["holdout_indices"]
            assert set(inner_train) | set(inner_holdout) == set(train)
            assert not set(groups[inner_train]) & set(groups[inner_holdout])
            for i in inner_holdout:
                inner_counts[i] += 1
        assert set(inner_counts.values()) == {1}
    assert np.all(heldout_counts == 1)
    cal_prob = model.predict_proba(cal_x, cal_texts)
    np.testing.assert_allclose(cal_prob.sum(axis=1), 1)
    scores = 1 - cal_prob[np.arange(4), cal_y]
    np.testing.assert_allclose(model.calibration_scores_, scores)
    assert model.q_ == pytest.approx(v2_math.conformal_quantile(scores, alpha=.4))
    evaluation = cal_x.copy()
    evaluation[0, 3] = np.nan  # A feature can first become unavailable at inference.
    probs = model.predict_proba(evaluation, cal_texts)
    assert probs.shape == (4, 2)
    assert np.isfinite(probs).all()
    expected = [v2_math.prediction_set(p, model.q_) for p in probs[:, 1]]
    assert model.predict_sets(evaluation, cal_texts) == expected
    predictions = model.predict(evaluation, cal_texts, ["open", "closed", "open", "closed"])
    assert set(predictions) <= {0, 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), *sys.argv[1:]]))

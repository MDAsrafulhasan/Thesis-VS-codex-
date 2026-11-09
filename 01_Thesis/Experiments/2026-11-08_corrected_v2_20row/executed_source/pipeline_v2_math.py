"""Offline mathematical primitives for Update_formulation.md (corrected v2).

Labels/probabilities: 0 = faithful (Track A), 1 = hallucinated (Track B).
This module performs no I/O, model downloads, API calls, benchmark lookup, or
training at import time. sklearn is imported lazily only when fit is requested.
No reference implementation or answer bank is imported.

Separation contract
-------------------
Keep raw scientific text outside canonical keys. Supply independently approved
training data, a source-disjoint development split for threshold selection, and
a further source-disjoint calibration split for conformal scores. The compact
fit API intentionally receives groups ONLY for training: the CALLER MUST audit
train/dev/cal/test source and derivative disjointness. Hash13 is not a complete
lineage audit. Evaluation labels must never enter fitting or threshold tuning.

Feature contract
----------------
Every matrix has exactly 30 numeric columns, in the SAME caller-recorded order
at fit and inference. None/NaN mean unavailable; positive/negative infinity is
invalid. Suggested blocks are 16 semantic/NLI, 4 judge/JSD, and 10 surface
features, but this module does not invent their meaning or extract them. Logical
meta column 30 (zero based) is the scalar P(hallucinated) lexical OOF score,
giving 31 columns recorded in feature_order_. Within each outer training fold,
a median SimpleImputer(keep_empty_features=True) retains all logical columns;
a wholly missing training column gets deterministic 0 plus its explicit missing
indicator, NOT a valid zero neural score. All 30 numeric missingness indicators
are appended, even for columns complete during training, so new missingness is
represented at inference. The scaler/classifier thus receive 61 columns, whose
names are recorded in internal_feature_order_. Optional feature_names records
the 30 input names; array inputs cannot detect a semantic column permutation.

Stage 5 uses nested grouped cross-fitting. Within every outer training fold,
inner training-only TF-IDF/logistic models produce lexical OOF scores. An outer
training-only median imputer, StandardScaler and logistic meta model fit the
31 logical features plus missingness indicators. Outer holdouts never participate
in their predictors' vocabulary, lexical fit, imputer, scaler, or meta fit. New rows use the mean of inner lexical scores within each
outer model, then the mean of outer meta probabilities. dev changes only point
thresholds; cal changes only the conformal quantile. OOF scores are diagnostic,
not a substitute for a final independent performance estimate. This optional
trained profile makes no empirical accuracy or calibration claims.

Caller-owned persistence may pickle a fitted instance in a TRUSTED local
context, with this module importable under the same name and compatible numpy /
sklearn versions. Never unpickle untrusted artifacts. Record versions, source
hashes, feature manifest, seeds, groups, and the split/threshold protocol outside
this module. Conformal coverage requires exchangeability and is marginal, not
BHS risk control. There are no benchmark-memory overrides.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "canonicalize", "dual_views", "hash13", "sigmoid_score", "js_divergence",
    "smooth_extrema", "route", "polarity_rrf", "operational_bhs",
    "conformal_quantile", "prediction_set", "cad_logits", "GroupedMetaModel",
]


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _probability(value, name):
    result = _finite_scalar(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _vector(values, name, probabilities=False):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if probabilities and np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def canonicalize(text) -> str:
    """NFKC then collapse Unicode whitespace; retain punctuation and signs.

    No lowercasing, digit translation, sign stripping, or zero-width deletion.
    NFKC itself is not lossless (e.g. superscripts); always retain raw text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string; retain missingness separately")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def dual_views(text, reviewed_map=None, protected_terms=()) -> dict:
    """Return native canonical text and a selectively mapped code view.

    reviewed_map is an explicit caller-reviewed {whole_token: replacement}
    transducer into the desired code view; direction/content are caller-owned.
    Tokens are Unicode letter/mark/number/underscore runs (marks keep Bengali
    vowel signs attached). Protected terms win over mappings, case-sensitively.
    Unknown tokens, punctuation and signs are identity fallbacks. No built-in
    Bengali romanizer or guessed vocabulary exists; without a map views agree.
    Multiword keys/protected phrases are rejected, rather than partly applied.
    """
    native = canonicalize(text)

    def tokens(value):
        return re.findall(r"\s+|[^\s]+", value)

    def is_word(char):
        return char == "_" or unicodedata.category(char)[0] in "LMN"

    def checked_token(value, name):
        value = canonicalize(value)
        if not value or len(tokens(value)) != 1 or not all(map(is_word, value)):
            raise ValueError(f"{name} must be one nonempty letter/mark/number token")
        return value

    if isinstance(protected_terms, str):
        raise TypeError("protected_terms must be a collection of tokens")
    protected = {checked_token(term, "protected term") for term in protected_terms}
    if reviewed_map is None:
        reviewed_map = {}
    if not isinstance(reviewed_map, Mapping):
        raise TypeError("reviewed_map must be a mapping or None")
    replacements = {}
    for key, value in reviewed_map.items():
        key = checked_token(key, "mapping key")
        replacement = canonicalize(value)
        if not replacement:
            raise ValueError("reviewed replacements must not delete tokens")
        if key in replacements and replacements[key] != replacement:
            raise ValueError("mapping keys conflict after canonicalization")
        replacements[key] = replacement
    chunks, word = [], []

    def flush():
        if word:
            token = "".join(word)
            chunks.append(token if token in protected else replacements.get(token, token))
            word.clear()

    for char in native:
        if is_word(char):
            word.append(char)
        else:
            flush()
            chunks.append(char)
    flush()
    return {"native": native, "code": "".join(chunks)}


def hash13(text) -> set:
    """SHA256 hex digests of canonical whitespace-token 13-grams.

    Protocol: canonicalize, str.split, single U+0020 separator, UTF-8, SHA256
    lowercase hex. Fewer than 13 tokens returns an empty set.
    """
    tokens = canonicalize(text).split()
    return {
        hashlib.sha256(" ".join(tokens[i:i + 13]).encode("utf-8")).hexdigest()
        for i in range(len(tokens) - 12)
    }


def sigmoid_score(lfaith, lhall) -> float:
    """Stable two-label P(hallucinated), not a full-vocabulary probability."""
    lf = _finite_scalar(lfaith, "lfaith")
    lh = _finite_scalar(lhall, "lhall")
    # Subtract smaller from larger: overflow to +inf is harmless to exp(-gap).
    gap = abs(lh - lf)
    tail = math.exp(-gap)
    return float(1.0 / (1.0 + tail) if lh >= lf else tail / (1.0 + tail))


def js_divergence(p, q) -> float:
    """Binary JSD in nats for scalar P(class 1) scores, including 0 and 1."""
    p = _probability(p, "p")
    q = _probability(q, "q")
    # Log differences, not ratios, also handle subnormal positive inputs.
    total = 0.0
    for left, right in ((p, q), (1.0 - p, 1.0 - q)):
        middle = left / 2.0 + right / 2.0
        if middle == 0.0:
            # Both zero, or an underflowing average of subnormals. Scale first.
            if left or right:
                log_middle = math.log(max(left, right)) + math.log(
                    (left / max(left, right) + right / max(left, right)) / 2.0
                )
            else:
                continue
        else:
            log_middle = math.log(middle)
        if left:
            total += 0.5 * left * (math.log(left) - log_middle)
        if right:
            total += 0.5 * right * (math.log(right) - log_middle)
    return float(min(math.log(2.0), max(0.0, total)))


def smooth_extrema(entails, contrads, tau=.15, weights=None) -> dict:
    """Weighted normalized smooth extrema plus hard guards; empty -> None.

    Scores must be paired finite probabilities. Weights must be nonnegative,
    have positive sum and match scores; they are normalized here. Hard extrema
    include zero-weight claims so weights cannot disable routing safeguards.
    """
    tau = _finite_scalar(tau, "tau")
    if tau <= 0:
        raise ValueError("tau must be positive")
    e = _vector(entails, "entails", probabilities=True)
    c = _vector(contrads, "contrads", probabilities=True)
    if e.shape != c.shape:
        raise ValueError("entails and contrads must have the same length")
    if weights is not None:
        w = _vector(weights, "weights")
        if w.shape != e.shape or np.any(w < 0):
            raise ValueError("weights must match scores and be nonnegative")
    else:
        w = np.ones(e.size, dtype=float)
    if not e.size:
        return dict(smin_ent=None, smax_con=None, emin=None, cmax=None)
    if not np.any(w > 0):
        raise ValueError("weights must have a positive sum")
    w = w / np.max(w)
    w = w / np.sum(w)
    active = w > 0
    # Center in score space before dividing by tau, avoiding +inf extrema.
    ebase, cbase = float(np.min(e[active])), float(np.max(c[active]))
    def log_average(scaled):
        # log1p/expm1 preserves the weighted-mean limit at very large tau;
        # ordinary log-sum-exp preserves tiny nonzero base weights at small tau.
        if np.min(scaled) > -.5:
            return math.log1p(float(np.sum(w[active] * np.expm1(scaled))))
        return float(np.log(np.sum(w[active] * np.exp(scaled))))

    with np.errstate(over="ignore", under="ignore"):
        emin = ebase - tau * log_average(-(e[active] - ebase) / tau)
        cmax = cbase + tau * log_average((c[active] - cbase) / tau)
    emin = np.clip(emin, np.min(e[active]), np.max(e[active]))
    cmax = np.clip(cmax, np.min(c[active]), np.max(c[active]))
    return dict(smin_ent=float(emin), smax_con=float(cmax),
                emin=float(np.min(e)), cmax=float(np.max(c)))


def route(has_context, is_math, entails, contrads, tau=.15,
          gamma_e=.85, gamma_c=.85, lambda_c=.15) -> str:
    """Context-only, non-math tri-state route; undefined/conflict -> full.

    Invalid scores raise rather than silently producing a confident verdict.
    The E_min guard blocks high-entailment/high-contradiction fast rejection.
    """
    if not isinstance(has_context, (bool, np.bool_)) or not isinstance(is_math, (bool, np.bool_)):
        raise ValueError("has_context and is_math must be booleans")
    ge = _probability(gamma_e, "gamma_e")
    gc = _probability(gamma_c, "gamma_c")
    lc = _probability(lambda_c, "lambda_c")
    stats = smooth_extrema(entails, contrads, tau=tau)
    if not has_context or is_math or stats["emin"] is None:
        return "full-verification"
    faithful = stats["smin_ent"] >= ge and stats["cmax"] <= lc
    hallucinated = stats["smax_con"] >= gc and stats["emin"] < ge
    if faithful == hallucinated:
        return "full-verification"
    return "fast-faithful" if faithful else "fast-hallucinated"


def polarity_rrf(rankings, k=60) -> dict:
    """Separate polarity -> document -> RRF score maps; ranks start at 1.

    Duplicate IDs within a retriever list count only at their first rank;
    subsequent unique documents retain their original rank positions.
    """
    k = _finite_scalar(k, "k")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if not isinstance(rankings, Mapping):
        raise TypeError("rankings must map polarity to ranked-list collections")
    result = {}
    for polarity, lists in rankings.items():
        scores = {}
        for ranking in lists:
            if isinstance(ranking, (str, bytes)):
                raise ValueError("each ranking must be a sequence of document IDs")
            seen = set()
            for rank, doc_id in enumerate(ranking, 1):
                try:
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                except TypeError as exc:
                    raise ValueError("document IDs must be hashable") from exc
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        result[polarity] = scores
    return result


def _labels(values, name, allow_none=False):
    result = []
    for value in values:
        if value is None and allow_none:
            result.append(None)
        elif (isinstance(value, (bool, np.bool_))
              or not isinstance(value, numbers.Real)
              or not math.isfinite(float(value)) or value not in (0, 1)):
            raise ValueError(f"{name} must contain only numeric 0/1"
                             + (" or None" if allow_none else ""))
        else:
            result.append(int(value))
    return result


def operational_bhs(labels, predictions, w_a=.5, w_b=.5) -> dict:
    """JSON-safe weighted error percentages (lower better).

    None predictions are operational errors, not labels or valid verdicts.
    wrong_a/b include failures; wrong_valid_a/b exclude them. fpr/fnr use
    operational denominators. Both tracks are required even at zero weight.
    valid_only uses valid denominators and is None if either has no valid row.
    """
    wa, wb = _finite_scalar(w_a, "w_a"), _finite_scalar(w_b, "w_b")
    if wa < 0 or wb < 0 or not math.isclose(wa + wb, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("BHS weights must be nonnegative and sum to 1")
    y = _labels(labels, "labels")
    pred = _labels(predictions, "predictions", allow_none=True)
    if len(y) != len(pred):
        raise ValueError("labels and predictions must have equal lengths")
    n, valid, wrong, wrong_valid = [0, 0], [0, 0], [0, 0], [0, 0]
    for label, prediction in zip(y, pred):
        n[label] += 1
        if prediction is not None:
            valid[label] += 1
            wrong_valid[label] += int(prediction != label)
        wrong[label] += int(prediction != label)
    rates = [wrong[i] / n[i] if n[i] else None for i in (0, 1)]
    valid_rates = [wrong_valid[i] / valid[i] if valid[i] else None for i in (0, 1)]

    def aggregate(pair):
        return None if None in pair else float(100 * (wa * pair[0] + wb * pair[1]))

    return dict(n_a=n[0], n_b=n[1], wrong_a=wrong[0], wrong_b=wrong[1],
                valid_a=valid[0], valid_b=valid[1], n_valid=sum(valid),
                wrong_valid_a=wrong_valid[0], wrong_valid_b=wrong_valid[1],
                failures_a=n[0] - valid[0], failures_b=n[1] - valid[1],
                coverage=sum(valid) / len(y) if y else None,
                fpr=rates[0], fnr=rates[1],
                valid_fpr=valid_rates[0], valid_fnr=valid_rates[1],
                BHS=aggregate(rates), op=aggregate(rates),
                valid_only=aggregate(valid_rates), w_a=wa, w_b=wb)


def conformal_quantile(scores, alpha=.05) -> float:
    """Finite-sample ceil((n+1)*(1-alpha)) order statistic, no interpolation.

    Scores must be finite; empty calibration or an index above n returns +inf,
    yielding both classes, rather than spuriously claiming a finite guarantee.
    """
    alpha = _finite_scalar(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    values = _vector(scores, "scores")
    index = math.ceil((len(values) + 1) * (1.0 - alpha))
    if index > len(values):
        return math.inf
    return float(np.partition(values, index - 1)[index - 1])


def prediction_set(prob, q) -> list:
    """Classes satisfying 1-p_y <= q; prob is scalar P(hallucinated)."""
    p = _probability(prob, "prob")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, numbers.Real):
        raise ValueError("q must be a real number or positive infinity")
    q = float(q)
    if math.isnan(q) or q == -math.inf:
        raise ValueError("q must be finite or positive infinity")
    return [label for label, score in ((0, p), (1, 1.0 - p)) if score <= q]


def cad_logits(context_logits, prior_logits, alpha=.5, temp=1) -> np.ndarray:
    """Normalized CAD probabilities, softmax on the last vocabulary axis.

    Arrays must align in shape/vocabulary/step; the caller verifies semantic
    alignment. Long-double intermediates reduce overflow for finite float64
    logits; reject unrepresentable intermediate values instead of returning NaN.
    """
    alpha = _finite_scalar(alpha, "alpha")
    temp = _finite_scalar(temp, "temp")
    if alpha < 0 or temp <= 0:
        raise ValueError("alpha must be nonnegative and temp positive")
    try:
        context = np.asarray(context_logits, dtype=np.longdouble)
        prior = np.asarray(prior_logits, dtype=np.longdouble)
    except (TypeError, ValueError) as exc:
        raise ValueError("logits must be finite numeric arrays") from exc
    if (context.shape != prior.shape or context.ndim < 1 or not context.size
            or not np.all(np.isfinite(context)) or not np.all(np.isfinite(prior))):
        raise ValueError("logits must have identical nonempty shapes and finite values")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        logits = context + np.longdouble(alpha) * (context - prior)
        if not np.all(np.isfinite(logits)):
            raise ValueError("CAD logit combination exceeds numeric range")
        # Divide only nonpositive centered logits; -inf exponent -> safe zero.
        centered = (logits - np.max(logits, axis=-1, keepdims=True)) / np.longdouble(temp)
        probs = np.exp(centered)
        probs /= np.sum(probs, axis=-1, keepdims=True)
    result = np.asarray(probs, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("CAD normalization produced nonfinite probabilities")
    return result


@dataclass
class _LexicalModel:
    vectorizer: object
    classifier: object

    def predict(self, texts):
        return self.classifier.predict_proba(self.vectorizer.transform(texts))[:, 1]


@dataclass
class _MetaFold:
    lexical_models: list
    imputer: object
    scaler: object
    classifier: object

    def transform(self, design):
        """Retain 31 logical values, then append all 30 numeric missing flags."""
        missing = np.isnan(design[:, :30]).astype(float)
        return np.column_stack((self.imputer.transform(design), missing))

    def predict(self, features, texts):
        lexical = np.mean([model.predict(texts) for model in self.lexical_models], axis=0)
        design = self.transform(np.column_stack((features, lexical)))
        return self.classifier.predict_proba(self.scaler.transform(design))[:, 1]


class GroupedMetaModel:
    """Optional nested grouped lexical/meta logistic ensemble; no automatic fit.

    fit(train_features, train_texts, train_labels, train_groups,
        dev_features, dev_texts, dev_labels, dev_regimes,
        cal_features, cal_texts, cal_labels, alpha=.05) -> self
    predict_proba(features, texts) -> (n, 2) ndarray in class order [0, 1]
    predict(features, texts, regimes) -> (n,) integer point predictions
    predict_sets(features, texts) -> list[list[int]] conformal class sets

    Regimes are exactly 'open' or 'closed'. Both regimes and both labels must
    occur globally on dev, but a regime can lack a label: joint global BHS
    denominators still apply. Calibration may have one class, but is nonempty.
    Outer and inner fold counts adapt down to 2; infeasible nested class/group
    support raises ValueError, never falls back to row-wise CV or in-sample lex.
    All logistic fits use fixed C, L2, lbfgs, and no class-weight tuning.

    None/NaN numeric features are imputed by each outer-training-fold median.
    SimpleImputer(keep_empty_features=True), requiring sklearn >= 1.2, retains
    wholly missing training columns with deterministic 0; explicit indicators
    distinguish unavailable values from observed zeros. All 30 numeric flags
    are appended, including columns first missing at inference. feature_order_
    records the 31 logical inputs; internal_feature_order_ records 61 scaler/LR
    inputs. Each fold manifest records all-missing names and imputation medians.
    Fitted imputers travel with models_ and transform holdout/dev/cal/eval alike.

    thresholds_ optimizes balanced joint dev BHS with >= comparisons. Candidate
    thresholds are each regime's distinct observed probabilities plus 0 and the
    next representable float above 1 (all-negative). Since the objective is an
    additive sum of regime error counts divided by GLOBAL dev track sizes, its
    Cartesian argmin is computed exactly by independent contribution minima,
    NOT regime-local F1/BHS. Exact integer costs avoid floating tie ambiguity;
    ties choose closest to 0.5, then the lower threshold, in each regime.
    Calibration labels are used only for q_ = quantile(1-p_true).
    """

    def __init__(self, n_splits=5, random_state=42, C=1.0, max_iter=2000,
                 feature_names=None):
        if isinstance(n_splits, bool) or not isinstance(n_splits, numbers.Integral) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        if isinstance(max_iter, bool) or not isinstance(max_iter, numbers.Integral) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if (isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral)
                or not 0 <= random_state < 2 ** 32):
            raise ValueError("random_state must be a nonnegative 32-bit integer")
        self.C = _finite_scalar(C, "C")
        if self.C <= 0:
            raise ValueError("C must be positive")
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        if feature_names is None:
            names = tuple(f"numeric_{i:02d}" for i in range(30))
        else:
            if isinstance(feature_names, str):
                raise ValueError("feature_names must contain 30 distinct names")
            names = tuple(feature_names)
        if (len(names) != 30 or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != 30 or "lexical_probability" in names):
            raise ValueError("feature_names must contain 30 distinct nonempty numeric names")
        self.feature_names = names

    @staticmethod
    def _data(features, texts, name, allow_empty=False):
        try:
            features = np.asarray(features, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} features must be a numeric (n, 30) matrix; None/NaN are allowed") from exc
        if (features.ndim != 2 or features.shape[1] != 30
                or np.any(np.isinf(features))):
            raise ValueError(f"{name} features must have exactly 30 numeric columns and no infinity; "
                             "use None/NaN for unavailable scores")
        if isinstance(texts, str):
            raise ValueError(f"{name} texts must be a sequence, not a single string")
        texts = list(texts)
        if len(texts) != len(features) or any(not isinstance(text, str) for text in texts):
            raise ValueError(f"{name} requires one string per feature row")
        if not allow_empty and len(texts) == 0:
            raise ValueError(f"{name} must not be empty")
        return features, texts

    @staticmethod
    def _targets(labels, n, name, both=False):
        y = np.asarray(_labels(labels, name), dtype=int)
        if len(y) != n or (both and len(np.unique(y)) != 2):
            raise ValueError(f"{name} must match rows" + (" and contain both classes" if both else ""))
        return y

    @staticmethod
    def _regimes(regimes, n, require_both=False):
        if isinstance(regimes, str):
            raise ValueError("regimes must contain one 'open'/'closed' value per row")
        values = list(regimes)
        if len(values) != n or any(not isinstance(r, str) or r not in ("open", "closed") for r in values):
            raise ValueError("regimes must contain one 'open'/'closed' value per row")
        if require_both and set(values) != {"open", "closed"}:
            raise ValueError("dev must cover both open and closed regimes for threshold selection")
        return np.asarray(values)

    @staticmethod
    def _group_codes(groups, n):
        if isinstance(groups, (str, bytes)):
            raise ValueError("train_groups must contain one source-lineage ID per row")
        values = list(groups)
        if len(values) != n:
            raise ValueError("train_groups must match training rows")
        mapping, codes = {}, []
        for value in values:
            if (value is None or (isinstance(value, str) and not value.strip())
                    or (isinstance(value, numbers.Real) and not math.isfinite(float(value)))):
                raise ValueError("train_groups must not contain missing/empty IDs")
            try:
                if value not in mapping:
                    mapping[value] = len(mapping)
                codes.append(mapping[value])
            except TypeError as exc:
                raise ValueError("source-lineage group IDs must be hashable") from exc
        return np.asarray(codes, dtype=int)

    def _splits(self, y, groups, nested=False):
        from sklearn.model_selection import StratifiedGroupKFold

        cap = min(self.n_splits, len(np.unique(groups)), int(np.bincount(y, minlength=2).min()))
        for count in range(cap, 1, -1):
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=self.random_state)
            splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
            feasible = True
            for train, heldout in splits:
                if (not len(heldout) or len(np.unique(y[train])) != 2
                        or set(groups[train]) & set(groups[heldout])):
                    feasible = False
                    break
                if nested:
                    try:
                        self._splits(y[train], groups[train])
                    except ValueError:
                        feasible = False
                        break
            if feasible:
                return splits
        raise ValueError("Insufficient source-group/class support for "
                         + ("nested " if nested else "")
                         + "grouped OOF fitting; supply more independent labeled groups")

    def _logistic(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=self.C, solver="lbfgs", penalty="l2",
                                  max_iter=self.max_iter, random_state=self.random_state)

    def _fit_lexical(self, texts, y):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=4000, analyzer="char_wb",
                                     ngram_range=(2, 5), sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as exc:
            raise ValueError("A training-only lexical fold has no usable character vocabulary") from exc
        classifier = self._logistic()
        classifier.fit(matrix, y)
        return _LexicalModel(vectorizer, classifier)

    @staticmethod
    def _thresholds(probabilities, labels, regimes):
        na, nb = int(np.sum(labels == 0)), int(np.sum(labels == 1))
        thresholds = {}
        for regime in ("open", "closed"):
            mask = regimes == regime
            p, y = probabilities[mask], labels[mask]
            # Sort once; moving a tied probability block to negative changes
            # only its track error counts. Costs are proportional to joint BHS.
            order = np.argsort(p, kind="stable")
            p, y = p[order], y[order]
            fp, fn = int(np.sum(y == 0)), 0
            candidates = [(fp * nb + fn * na, 0.0)]
            i = 0
            while i < len(p):
                value = float(p[i])
                candidates.append((fp * nb + fn * na, value))
                end = i + 1
                while end < len(p) and p[end] == value:
                    end += 1
                fp -= int(np.sum(y[i:end] == 0))
                fn += int(np.sum(y[i:end] == 1))
                i = end
            candidates.append((fp * nb + fn * na, float(np.nextafter(1.0, math.inf))))
            thresholds[regime] = min(candidates, key=lambda pair: (pair[0], abs(pair[1] - .5), pair[1]))[1]
        return thresholds

    def fit(self, train_features, train_texts, train_labels, train_groups,
            dev_features, dev_texts, dev_labels, dev_regimes,
            cal_features, cal_texts, cal_labels, alpha=.05):
        """Fit without evaluation data; caller enforces cross-split disjointness.

        Local fitting is transactional: a failure does not replace a previously
        fitted state. No dev/cal row fits a vectorizer, imputer, scaler, or
        classifier. Cross-split source/derivative disjointness is caller-owned.
        """
        from inspect import signature

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if "keep_empty_features" not in signature(SimpleImputer).parameters:
            raise RuntimeError("GroupedMetaModel requires scikit-learn >= 1.2 "
                               "with SimpleImputer(keep_empty_features=True)")
        logical_order = self.feature_names + ("lexical_probability",)
        internal_order = logical_order + tuple(f"missing::{name}" for name in self.feature_names)
        train_x, train_t = self._data(train_features, train_texts, "train")
        dev_x, dev_t = self._data(dev_features, dev_texts, "dev")
        cal_x, cal_t = self._data(cal_features, cal_texts, "cal")
        train_y = self._targets(train_labels, len(train_t), "train_labels", both=True)
        dev_y = self._targets(dev_labels, len(dev_t), "dev_labels", both=True)
        cal_y = self._targets(cal_labels, len(cal_t), "cal_labels")
        regimes = self._regimes(dev_regimes, len(dev_t), require_both=True)
        groups = self._group_codes(train_groups, len(train_t))
        _finite_scalar(alpha, "alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        outer_splits = self._splits(train_y, groups, nested=True)
        models, manifest = [], []
        oof = np.full(len(train_y), np.nan)
        for outer_train, outer_holdout in outer_splits:
            x, y, g = train_x[outer_train], train_y[outer_train], groups[outer_train]
            texts = [train_t[i] for i in outer_train]
            inner_splits = self._splits(y, g)
            lexical_oof = np.full(len(y), np.nan)
            lexical_models, inner_manifest = [], []
            for inner_train, inner_holdout in inner_splits:
                lexical = self._fit_lexical([texts[i] for i in inner_train], y[inner_train])
                lexical_oof[inner_holdout] = lexical.predict([texts[i] for i in inner_holdout])
                lexical_models.append(lexical)
                inner_manifest.append({"train_indices": outer_train[inner_train].tolist(),
                                       "holdout_indices": outer_train[inner_holdout].tolist()})
            if not np.all(np.isfinite(lexical_oof)):
                raise ValueError("Inner grouped OOF did not cover every outer-training row")
            design = np.column_stack((x, lexical_oof))
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            imputer.fit(design)
            scaler = StandardScaler()
            classifier = self._logistic()
            model = _MetaFold(lexical_models, imputer, scaler, classifier)
            scaled = scaler.fit_transform(model.transform(design))
            classifier.fit(scaled, y)
            oof[outer_holdout] = model.predict(train_x[outer_holdout], [train_t[i] for i in outer_holdout])
            models.append(model)
            all_missing = np.all(np.isnan(x), axis=0)
            manifest.append({"train_indices": outer_train.tolist(),
                             "holdout_indices": outer_holdout.tolist(), "inner_folds": inner_manifest,
                             "all_missing_features": [name for name, missing in zip(self.feature_names, all_missing)
                                                      if missing],
                             "imputation_statistics": dict(zip(logical_order, imputer.statistics_.tolist()))})
        if not np.all(np.isfinite(oof)):
            raise ValueError("Outer grouped OOF did not cover every training row")
        dev_p = np.mean([model.predict(dev_x, dev_t) for model in models], axis=0)
        thresholds = self._thresholds(dev_p, dev_y, regimes)
        dev_predictions = np.asarray([int(p >= thresholds[r]) for p, r in zip(dev_p, regimes)])
        cal_p = np.mean([model.predict(cal_x, cal_t) for model in models], axis=0)
        scores = np.where(cal_y == 1, 1.0 - cal_p, cal_p)
        q = conformal_quantile(scores, alpha)
        self.models_ = models
        self.thresholds_ = thresholds
        self.q_ = q
        self.alpha_ = float(alpha)
        self.feature_order_ = logical_order
        self.internal_feature_order_ = internal_order
        self.missing_indicator_features_ = self.feature_names
        self.n_features_in_ = 30
        self.classes_ = np.asarray([0, 1])
        self.oof_probabilities_ = oof
        self.fold_manifest_ = manifest
        self.train_group_codes_ = groups.copy()
        self.dev_bhs_ = operational_bhs(dev_y, dev_predictions)
        self.calibration_scores_ = scores
        return self

    def _check_fitted(self):
        if not hasattr(self, "models_"):
            raise RuntimeError("GroupedMetaModel must be fitted before prediction")

    def predict_proba(self, features, texts) -> np.ndarray:
        """Class probabilities [faithful, hallucinated]; preserves row order."""
        self._check_fitted()
        x, t = self._data(features, texts, "prediction", allow_empty=True)
        if not len(t):
            return np.empty((0, 2), dtype=float)
        p = np.mean([model.predict(x, t) for model in self.models_], axis=0)
        return np.column_stack((1.0 - p, p))

    def predict(self, features, texts, regimes) -> np.ndarray:
        """Dev-threshold point verdicts; distinct from conformal class sets."""
        probabilities = self.predict_proba(features, texts)[:, 1]
        regimes = self._regimes(regimes, len(probabilities))
        return np.asarray([int(p >= self.thresholds_[r]) for p, r in zip(probabilities, regimes)], dtype=int)

    def predict_sets(self, features, texts) -> list:
        """Calibration-only conformal sets; [] and [0,1] remain explicit."""
        return [prediction_set(p, self.q_) for p in self.predict_proba(features, texts)[:, 1]]

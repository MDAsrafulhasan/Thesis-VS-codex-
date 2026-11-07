# Chapter 4: Methodology — Standalone Master Architecture for Bengali LLM Hallucination Detection

This chapter details the formal design, mathematical formulations, and modular architecture of the **BenHalluEval Master Architecture**, a 5-stage fact-decomposed, dual-track verification system designed to overcome script anchoring bias, token fertility degradation, noisy retrieval echoing, and circular judge loops in low-resource Bengali NLP.

Refer to the formal specification at [`01_Thesis/Docs/MASTER_ARCHITECTURE.md`](file:///home/itshimelz/Thesis-VS-codex-/01_Thesis/Docs/MASTER_ARCHITECTURE.md) and the runnable Python implementation in [`01_Thesis/Code/Evaluation_Pipeline/update_evaluation.py`](file:///home/itshimelz/Thesis-VS-codex-/01_Thesis/Code/Evaluation_Pipeline/update_evaluation.py).

---

## 4.1 System Overview and Operational Tuple

The evaluation pipeline consumes an input query triple:

$$x_i = (q_i, r_i, c_i)$$

where $q_i$ is the prompt (native Bengali or Romanized Banglish), $r_i$ is the candidate response under test, and $c_i$ is the optional reference context. The system decomposes detection across five sequential stages:

1. **Stage 1**: Dual-Script Normalizer & Orthographic BPE Canonicalizer Engine.
2. **Stage 2**: Fact-Decomposed Query & Decontaminated Hybrid RAG Engine.
3. **Stage 3**: Decoupled mDeBERTa-v3 Per-Claim NLI & Semantic Entailment Engine.
4. **Stage 4**: First-Token Softmax Logit Judge & Symbolic Execution Sandbox.
5. **Stage 5**: Multi-Seed Grouped Meta-Classifier & Dual-Threshold BHS Engine.

---

## 4.2 Key Mathematical Formulations

### 4.2.1 Token Fertility Ratio
$$F(S) = \frac{|T|}{W}$$
Captures token over-fragmentation in standard Byte-Pair Encoding (BPE) tokenizers on Bengali text.

### 4.2.2 Reciprocal Rank Fusion (RRF) Retrieval
$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}, \quad k = 60, \; M = \{\text{mE5-Large}, \text{BM25}\}$$
Fuses dense semantic similarity and sparse keyword matching over sentence segments to avoid single-retriever bias.

### 4.2.3 Per-Claim NLI Extrema
Multi-sentence answers are split into atomic claims $c_k$ and scored against retrieved evidence $e_j$:
$$\text{min\_claim\_entailment} = \min_{k} \left( \max_{j} P_{\text{entail}}(e_j, c_k) \right)$$
$$\text{max\_claim\_contradiction} = \max_{k} \left( \max_{j} P_{\text{contrad}}(e_j, c_k) \right)$$
Resolves whole-answer dilution bias where answers that are partially true mask single fabricated facts.

### 4.2.4 Continuous First-Token Softmax Logits
Using 4-bit CausalLM first-token unnormalized logits $L_{\text{hall}}$ and $L_{\text{faith}}$:
$$S_{\text{soft}} = \frac{\exp(L_{\text{hall}})}{\exp(L_{\text{hall}}) + \exp(L_{\text{faith}})}$$
$$V_{\text{hard}} = \mathbb{I}\left( S_{\text{soft}} > 0.5 \right), \quad M_{\text{logit}} = |2 S_{\text{soft}} - 1|$$

### 4.2.5 Dual-Threshold Grid Optimization
Optimizes decision boundaries independently across Open-Book and Closed-Book regimes:
$$\text{thr}_{\text{open}}^*, \text{thr}_{\text{closed}}^* = \arg\max_{\text{thr}_o, \text{thr}_c} \left[ F_1^{\text{open}}(\text{thr}_o) + F_1^{\text{closed}}(\text{thr}_c) \right]$$

### 4.2.6 Dual-Track BenHalluScore (BHS) Calibration
$$\text{BenHalluScore} = \frac{1}{2} \left( \frac{W_A}{N_A} + \frac{W_B}{N_B} \right) \times 100\%$$
A lower-is-better weighted error rate balancing Track A false alarms and Track B missed hallucinations.

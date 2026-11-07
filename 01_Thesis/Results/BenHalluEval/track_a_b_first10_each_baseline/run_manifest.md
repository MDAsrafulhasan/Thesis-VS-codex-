# Run manifest: BenHalluScore, 10 rows per track

- Run date: 2026-11-07 (Asia/Dhaka)
- Input CSV: `01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv`
- Input size: 20 unique IDs; 10 Track_A and 10 Track_B; all `bn_qa`
- Track_A selection: the same first 10 rows used in the Track-A-only baseline
- Track_B selection: first 10 Track_B rows in full-dataset order
- Track_B type counts: Comprehension 3, Factualness 3, Inference 2, Specificity 2
- Evaluator: `01_Thesis/Code/Evaluation_Pipeline/eval.py`
- Backend: local Ollama CPU server, `http://127.0.0.1:11436/api/generate`
- Prompt mode: `zero_shot`
- Decoding: temperature 0.0, seed 42, `num_predict=1024`
- Models: `gemma3:1b`, `qwen3:1.7b`, `qwen2.5:1.5b`, `bangla-ai-1.7b-q4`
- Track_A checkpoints: copied from the existing `track_a_first10` run without changing predictions
- Track_B inference: run on all 10 added rows per model
- Results: `results/`
- JSONL checkpoints: `outputs/`

Reproduce from the workspace root (the models must be available locally and the Ollama endpoint running):

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\eval.py" `
  --input_csv "01_Thesis\Data\BenHalluEval\Processed\Splits\track_a_b_first10_each.csv" `
  --output_dir "01_Thesis\Results\BenHalluEval\track_a_b_first10_each_baseline" `
  --backend ollama `
  --models "gemma3:1b" "qwen3:1.7b" "qwen2.5:1.5b" "bangla-ai-1.7b-q4" `
  --prompt_modes zero_shot `
  --split_name track_a_b_first10_each `
  --ollama_endpoint "http://127.0.0.1:11436/api/generate"
```

BHS is `100 × (0.5 × FPR_A + 0.5 × FNR_B)`. Each rate uses parsed verdicts from that track. Parse failures have no imputed labels; report their counts and collapse rate alongside BHS. On restart, checkpoints are deduplicated by ID using the latest attempt.

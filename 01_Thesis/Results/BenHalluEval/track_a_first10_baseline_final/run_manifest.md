# Run manifest: Track-A first-10 baseline

- Run date: 2026-11-07 (Asia/Dhaka)
- Dataset: `01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_first10_full.csv`
- Input scope: first 10 Track-A rows in source order; 10 unique IDs; all `bn_qa`; all ground-truth label 0.
- Evaluation script: `01_Thesis/Code/Evaluation_Pipeline/eval.py`
- Backend: local Ollama, CPU-only server at `http://127.0.0.1:11436/api/generate`
- Prompt mode: `zero_shot`
- Decoding: temperature 0.0, maximum 1,024 generated tokens, seed 42
- Models: `gemma3:1b`, `qwen3:1.7b`, `qwen2.5:1.5b`, `bangla-ai-1.7b-q4` (local alias for `hf.co/mradermacher/Bangla-AI-1.7B-i1-GGUF:Q4_K_M`)
- Checkpoint suffix: `track_a_first10`
- Results directory: `results/`
- Row checkpoints: `outputs/`

Reproduce from the workspace root (with the listed models available in Ollama and the endpoint running):

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\eval.py" `
  --input_csv "01_Thesis\Data\BenHalluEval\Processed\Splits\track_a_first10_full.csv" `
  --output_dir "01_Thesis\Results\BenHalluEval\track_a_first10_baseline_final" `
  --backend ollama `
  --models "gemma3:1b" "qwen3:1.7b" "qwen2.5:1.5b" "bangla-ai-1.7b-q4" `
  --prompt_modes zero_shot `
  --split_name track_a_first10 `
  --ollama_endpoint "http://127.0.0.1:11436/api/generate"
```

The checkpoint is append-only and may contain failed attempts followed by successful retries. Summary generation deduplicates by ID and uses the latest record. Track-A FPR is calculated only over successfully parsed verdicts; parse failures are counted separately. BHS is unavailable without any Track-B rows.

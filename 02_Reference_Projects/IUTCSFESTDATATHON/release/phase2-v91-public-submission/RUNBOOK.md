# Run the frozen notebook on Kaggle

The published Kaggle notebook is ready to run without code changes. A runner
still needs a Kaggle account, access to the competition data, and enough GPU
quota for a T4 pair.

## One-click path

1. Join the [competition](https://www.kaggle.com/competitions/bengali-hallucination)
   and accept its rules. Kaggle cannot mount the competition CSV for an account
   that has not accepted the rules.
2. Open the frozen [Kaggle notebook, version 3](https://www.kaggle.com/code/seyamalam/bengali-hallucination-v91-compliance-inference/versions/3).
3. Choose **Copy & Edit**.
4. Confirm the accelerator is **GPU T4 x2** and internet is **Off**.
5. Confirm that the Inputs panel contains:

   - the Bengali hallucination competition;
   - `bengali-hallucination-phase2-runtime-v91`, version 4;
   - `bengali-hallucination-llama-cpp-cuda-b9637-public`, version 2;
   - `Qwen3.6-27B-Q4_K_P`, version 1.

6. Select **Run All**.

The final file is `/kaggle/working/submission.csv`.

## Do not change the Phase 2 path

The notebook uses the literal path required by the organizers:

```python
TEST_PATH = Path(
    "/kaggle/input/competitions/bengali-hallucination/test set.csv"
)
```

There is no Phase 2 variable to edit. The organizers replace the file mounted
at this path. The notebook reads the new row count and IDs from that file.

## Expected Phase 1 result

On the current 2,516-row competition file, the complete run should produce:

- columns: exactly `id,label`;
- label counts: `0 = 1,181`, `1 = 1,335`;
- SHA-256:
  `0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28`;
- runtime: 440.87 to 593.43 seconds in three verified T4 x2 runs.

The Phase 1 hash and label counts are reproduction checks only. They are not
expected to match on the Phase 2 held-out file.

## If the copied notebook is missing an input

Attach the exact public version from [SUBMISSION_LINKS.md](SUBMISSION_LINKS.md).
Do not replace it with “latest,” edit the notebook path, enable internet, or
upload a saved prediction file.

Someone who is not signed in, has not joined the competition, or has no T4 x2
quota cannot run the notebook as published. That is a Kaggle access constraint,
not a missing notebook dependency.

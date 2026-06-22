# Fine-Tune Dataset Manifest: qlora_2_5k_v1

Purpose: LoRA fine-tuning experiment for the single-model Vietnamese MCQ solver.

Approval status: user confirmed all included sources have acceptable licenses or
permission for this hackathon training use.

## Files

```text
data/finetune/qlora_2_5k_v1_train.jsonl
data/finetune/qlora_2_5k_v1_val.jsonl
data/finetune/qlora_2_5k_v1_test.jsonl
data/finetune/qlora_2_5k_v1_stats.json
```

## Counts

| Split | Count |
|---|---:|
| train | 1905 |
| val | 241 |
| test | 231 |
| total | 2377 |

## Intended Use

- Train LoRA adapters only.
- Preserve current reading/STEM/safety behavior while improving closed-book
  knowledge domains.
- Evaluate against public/reference diagnostics before any final submission.

## Notes

- The dataset name still contains `qlora` because it came from the original
  data-prep naming. The current training plan uses bf16 LoRA.
- Do not mix public-test answers into this dataset.
- Raw source data and transformation code are managed outside this inference
  repo.

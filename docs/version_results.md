# Model Architecture Logging

| Version | Owner | Goal | Architecture | Key Changes | Models | Retrieval | Accuracy | Inference Time | Leaderboard | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v01_baseline | @minh | Establish the current repo baseline before testing new agent architectures | LLM Only | Initial baseline from current codebase | Primary: `Qwen/Qwen2.5-7B-Instruct` | N/A | 31.75% | 203.3s | TBD | N/A |
| v02_alpha | @minh | Validate a route-aware direct-answer pipeline before adding retrieval or confidence gating | Parser -> Router -> Route-Specific Prompt -> Guided Choice | Added question parser, 4-route classifier (`reading`, `stem`, `knowledge`, `safety`), route-specific prompts, forced safety refusal on explicit harmful cases, and vLLM constrained label scoring | Primary: `Qwen/Qwen3.5-4B` via vLLM | None | 54.43% | 1291s total, 279s inference loop, 0.60s/question | TBD | Route counts: `reading=100`, `stem=201`, `knowledge=158`, `safety=4`; forced safety answers: `4` |



## Key Notes:

- __Accuracy__: Number of correct questions over the total questions
- __Inference Time__: Total inference time to answer all questions

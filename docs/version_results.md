# Version Results

> Scores are from the **HackAIthon Bảng C leaderboard** (public test set).
> Model: `Qwen/Qwen3.5-4B` on vLLM throughout.

| Version | Score | Total time | s/question | What changed |
| --- | --- | --- | --- | --- |
| `v01_baseline` | 28.73% | — | — | Bare LLM call — no routing, no guided-choice, no SC. Answer parsed from free-form output. |
| `v02_alpha` | 60.48% | — | — | Added rule router (READING / STEM / KNOWLEDGE / SAFETY) + two-pass guided-choice decoding (reason freely → constrain to one letter via `allowed_token_ids`). No self-consistency. |
| `v02_beta` | 80.13% | — | — | Added S4 self-consistency escalation: STEM always runs SC, low-margin KNOWLEDGE and READING-reason items escalate. Per-question loop (no wave batching). |
| `v02_gamma` | **85.31%** | 6424.4 s | 12.77 s/q | Restructured to wave batching: all first passes in one vLLM call, all SC escalations in another. Adaptive STEM SC depth (n=3 high-margin / n=7 low-margin). Option shuffle de-bias. Per-wave checkpoint. |

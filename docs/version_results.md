# Model Architecture Logging

| Version | Owner | Goal | Architecture | Key Changes | Models | Retrieval | Accuracy | Inference Time | Leaderboard | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v01_baseline | @uyen | Establish the current repo baseline before testing new agent architectures | Multi-agent retrieval + reasoning + routing pipeline | Initial baseline from current codebase | Primary: `Qwen/Qwen2.5-7B-Instruct`<br>Secondary: `google/gemma-2-9b-it`<br>Embedder: `keepitreal/vietnamese-sbert` | Hybrid retrieval: `FAISS dense + BM25`<br>`top_k=5`<br>`relevance_threshold=0.65` | TBD | TBD | TBD | First-pass CoT without context, optional context pass, confidence gate, consistency sampling, ensemble fallback |

## Key Notes:

- __Accuracy__: Number of correct questions over the total questions
- __Inference Time__: Total inference time to answer all questions
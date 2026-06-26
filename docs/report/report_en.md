# VietMind MCQ Method Report

![alt text](/assets/image.png)

| Field | Value |
| --- | --- |
| Team | Cow |
| Project | VietMind MCQ |
| Competition | HackAIthon 2026, Track C, Innovator |
| Final Docker image | `powato/hackaithon-cow:latest` |
| Final runner | `src/v03_gamma.py` |
| Model | `Qwen/Qwen3.5-4B` |
| Model constraint | One open LLM under 5B parameters |
| Public score | 85.96 percent on the 463 question public set |
| Proxy accuracy | 91.58 percent, 424 / 463, against our group reference answers |

## 1. Problem And Submission Contract

Track C asks each team to submit an offline Docker container that reads a test
file from `/code/private_test.json` and writes `/code/submission.csv` plus
`/code/submission_time.csv`. The final private test is expected to be much
larger than the public set, so our system was designed around three goals:

| Goal | Why It Matters |
| --- | --- |
| Accuracy | This is the main score component, so the model must solve many Vietnamese multiple choice domains well. |
| Inference speed | The private set is around 2000 questions, so slow designs can lose time score or fail to finish. |
| Reliability | A container crash, OOM, or invalid output can destroy the whole submission, even if the model is accurate. |

Our final system uses one open LLM only, `Qwen/Qwen3.5-4B`, which is under the
5B parameter limit. It runs offline at inference time and does not use RAG,
embedding models, rerankers, external APIs, or a second LLM.

The score is not only about accuracy. The rubric also rewards inference speed
and creativity in architecture. Because of that, we treated optimization as part
of the model design. A slower system with slightly higher public accuracy was
not automatically better if it was likely to fail or timeout on the larger
private set.

## 2. Main Idea

VietMind MCQ is an adaptive reasoning agent for Vietnamese multiple choice
questions. The central idea is simple: not every question should receive the
same amount of computation.

This design comes from our own experience with Vietnamese entrance style exams.
As high school students, we learned that a simple fact question can be answered
quickly, a math problem usually needs scratch work, a reading question often
needs returning to the passage, and confusing answer choices need comparison.
VietMind MCQ follows that same exam taking instinct.

The system first identifies the shape of the question, then chooses how much
reasoning to spend. This is the creative part of our architecture: the model is
not just asked to answer every question in one generic way. It is guided by a
small exam strategy layer that decides when to move fast and when to slow down.

## 3. Final Architecture

The final `v03_gamma` pipeline is a wave batched, route based system.

![VietMind MCQ report architecture](../../assets/report/vietmind_report_architecture.jpeg)

```text
Input CSV or JSON
  -> parser and loader
  -> deterministic router
  -> first reasoning wave
  -> constrained answer extraction
  -> targeted self consistency wave
  -> vote merge and fallback repair
  -> /code/submission.csv and /code/submission_time.csv
```

The router assigns each question to one of four routes:

| Route | Signal | Treatment |
| --- | --- | --- |
| `READING` | Passage, context, detail lookup, reason or purpose wording | Reread style self consistency for questions where exact evidence matters. |
| `STEM` | Formula, quantity, calculation, science or math reasoning | More deliberate reasoning and self consistency because small calculation errors can flip the answer. |
| `KNOWLEDGE` | Concept recall, many choices, ambiguous options, combination answers | Direct answer for simple cases, extra compute for high choice or tricky option structures. |
| `SAFETY` | Harmful instruction pattern with refusal style answer option | Deterministic refusal option handling when appropriate. |

After reasoning, the system does constrained extraction. Instead of trusting a
free form answer string, it asks the model to select only from valid labels for
that question. This reduces invalid outputs and supports questions with more
than four choices.

The final architecture also uses option shuffle voting during self consistency.
This reduces position bias because the model does not always see the same
answer in the same slot.

## 4. Why We Chose Qwen3.5 4B

The competition limit and hardware target made the model choice practical, not
only theoretical. We needed a model small enough for the target GPU, strong enough
for Vietnamese reasoning, and compatible with a simple offline Docker path.

`Qwen/Qwen3.5-4B` gave us the best balance. It is small enough to fit the target
hardware, strong enough to benefit from reasoning prompts, and fast enough for a
2000 question private set when paired with our wave batching and safe mode.

We considered heavier ideas after the public score improved, but the final
model path stayed conservative because completion reliability is part of the
real score.

## 5. Iteration History

We did not arrive at `v03_gamma` directly. Each version answered one question
about the system.

Alongside official public score, we also kept an internal proxy evaluation
against `data/reference/reference_answers.csv`, our group reference answer file
for the public set. On this proxy, the final `v03_gamma` submission reaches
424 / 463 correct, or 91.58 percent. We treat this as a debugging signal rather
than an official leaderboard number, but it helped us compare versions quickly
while iterating.

| Version | Public Score | Proxy Accuracy | Main Question | What We Learned |
| --- | --- | --- | --- | --- |
| `v01_baseline` | 28.73% | 28.94%, 134 / 463 | Can the model answer directly with simple parsing? | No. Free form parsing and one shot answering were too weak. |
| `v02_alpha` | 60.48% | 63.50%, 294 / 463 | Does routing plus constrained extraction help? | Yes. Valid letter extraction and broad routes gave a large gain. |
| `v02_beta` | 80.13% | 84.67%, 392 / 463 | Does extra reasoning improve hard questions? | Yes, but the per question loop was too slow. |
| `v02_gamma` | 85.31% | 90.50%, 419 / 463 | Can we keep accuracy while batching work? | Yes. Wave batching made the stronger strategy much more practical. |
| `v03_alpha` | 84.23% | 89.42%, 414 / 463 | Can we make the router more general for private data? | The cleaner router was directionally right, but some hard knowledge questions lost compute. |
| `v03_gamma` | 85.96% | 91.58%, 424 / 463 | Can we keep the cleaner router and restore useful compute? | Yes. Targeted compute recovery improved accuracy while keeping runtime realistic. |
| `v03_delta` | 87.04% | 92.22%, 427 / 463 | Do exact continuation scored margins help? | Yes for accuracy, but the method became about 4 times slower and more memory fragile. |
| `v03_epsilon` | Not promoted | Not available in local submission files | Can delta be made safe with microbatching? | It reduced some risk, but still hit OOM in smaller-memory judge-like runs. |

This history shaped the final decision. `v03_delta` proved that exact margins
can improve public accuracy, but it also showed the cost of making every
confidence decision expensive. For a 463 question public set that trade can look
attractive. For a 2000 question private set on constrained hardware, it is too
risky.

We chose `v03_gamma` because it is the best operating point: stronger than the
older fast versions, much faster than delta, and safer for a full private run.

## 6. Research And Evidence Behind The Agent

Our design was guided by a research and evidence map, not only by trial and
error. We separated evidence into two levels: our own public set measurements
and published research used as supporting rationale.

The ideas that stayed in the agent all had a practical reason:

| Idea | Research Or Evidence Source | How It Appears In VietMind MCQ |
| --- | --- | --- |
| Adaptive compute routing | Chain of thought and deliberate reasoning papers show that harder reasoning tasks benefit from extra inference time. Our own traces also showed that STEM, reading detail, and high choice knowledge had different failure patterns. | The router sends questions to `READING`, `STEM`, `KNOWLEDGE`, or `SAFETY`, then the policy decides when to escalate. |
| Self consistency | Wang et al. introduced self consistency as sampling multiple reasoning paths and choosing the most consistent answer. Our own versions also showed a large gain once extra reasoning was added. | STEM receives deliberate self consistency, while reading and knowledge use targeted escalation. |
| Two pass guided choice | Chain of thought work supports reasoning before the final answer. Our own baseline showed that direct free form parsing was weak and could produce invalid labels. | The model reasons first, then a constrained extraction step selects a valid answer letter. |
| Option shuffle voting | Zheng et al. showed that LLMs can be sensitive to multiple choice option positions. | Escalation samples can shuffle options before voting, reducing dependence on fixed label positions. |
| Wave batching | vLLM and PagedAttention show that batching and efficient KV cache management improve LLM serving throughput. | First pass reasoning and escalation are batched in waves. |
| Reliability guards | This came from contest engineering evidence: a crash or invalid file can be worse than a few wrong answers. | The runner uses checkpointing, fallback prefill, atomic writes, and best effort output. |

We also researched ideas that we chose not to ship. This was part of the
optimization process. RAG and reranking were rejected because they require
additional models, add memory risk, and do not fit the final one model
submission path. Tool based code reasoning was not chosen because it would add
another execution subsystem and was too risky for a contest container. Naive
fine tuning was not chosen because we did not validate a reliable held out gain
over the base model. Exact continuation scored margins were useful, but too
expensive for the final deployment target.

This research process helped us avoid a common trap: adding impressive
components that look creative but make the final system slower, less compliant,
or less reliable. Our final architecture is creative because it is selective.
It keeps the ideas that helped the model think better under the actual contest
constraints.

Clear research citations used in this report:

| Source                                                                                                                                     | What We Took From It                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jason Wei et al., 2022, [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Reasoning before answering can improve complex arithmetic, commonsense, and symbolic tasks, which supports our reason first, extract second pattern. |
| Xuezhi Wang et al., 2022, [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | Multiple sampled reasoning paths can improve final answer reliability, which supports targeted self consistency. |
| Chujie Zheng et al., 2023, [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882) | LLMs can have option position bias in MCQ settings, which supports option shuffle voting. |
| Woosuk Kwon et al., 2023, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Efficient KV cache management and batching improve LLM serving throughput, which supports using vLLM and wave batching. |
| Shunyu Yao et al., 2023, [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | Deliberate exploration can help hard reasoning tasks, which supports the broader idea of spending extra compute selectively, even though we did not ship tree search. |

## 7. What Changed In The Final Version

The main improvement from `v03_alpha` to `v03_gamma` was not to undo the cleaner
router. Instead, we changed the compute policy.

| Issue Found | Final Fix |
| --- | --- |
| High choice knowledge questions were no longer routed to STEM, so they lost extra reasoning. | Keep them as `KNOWLEDGE`, but give them targeted self consistency. |
| Reading questions with detail retrieval were too cheap. | Expand reading self consistency to exact evidence and context lookup questions. |
| Some knowledge questions had ambiguous or combination style options. | Add escalation rules for option structures that are known to be risky. |
| Long reasoning outputs could make extraction prompts too large. | Add length safe extraction prompts for later waves. |
| Duplicate or near duplicate options could confuse vote remapping. | Use safer label handling after option shuffling. |

This is important because it shows the final model was not tuned by blindly
adding more tokens everywhere. We added compute where the question structure
made mistakes more likely.

## 8. Optimization And Reliability

The final system includes several practical optimizations that matter for the
score, not just for engineering neatness.

| Optimization | Effect |
| --- | --- |
| Wave batching | Groups first pass and escalation calls so vLLM can use the GPU more efficiently. |
| Safe mode | Uses conservative vLLM settings for the 32 GB VRAM target. |
| Constrained extraction | Reduces invalid answers and keeps labels inside the legal option set. |
| Option shuffle voting | Reduces answer position bias during self consistency. |
| Warmup pass | Primes vLLM kernels to reduce first run latency spikes. |
| Fallback prefill and atomic writes | Helps guarantee that `submission.csv` is still complete if the run is interrupted or degraded. |
| CSV and JSON loader | Supports both official input styles and questions with more than four choices. |

We also chose not to ship RAG, embedding models, rerankers, or a second model.
This keeps the system compliant with the rules and avoids memory contention on
the target GPU.

## 9. Why Not The Highest Public Score Version

Our strongest public score came from `v03_delta` at 87.04 percent. It used a
more faithful continuation scored margin to decide when the model was uncertain.
This was a useful experiment because it validated the idea that better
confidence signals can improve accuracy.

However, `v03_delta` took about 27.53 seconds per question, compared with about
7.98 seconds per question for `v03_gamma` on our local 24 GB RTX class setup.
It also remained OOM prone on smaller hardware during long runs.

For the final private set, we expect around 2000 questions. A method that is
more accurate on the public set but much slower and less stable can become a
worse submission in the real environment. Our final choice favors expected
score under judge constraints, not only public leaderboard maximum.

## 10. Limitations And Deployment Readiness

The final system is designed for the contest constraints, but it still has
clear limitations. Some niche knowledge questions remain beyond the reliable
capability of a 4B model without retrieval or a larger model. `v03_gamma` also
uses a lightweight confidence signal rather than the exact real margin explored
in `v03_delta`, so its escalation decisions are intentionally simpler. Self
consistency improves difficult questions, but it still increases runtime. Finally,
the public set is much smaller than the private set, so we treat leaderboard
results as evidence, not a guarantee.

These limitations are also why the final design stays conservative. We do not
use RAG, fine tuning, external APIs, embedding models, rerankers, or a second
LLM. This keeps the system compliant, easier to reproduce, and less likely to
fail under the VRAM target.

The submitted system is deployment ready in the following sense:

| Readiness Item | Status |
| --- | --- |
| Offline Docker inference | The model and code are packaged for container execution without runtime internet. |
| Single model | Uses only `Qwen/Qwen3.5-4B`. |
| Competition I/O | Reads `/code/private_test.json` and writes `/code/submission.csv` plus `/code/submission_time.csv`. |
| Output format | Writes `qid,answer` and `qid,answer,time`. |
| Fault tolerance | Uses checkpointing, fallback answers, atomic writes, and best effort always emit behavior. |
| 32 GB GPU safety | Uses `--safe-mode` with conservative vLLM settings. |

In short, the operating principle is: answer easy questions quickly, reason more
carefully on risky questions, and remain robust during offline deployment.

## 11. Final Submission

VietMind MCQ ships `v03_gamma`.

| Item | Final Choice |
| --- | --- |
| Docker image | `powato/hackaithon-cow:latest` |
| Runner | `src/v03_gamma.py` |
| Model | `Qwen/Qwen3.5-4B` |
| Model constraint | One open LLM under 5B parameters |
| Inference | Offline, one model only |
| Input | `/code/private_test.json` |
| Output | `/code/submission.csv` and `/code/submission_time.csv` |
| Target GPU | NVIDIA CUDA GPU with at least 32 GB VRAM |

Our final design is not simply a prompt. It is an exam taking system around a
small LLM: parse the problem, identify the route, spend compute where risk is
high, constrain the answer, and always write a valid submission. That balance of
adaptive reasoning, batching, and deployment safety is the main contribution of
VietMind MCQ.

## 12. Acknowledgements

We would like to thank HackAIthon 2026, VSDS, Vietcombank, VNPT AI, the
organizers, sponsors, technical supporters, mentors, and judges for creating a
serious and welcoming space for students to build, evaluate, and improve real
AI systems.

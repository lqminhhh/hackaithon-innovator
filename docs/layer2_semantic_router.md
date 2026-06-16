# Layer 2 Semantic Router

This document defines the first implementation target for Layer 2 semantic routing.

Layer 2 is designed as a review layer on top of the existing Layer 1 rule-based router. It should not replace Layer 1 blindly.

## Purpose

Layer 1 is fast and deterministic, but it can misroute ambiguous questions.

Layer 2 uses semantic similarity with `BGE-M3` to:

- review ambiguous cases
- score all route candidates
- recommend whether the final route should stay with Layer 1 or be overridden

## Target Flow

1. Parse question with [src/parser.py](/Users/minhle/Documents/hackaithon-innovator/src/parser.py)
2. Get Layer 1 route from [src/router.py](/Users/minhle/Documents/hackaithon-innovator/src/router.py)
3. Build semantic query text
4. Embed query text with `BGE-M3`
5. Compare against route prototypes
6. Aggregate similarity by route
7. Return:
   - `layer1_route`
   - `layer2_route`
   - `final_route`
   - score breakdown
   - override recommendation

## Files Added

- Config: [semantic_router_config.yaml](/Users/minhle/Documents/hackaithon-innovator/configs/semantic_router_config.yaml)
- Prototype bank: [route_prototypes.yaml](/Users/minhle/Documents/hackaithon-innovator/data/route_prototypes.yaml)
- Router module: [semantic_router.py](/Users/minhle/Documents/hackaithon-innovator/src/semantic_router.py)
- Tests: [test_semantic_router.py](/Users/minhle/Documents/hackaithon-innovator/tests/test_semantic_router.py)

## Semantic Router API

Main class:

- `SemanticRouter`

Important methods:

- `build_query_text(parsed)`
- `score_routes(parsed)`
- `decide_route(parsed, layer1_route=None)`

Return object:

- `SemanticRouterResult`

Fields:

- `layer1_route`
- `layer2_route`
- `final_route`
- `route_scores`
- `should_override`
- `was_ambiguous`
- `query_text`

## Prototype Format

The prototype bank is stored in YAML.

Each route has:

- `description`
- `examples`

The current prototype bank is only a starter set. It should be replaced with a stronger curated bank built from representative real questions.

## First Evaluation Workflow

Start with offline validation before integrating into `v02_alpha`.

### Step 1

Create a small hand-checked routing set:

- 15 `reading`
- 15 `stem`
- 15 `knowledge`
- all available `safety` / near-safety cases

### Step 2

For each sample, record:

- `qid`
- Layer 1 route
- Layer 2 route
- final route recommendation
- route scores
- whether override happened
- whether override looks correct

### Step 3

Inspect where Layer 2 helps:

- legal questions misrouted as `stem`
- mixed-format questions with many options
- questions with refusal choices but not truly harmful
- vague knowledge questions that Layer 1 cannot classify strongly

### Step 4

Inspect where Layer 2 hurts:

- obvious reading questions
- obvious quantitative questions
- easy knowledge questions
- false safety triggers

## Recommended Next Build Step

Before integration into `v02_alpha`, improve two things:

1. Replace the starter prototype bank with curated examples from the dataset
2. Build a notebook or small script that prints Layer 1 vs Layer 2 routing decisions on a hand-picked validation slice

Only after that should Layer 2 be wired into the main pipeline.

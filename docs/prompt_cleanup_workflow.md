# Prompt Cleanup Workflow

This document is a handoff guide for improving the route-specific prompts in [configs/prompts.yaml](/Users/minhle/Documents/hackaithon-innovator/configs/prompts.yaml).

The goal is to improve prompt quality without changing the overall `v02_alpha` architecture:

- `parser -> Layer 1 router -> route-specific prompt -> guided-choice scoring`

This task should stay focused on prompt behavior only. Do not mix in retrieval changes, confidence-gate logic, or Layer 2 semantic routing work.

## Scope

The prompt cleanup owner should work on these prompt families:

- `reading_direct`
- `stem_direct`
- `knowledge_direct`
- `safety_direct`
- optionally `guided_choice_no_context`
- optionally `guided_choice_with_context`

The older CoT prompts can be ignored for now unless they are still used in a notebook experiment.

## Main Objective

Improve the prompts so they are:

- clearer
- less biased toward one label such as `A`
- more distinct by route
- more reliable on Vietnamese multiple-choice questions
- more consistent about when refusal is correct

## Non-Goals

Do not do these in this task:

- change parser logic
- change rule-based routing
- add retrieval
- add confidence-gate logic
- add Layer 2 semantic router logic
- redesign the pipeline

If a prompt issue appears to require code changes, note it separately instead of mixing it into this task.

## Current Prompt Intent

Use these route behaviors as the target:

- `reading_direct`: answer only from the provided passage/context
- `stem_direct`: solve using quantitative or analytical reasoning
- `knowledge_direct`: answer with general knowledge or domain knowledge, without external context passage
- `safety_direct`: choose refusal only when the question truly asks for harmful, illegal, or dangerous guidance

## Cleanup Workflow

### Step 1: Inspect the Current Prompts

Read [configs/prompts.yaml](/Users/minhle/Documents/hackaithon-innovator/configs/prompts.yaml) and answer:

- Is each route prompt clearly different from the others?
- Does each prompt say what evidence source the model should use?
- Does each prompt clearly constrain the output to one legal label?
- Does each prompt contain unnecessary wording that may confuse the model?

Write down a short note for each route:

- what the prompt is trying to do
- what looks weak or ambiguous
- what should be simplified

### Step 2: Build a Small Route-Balanced Inspection Set

Pick around 10 examples per route from [data/public-test_1780368312.json](/Users/minhle/Documents/hackaithon-innovator/data/public-test_1780368312.json):

- 10 `reading`
- 10 `stem`
- 10 `knowledge`
- 10 `safety` or near-safety cases if available

This does not need to be a formal benchmark yet. It is a prompt inspection set.

Recommended fields to track in a sheet or markdown note:

- `qid`
- route
- question summary
- prompt used
- predicted answer
- expected behavior
- observed issue

### Step 3: Identify Failure Patterns

For each route, look for recurring prompt-level issues such as:

- over-selecting option `A`
- over-selecting refusal options
- not respecting the provided passage
- weak handling of quantitative reasoning
- prompt too verbose
- prompt contains conflicting instructions
- prompt does not clearly tell the model how to choose among similar distractors

Try to classify failures as one of:

- prompt issue
- model knowledge issue
- router issue
- dataset ambiguity

Only prompt issues should be addressed in this task.

### Step 4: Revise One Prompt Family at a Time

Do not rewrite all prompts at once.

Recommended order:

1. `knowledge_direct`
2. `reading_direct`
3. `stem_direct`
4. `safety_direct`

Why this order:

- `knowledge_direct` is the default fallback route and often affects many samples
- `reading_direct` needs clean context discipline
- `stem_direct` needs explicit quantitative reasoning guidance
- `safety_direct` needs careful refusal behavior, but the route count is smaller

For each revision:

- make only a small number of wording changes
- keep the output format stable
- re-test on the same small inspection set
- compare behavior before and after

### Step 5: Keep the Output Constraint Stable

Every route prompt should continue to enforce:

- choose exactly one legal answer label
- return only one label
- do not generate explanations in the final output

The current output style is:

- legal label set is shown
- allowed labels are listed explicitly
- prompt ends with `Đáp án:`

Keep this stable unless there is a strong reason to change it.

### Step 6: Make the Prompts More Distinct

Each route should have a noticeably different instruction style:

- `reading_direct`
  - emphasize that only the supplied passage may be used
  - avoid outside knowledge

- `stem_direct`
  - emphasize careful reasoning and checking calculations
  - prefer correctness over quick guessing

- `knowledge_direct`
  - emphasize selecting the factually correct option
  - do not over-refuse when the question is harmless

- `safety_direct`
  - emphasize that refusal is correct only for genuinely harmful or illegal guidance requests
  - do not choose harmful operational options when a refusal option exists

### Step 7: Watch for Label Bias

This is especially important in the current system.

Check whether the prompt wording may accidentally encourage:

- first-option bias
- shortest-option bias
- refusal-option bias

If the model keeps picking `A` too often, try:

- reducing generic filler text
- making the comparison instruction more explicit
- simplifying the instruction so the model focuses on option discrimination

Do not assume all bias comes from prompts, but prompts are the first thing to clean up.

### Step 8: Document Every Prompt Revision

For each edit to [configs/prompts.yaml](/Users/minhle/Documents/hackaithon-innovator/configs/prompts.yaml), record:

- prompt name
- old wording summary
- new wording summary
- why the change was made
- what examples were used to inspect the change

This can be recorded in:

- a short markdown note in `docs/`
- or directly in the PR / commit message

## Practical Checklist

Use this checklist before marking the task done:

- each route prompt has a clear role
- route prompts do not sound nearly identical
- output format is still one legal label only
- `reading_direct` clearly forbids outside knowledge
- `stem_direct` clearly emphasizes quantitative reasoning
- `knowledge_direct` does not encourage unnecessary refusal
- `safety_direct` handles harmful guidance correctly
- at least one small route-balanced inspection pass was completed
- prompt changes were documented

## Suggested Deliverables

The teammate handling prompt cleanup should produce:

1. Updated [configs/prompts.yaml](/Users/minhle/Documents/hackaithon-innovator/configs/prompts.yaml)
2. A short note describing what changed and why
3. A few example cases showing before/after behavior
4. Any open questions that may require routing or model changes instead of prompt changes

## Handoff Back to Layer 2 Work

Once prompt cleanup is in a good place, the next architecture step is:

- build Layer 2 `BGE-M3` semantic router on top of the cleaned prompt system

That means prompt cleanup should aim for:

- stable route-specific behavior
- less answer bias
- cleaner base prompts for later semantic routing experiments

## Recommended Rule of Thumb

If a prompt sentence does not clearly improve behavior, remove or simplify it.

Prompt cleanup should move toward:

- shorter instructions
- clearer route identity
- less ambiguity
- easier evaluation

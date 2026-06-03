# Prompt Injection in Multi-Hop Retrieval

This repository is a small research-style project for testing how prompt injection affects a simple multi-hop retrieval pipeline.

It builds a HotpotQA-based corpus, poisons documents with redirect instructions, runs a multi-hop agent over clean or poisoned retrieval, and evaluates the results with:

- `F1` for answer quality
- `ASR` (attack success rate) for how often the model follows the injected instruction

The code is intentionally lightweight and easy to inspect.

## What is in the repo

- `dataset/` builds the corpus and question set, and applies poisoning strategies
- `retireval/` contains the BM25 retriever
- `multihop_agent/` contains the retrieval + reasoning loop
- `evaluation/` runs experiments and computes metrics
- `classifier/` contains an optional adversarial-intent guard
- `results/` and `results_old/` contain saved outputs from previous runs

## Project flow

The typical workflow is:

1. Build a clean corpus and question set from HotpotQA
2. Create a poisoned corpus
3. Run the multi-hop agent with clean or poisoned retrieval
4. Measure answer quality and attack success

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```env
GROQ_API_KEY=gsk_***
BASE_URL=https://api.groq.com/openai/v1
MODEL=llama-3.1-8b-instant
TEMPERATURE=0.0
MAX_TOKENS=256
```

`OPENAI_API_KEY` or `BERGET_API_KEY` can also be used instead of `GROQ_API_KEY`, depending on the endpoint you want to call.

## Prepare the data

Generate the clean corpus, poisoned corpus, and question set:

```bash
python3 dataset/save_docs_to_local.py
```

This writes JSONL files into `data/processed/`, including:

- `corpus.jsonl`
- `corpus_poisoned_all_embedded.jsonl`
- `questions.jsonl`

Poisoning behavior is configured in [dataset/poisoning.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/dataset/poisoning.py>) and [dataset/save_docs_to_local.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/dataset/save_docs_to_local.py>).

## Run an evaluation

The current main script is [main_eval.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/main_eval.py>), which:

- loads the clean corpus
- loads the poisoned embedded corpus
- runs the multi-hop evaluation
- saves comparison outputs into `results/`

Run it with:

```bash
python3 main_eval.py
```

Evaluation logic lives in [evaluation/evaluator.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/evaluation/evaluator.py>) and metrics are defined in [evaluation/metrics.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/evaluation/metrics.py>).

## Optional components

The agent itself is in [multihop_agent/agent.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/multihop_agent/agent.py>). It:

- retrieves documents with BM25
- generates follow-up queries across hops
- produces a final answer from the accumulated context

There is also an optional classifier-based guard in [classifier/guard.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/classifier/guard.py>) for flagging adversarial documents before they enter context.

## Results processing

If you have multiple result folders, these scripts help combine them:

- [results_processing/merge_results.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/results_processing/merge_results.py>)
- [results_processing/evaluate_results.py](</Users/zuzamakowska/Documents/AI AND LANGUAGE/Information Retrieval/prompt-injection/results_processing/evaluate_results.py>)

They merge saved `comparison.csv` files and compute aggregate metrics across runs.

## Notes

- The repository currently focuses on experimentation rather than packaging
- Some scripts are intentionally simple and use fixed paths for faster iteration
- The `retireval/` directory name is kept as-is to match the existing codebase

## Quick start

If you just want the shortest path:

```bash
source venv/bin/activate
pip install -r requirements.txt
python3 dataset/save_docs_to_local.py
python3 main_eval.py
```

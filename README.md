# Subagent orchestration metrics

How the format of a subagent's report to its orchestrator changes token cost and accuracy on BrowseComp-Plus.

| Path | What it holds |
| --- | --- |
| `proposal/` | The LaTeX project proposal (see `proposal/README.md` to build it) |
| `docs/tasks.md` | The task plan |
| `docs/checklist.md` | Status of every task: what is done, how to verify it, what comes next |
| `docs/harness-design.md` | Design of the agent harness: steps, specs and tests |
| `harness/` | The harness: one question through plan, search, return, decide, with token counts per call |
| `configs/` | Model endpoints and run configs |
| `tests/` | Offline tests against a mock OpenAI-compatible server and a tiny corpus |
| `data/`, `runs/` | Questions, corpus, index and run outputs; gitignored, never committed |

## Setup

Python 3.11 or newer.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps   # the tested versions
.venv/bin/pip install -e '.[bm25]'   # only for the official BM25 index; Pyserini needs Java 21 and pulls in torch
```

`requirements.lock` pins the `dev`, `data` and `analysis` extras as tested with Python 3.14.2. Without it, `pip install -e '.[dev,data,analysis]'` takes the newest versions.

## Tests

```sh
.venv/bin/pytest                      # offline: a mock server stands in for every model, no tokens spent
HARNESS_ENDPOINTS=configs/endpoints.toml HARNESS_ENDPOINT=gptoss-20b-vllm .venv/bin/pytest -m live
```

The live test runs the endpoint smoke test against a real server and spends a few thousand tokens.

## Running

1. Download BrowseComp-Plus from Hugging Face into `data/`. Files that already exist are skipped, so rerunning is safe.

   ```sh
   .venv/bin/python -m harness.data      # about 4 GB; add --skip-index to leave out the 2.2 GB BM25 index
   ```

   This writes:
   - `questions.jsonl`: one `{"qid", "question"}` object per line, decrypted.
   - `labels.jsonl`: one `{"qid", "answer", "evidence_docids", "gold_docids"}` object per line, for grading and evidence recall. The harness never reads it, so answers can't reach a prompt.
   - `corpus.jsonl`: one `{"docid", "title", "text", "url"}` object per line (100,195 documents).
   - `bm25/`: the official prebuilt Pyserini index. Without `index` in the config, BM25 runs in memory, which is only fast enough for a small corpus.

   `--released` also fetches the BrowseComp-Plus authors' GPT-5 and o3 runs on the same BM25 index (about 480 MB decrypted) into `released_runs/`, for budget calibration.
2. Point `configs/endpoints.toml` at a server. Any OpenAI-compatible API works: vLLM or SGLang on our own GPU, a hosted provider, or a LiteLLM proxy. API keys are read from the environment variable named in `api_key_env`.
3. Check each endpoint before using it, and keep the report with the run:

   ```sh
   mkdir -p runs/pilot
   .venv/bin/python -m harness.smoke configs/endpoints.toml gptoss-20b-vllm > runs/pilot/smoke-gptoss-20b-vllm.json
   ```

4. Extract the clues the planner groups into subtasks. Clues are not part of the dataset:

   ```sh
   .venv/bin/python -m harness.clues configs/endpoints.toml gptoss-120b-vllm --limit 30   # then without --limit
   ```

   This writes `data/clues.json` (`{qid: [clue, ...]}`), a hand-check sample in `data/clues.review.md`, and prints how many questions have at least 1, 3 and 6 clues. Without `clues.json`, remove the `clues` line from the run config and the planner splits the question itself.
5. Run and check:

   ```sh
   .venv/bin/python -m harness.run configs/pilot.toml --limit 5     # also --shard 0/4, --retry-failed
   .venv/bin/python -m harness.check runs/pilot
   ```

   A run resumes where it stopped: finished results are skipped and failed ones are retried only with `--retry-failed`. Results go to `runs/<name>/results/n<N>/s<seed>/<format>/<qid>.json.gz`. Each holds the answer, every subagent's conversation and report, every model call, and token totals by phase.
6. Grade with the official BrowseComp-Plus judge prompt, then hand-check a sample:

   ```sh
   .venv/bin/python -m harness.grade judge runs/pilot configs/endpoints.toml qwen3-32b-judge
   .venv/bin/python -m harness.grade review runs/pilot          # writes runs/pilot/grades/review.md
   .venv/bin/python -m harness.grade export runs/pilot          # optional: their run format, for evaluate_run.py
   ```

   Grades go to `runs/<name>/grades/`, one per result. Rerunning `judge` grades only what is missing or unreadable.
7. Analyze: accuracy with confidence intervals, tokens by phase, tokens per correct answer, evidence recall, and every format against the medium summary at the same N:

   ```sh
   .venv/bin/python -m harness.analyze runs/pilot runs/baseline   # writes runs/pilot/analysis/
   ```

   This writes `rows.csv` (one row per result), `configs.csv` (one per configuration), `paired.csv` (McNemar test and bootstrap intervals against the medium summary) and two plots.

To set return caps (task C6), compare the released runs with a pilot run. This prints typical sizes and suggests caps for `raw` and `quotes`:

```sh
.venv/bin/python -m harness.calibrate runs/pilot
```

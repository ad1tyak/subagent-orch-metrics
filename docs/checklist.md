# Task checklist

Status of every task in `tasks.md`, as of Oct 18, 2026. Use it to check what is done and to pick up the next step.

## Where things stand

- **All the code that can be written without a model is written and tested offline.** That covers the harness (B1 to B11), the formats (C1 to C5, C7), clue extraction (A4), grading (D1), evidence recall and analysis (D2, D3) and calibration (C6). All 130 offline tests pass against a mock model server.
- **The data is downloaded and decrypted** into `data/` (about 5.6 GB, gitignored): 830 questions, their labels, 100,195 documents, the BM25 index and the two released runs.
- **The whole pipeline has run once on a local model.** Qwen3-4B on Ollama took one question through all 12 configurations at N = 1 and 3, then check, grading, analysis and calibration (see A3). The 4B results only test the plumbing. Everything left needs a real model endpoint (A3). The order is in [What to do next](#what-to-do-next).
- **The official search engine works.** Pyserini with Java 21 reproduces the authors' BM25 results exactly: 500 of 500 replayed searches gave the same top 5 in the same order. The whole pipeline has also run on real questions, documents and the index, with the mock model standing in for the LLMs.
- **Committed through the local-Ollama plumbing test.** `18fccc2` added the harness and `c203863` the search-engine setup.

Status labels:

- **Done:** code and offline tests are finished; nothing is left but what the task lists under "Next".
- **Code done, needs a run:** the tool exists and is tested on the mock; the task finishes when it runs on real models or data.
- **Not started:** no work yet.

## Check the whole repo

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps
.venv/bin/pytest                             # expect: 130 passed, 2 deselected (the live and index tests)
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
.venv/bin/pytest -m index                    # expect: 1 passed (our BM25 matches the released runs)
.venv/bin/python -m harness.data             # expect: every file "exists; skipping"
.venv/bin/python -m harness.calibrate        # the released-run statistics in C6 below
```

## A. Setup and data

### A1 Repo and environment: Done

- [x] Git repo with the layout in `README.md`: `harness/`, `tests/`, `configs/`, `docs/`, `proposal/`.
- [x] Package and extras in `pyproject.toml`: `dev`, `data`, `analysis`, and `bm25` for Pyserini.
- [x] Pinned versions in `requirements.lock`, tested with Python 3.14.2.
- [x] One test command: `.venv/bin/pytest`.
- [x] Pyserini and its dependencies (torch among them) are pinned in `requirements.lock` too.
- [ ] Next: push the commits when the team is ready.

### A2 Data setup: Done

- [x] `harness/data.py` downloads from Hugging Face and decrypts with the BrowseComp canary. It skips files that already exist, and an interrupted download never leaves a partial file.
- [x] `data/questions.jsonl` holds 830 questions (qids 1 to 1266). `data/labels.jsonl` holds the answers and evidence and gold docids, about 6.1 evidence and 2.9 gold documents per question. The harness never reads the labels.
- [x] `data/corpus.jsonl` holds 100,195 documents, all titled; it loads in about 8 s with about 2.6 GB of memory. `data/bm25/` is the official Pyserini index (2.0 GB).
- [x] `data/released_runs/` holds the authors' GPT-5 and o3 runs with BM25 (830 questions each), decrypted. The files match an independent decryption of the raw download.
- [x] `data/` and `runs/` are gitignored. Decrypted questions and answers must never be committed or published.
- Verify: `.venv/bin/pytest tests/test_data.py`, then `wc -l data/*.jsonl data/released_runs/*.jsonl`.

### A3 Model access: Code done, needs a run

- [x] `harness/llm.py` works with any OpenAI-compatible endpoint. It normalizes usage, retries on errors and limits concurrency per endpoint.
- [x] Example endpoints in `configs/endpoints.toml`: gpt-oss-120b and gpt-oss-20b on vLLM, a hosted Qwen, and two judges.
- [x] `harness/smoke.py` checks what an endpoint reports:
  - the usage of a plain call;
  - cached tokens on a repeated prompt;
  - reasoning tokens with thinking on and off;
  - whether a fixed seed repeats;
  - the chat template's overhead over our token count, which also confirms the tokenizer setting.
- [ ] Decide between a hosted API and self-hosted vLLM, then fill in `base_url`, `model` and `api_key_env`.
- [ ] Run the smoke test and keep the report with the run: `python -m harness.smoke configs/endpoints.toml <endpoint> > runs/pilot/smoke-<endpoint>.json`.
- [ ] Confirm the report includes input, output, cached and reasoning tokens. `check` counts results where they are missing.
- [ ] Run the live test: `HARNESS_ENDPOINTS=configs/endpoints.toml HARNESS_ENDPOINT=<endpoint> .venv/bin/pytest -m live`.

Local single pass (Oct 18), a plumbing test with no study results:

- [x] Endpoint `qwen3-4b-ollama`: Qwen3-4B on Ollama with a 32k context, set up as in the README's "Local test with Ollama". It uses the `qwen3:4b-q4_K_M` tag, the original hybrid release. The plain `qwen3:4b` tag is the 2507 thinking model, and with thinking off its reasoning leaks into the answer.
- [x] Smoke test and live test pass. Ollama reports full input counts even on cache hits (the repeat matches the first call), and it reports cached tokens. It does not report reasoning tokens, so we estimate them from the reasoning text. A fixed seed repeats. Template overhead is 16 tokens.
- [x] `configs/local.toml` ran one question through all 12 configurations: 12 results, 0 errors, 88 calls, `check` clean. Judging, review, analysis and calibration all ran. It took about 35 minutes on an M3 MacBook Air (about 20 tokens/s), plus a minute for the judge.
- [x] Three fixes came out of it, each with a test:
  - The first subagent finished without searching; it read the subtask as a reading-comprehension question. Both system prompts now say the answer comes from the documents. The subagent loop also rejects a finish before any search as an invalid action (`test_finishing_before_any_search_is_invalid`).
  - A claims report cited passages the subagent never opened, and the orchestrator answered from those citations at 80% confidence. Claims now keep only evidence IDs from seen passages, and their status comes from the search (`test_claims_keep_only_seen_evidence_and_the_search_status`).
  - Calibrate suggested a raw cap of 0 when nothing was opened. It now skips formats with no data.
- What the 4B model did: it planned 3 valid subtasks at N = 3, but every subagent searched once or twice and gave up without opening a document. At N = 3 one search did surface the right author. The 4B judge marked 2 "cannot determine" answers correct by copying the correct answer as the extracted one, which shows why the hand check matters.

### A4 Clue extraction: Code done, needs a run

- [x] `harness/clues.py` holds the clue-listing prompt. It saves each reply as soon as it arrives, so a rerun only asks about missing or invalid questions. It repairs bad JSON once, writes a 30-question hand-check sample, and prints a chart of clue counts.
- [x] Tested on the mock in `tests/test_clues.py`: parsing, repair, resume, failed requests, and the outputs.
- [ ] Try 30 questions: `python -m harness.clues configs/endpoints.toml gptoss-120b-vllm --limit 30`. Read `data/clues.review.md` and tick the boxes.
- [ ] If the clues look right, run all 830 by dropping `--limit`. Changing the prompt changes the clues, so finish prompt changes before the full run.
- [ ] Record from the chart how many questions have at least 3 and at least 6 clues. That tells us whether N = 6 is supported.
- Until `data/clues.json` exists, `harness.run` stops at start-up with a hint. Remove the `clues` line from `configs/pilot.toml` to let the planner split the question alone.

## B. Harness: Done

Every module is tested offline against the mock server (`tests/mock_llm.py`) and the tiny corpus (`tests/conftest.py`).

| Task | Status | Code | Tests |
| --- | --- | --- | --- |
| B1 Shared interfaces | Done | `config.py`, `records.py`, `configs/pilot.toml` | `test_config.py` |
| B2 Model client | Done | `llm.py`, `tokens.py` | `test_llm.py` |
| B3 Search tool | Done | `search.py` | `test_search.py`; `test_index.py` (opt-in, the official index) |
| B4 Planner | Done | `planner.py` | `test_planner.py` |
| B5 Subagent search loop | Done | `subagent.py`, `parsing.py` | `test_subagent.py` |
| B6 Orchestrator decision | Done | `orchestrator.py` | `test_orchestrator.py` |
| B7 Parallel execution | Done | `run.py` (questions in flight), `llm.py` (per-endpoint limit) | `test_llm.py`, `test_pipeline.py` |
| B8 Shared search cache | Done | `pipeline.py` (stage 1 runs once; formats branch in stage 2) | `test_pipeline.py` |
| B9 Token accounting | Done | `accounting.py`, `check.py` | `test_pipeline.py` |
| B10 Resumable runner | Done | `run.py`, `pipeline.py` | `test_pipeline.py` |
| B11 Offline tests | Done | `tests/` | `.venv/bin/pytest` |

Left to do:

B3, the official index, is done on this machine:

- [x] Java 21 is installed with `brew install openjdk@21`, and `JAVA_HOME` must point to it (see `README.md`). Pyserini 2.4.0 is installed from the bm25 extra and runs on Python 3.14.
- [x] The index opens in about 20 s (mostly loading torch) and holds all 100,195 documents; a search takes about 60 ms.
- [x] Our results match the authors'. 500 searches sampled from both released runs return the same top 5 in the same order. `tests/test_index.py` replays 200 of them: `pytest -m index`.
- [x] The full pipeline ran on real data with the mock model standing in: 3 questions × 6 formats × N of 1 and 3, giving 36 results. `check` passed, and grading, analysis and calibration all ran.
  - Evidence recall came out at 0 because the mock sends placeholder queries.
  - Searching each evidence document's title finds 14 of 16, so the label and index docids do match.
- [ ] Everyone running the harness needs Java 21 and `JAVA_HOME` set as above.
- [ ] **B9:** on the first real run, `python -m harness.check runs/<name>` must report no mismatches. Look at its counts of missing cached and reasoning tokens.

## C. Return formats

| Task | Status | Code | Tests |
| --- | --- | --- | --- |
| C1 Common rules | Done | `formats.py`, `prompts.toml`, `subagent.py` | `test_formats.py`, `test_subagent.py` |
| C2 Written summaries | Done | `formats.py` (hard cap; the cap hit is recorded per report) | `test_formats.py` |
| C3 Structured claims | Done | `formats.py` (one repair, then the report is marked invalid) | `test_formats.py` |
| C4 Extracted quotes | Done | `formats.py` (falls back to passages ranked by BM25 when none are flagged) | `test_formats.py` |
| C5 Raw documents | Done | `formats.py` (passages in the order their documents were opened) | `test_formats.py` |
| C6 Budget calibration | Code done; the released half is done, the pilot half needs a run | `calibrate.py`, `data.py --released` | `test_calibrate.py`, `test_data.py` |
| C7 Delivered evidence | Done | `formats.py` (the documents cited in each report) | `test_formats.py` |

### C6 Budget calibration

Done: the released-run half. Output of `python -m harness.calibrate`, counted with tiktoken o200k_base, per question (one agent, BM25, 5 hits per search):

| Run | Searches (median, p90) | Search output tokens (median, p90) | Distinct documents (median) | Final answer tokens (median) | Completed |
| --- | --- | --- | --- | --- | --- |
| GPT-5 | 24, 35 | 56,981, 83,159 | 54 | 221 | 828 of 830 |
| o3 | 28, 37 | 65,871, 88,802 | 76 | 183 | 806 of 830 |

GPT-5 averages 23.2 searches, which matches the dataset card. One agent reads about 57k tokens of search results per question. So a subagent with a share of the question reads well under that, and the current raw cap of 8,000 is in a plausible range.

- [ ] After the pilot (E3), run `python -m harness.calibrate runs/pilot`. For each subagent it prints:
  - how much it read (the uncapped raw size) and flagged (the uncapped quotes size);
  - each format's report size, and how often the cap cut it;
  - the writers' visible output.
  It suggests raw and quotes caps at the 90th percentile of the uncapped sizes. It notes written formats whose cap cuts more than 25% of reports, or whose reports stay under half the cap.
- [ ] Set the final `[budgets]` in the run configs.
  - A run refuses a changed config, so new budgets need a new run name.
  - Budgets are not part of the stage-1 key. Copying `runs/pilot/stage1/` into the new run directory reuses the plans and searches.

## D. Evaluation

### D1 Grader: Code done, needs a run

- [x] `harness/grade.py judge` uses the official BrowseComp-Plus grader prompt (verbatim) and parsing. Any endpoint can judge.
  - A result whose decision is not an answer is wrong, and no judge call is made.
  - Grades go to `runs/<run>/grades/`, one per result.
  - A rerun grades only missing or unreadable grades.
- [x] `grade review` writes a hand-check sample to `grades/review.md`.
- [x] `grade export` writes the BrowseComp-Plus run format, plus `ground_truth.jsonl` and `qrel_evidence.txt`, so their `scripts_evaluation/evaluate_run.py` can grade our answers too.
- [x] Judge endpoints in `configs/endpoints.toml`:
  - `qwen3-32b-judge` is the official setup (temperature 0.7, top_p 0.8, top_k 20, thinking off).
  - `gpt41-judge` is their alternative, within 1% of it.
- Verify: `.venv/bin/pytest tests/test_grade.py`.
- [ ] Choose the judge: Qwen3-32B on our GPU, or GPT-4.1 through the API. Then run the smoke test on it.
- [ ] Grade the integration run (E1) and hand-check 30 judgments: `python -m harness.grade review runs/<run>`.
- [ ] Once, run `evaluate_run.py` on the export of the same run and check that its accuracy matches ours.

### D2 Evidence recall: Done (needs real runs for numbers)

- [x] `harness/analyze.py` computes, per result, the share of labeled evidence documents that were:
  - retrieved, meaning they appeared in any subagent's search results;
  - opened;
  - delivered, meaning they were cited in the reports the orchestrator read.
- [x] It also computes recall of the gold documents among those delivered.
- [x] Per configuration it reports the means, in `configs.csv`.
- Verify: `.venv/bin/pytest tests/test_analyze.py`. Delivered ≤ opened ≤ retrieved holds for every result.

### D3 Analysis: Code done, needs graded runs

- [x] `python -m harness.analyze runs/pilot runs/baseline` writes to `runs/pilot/analysis/`:
  - `rows.csv`, with one row per result.
  - `configs.csv`, with one row per configuration:
    - accuracy with a 95% bootstrap interval, resampled by question;
    - mean tokens in every phase, input and output kept separate, plus cached and reasoning tokens;
    - tokens per correct answer, the re-delegation rate, how often caps cut reports, and evidence recall.
  - `paired.csv`, comparing every format with the medium summary at the same N, on the same questions and seeds:
    - the accuracy difference, with an exact McNemar p-value and a bootstrap interval;
    - the change in output and input tokens, with intervals;
    - whether the success target is met: within 2 points, with at least 20% fewer output tokens.
  - `accuracy_vs_tokens.png` and `tokens_by_phase.png`.
- [x] Without grades, the accuracy columns stay empty, so token analysis works before judging.
- [ ] Settle the success target in F1. If it changes, edit `TARGET` in `harness/analyze.py`.
- [ ] Run it on the pilot, then on the main runs, for the results section.

## E. Experiments: Not started (need A3; the pilot also needs A4)

Commands are ready for each. A run directory is named by the config's `name`. A run resumes when restarted; add `--retry-failed` to rerun failed questions.

- [ ] **E1 Integration test:**
  - Run `python -m harness.run configs/pilot.toml --limit 5`, then `python -m harness.check runs/pilot`. That is 5 questions × 18 configurations.
  - `check` must report no mismatches.
  - These results become part of the pilot.
- [ ] **E2 Baseline:**
  - Run `python -m harness.run configs/baseline.toml`: one agent on the pilot's 50 questions, with the same search settings.
  - Grade it, then compare with the published BrowseComp-Plus accuracy for a model of that size with BM25.
- [ ] **E3 Pilot:**
  - Run `python -m harness.run configs/pilot.toml`: 50 questions × 18 configurations, 900 results from 150 shared stage-1 runs.
  - Grade the results, analyze them and calibrate (C6).
  - Review failures in `check`'s output.
  - Lock the budgets and N values.
- [ ] **E4 Main runs:**
  - Write `configs/main.toml`: 200 questions, `seeds = [0, 1]`, the locked budgets. That makes 7,200 results.
  - Write a second config for the key configurations on all 830 questions.
  - Use `--shard i/k` to split a run across machines.
- [ ] **E5 Free-choice check:** a config with `n_mode = "free"` on a subset of questions.

## F. Writing and coordination: Not started

- [ ] **F1:** update the proposal for tokens. `proposal/main.tex` is still framed around latency: the title, H2, H3, the metrics and the success target.
- [ ] **F2:** mid-term report and talk (Nov 5).
- [ ] **F3:** final report (Dec 9).
- [ ] **F4:** final talk (between Dec 1 and Dec 8).
- [ ] **F5:** code release.
  - The code is close to release as is.
  - Before publishing traces, check the BrowseComp-Plus terms: results contain decrypted questions.
- [ ] **F6:** team meetings (ongoing).

## What to do next

1. **Endpoint (A3).**
   - Stand up or choose the gpt-oss endpoints, fill in `configs/endpoints.toml`, and run the smoke test.
   - Run the live test.
2. **Clues (A4).**
   - Run 30, review them, then run all 830.
   - Settle on N = 6 from the chart.
3. **Integration (E1).** Run 5 questions through all 18 configurations and run `check`.
4. **Judge (D1).**
   - Choose the judge, run the smoke test on it, and grade E1.
   - Hand-check 30 judgments.
   - Cross-check once with `evaluate_run.py`.
5. **Baseline and pilot (E2, E3).** Then calibrate (C6) and lock the budgets.
6. **F1.** Update the proposal before the mid-term.

Decisions still open:

- Hosted API or our own GPU (A3).
- Which judge (D1).
- The exact success target (F1).
- Whether to keep N = 6 (A4 chart).
- The final caps (C6).

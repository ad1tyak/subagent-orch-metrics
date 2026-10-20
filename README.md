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
```

`requirements.lock` pins every extra as tested with Python 3.14.2: `dev`, `data`, `analysis` and `bm25`. The `bm25` extra is Pyserini, which reads the official BM25 index and pulls in torch. Without the lock, `pip install -e '.[dev,data,analysis,bm25]'` takes the newest versions.

Pyserini needs Java 21. On macOS:

```sh
brew install openjdk@21
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home   # add to ~/.zshrc
```

Opening the index takes about 20 seconds, once per run; each search then takes about 60 ms.

## Tests

```sh
.venv/bin/pytest                      # offline: a mock server stands in for every model, no tokens spent
HARNESS_ENDPOINTS=configs/endpoints.toml HARNESS_ENDPOINT=gptoss-20b-vllm .venv/bin/pytest -m live
```

The live test runs the endpoint smoke test against a real server and spends a few thousand tokens. A second opt-in test replays 200 searches from the BrowseComp-Plus authors' released runs and checks that our BM25 index returns the same top 5:

```sh
.venv/bin/pytest -m index    # needs Java 21, data/bm25 and data/released_runs
```

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

## Local test with Ollama

A laptop can run the whole pipeline end to end with a small local model, to check the plumbing before spending money on real runs. The results are not part of the study, because a 4B model is far weaker than the study's models. On a 16 GB Mac, the harness takes about 5 GB and the model about 5.5 GB.

1. Install [Ollama](https://ollama.com), then build Qwen3-4B with a 32k context. Ollama gives machines under 24 GB a 4k context and cuts longer prompts without warning:

   ```sh
   ollama pull qwen3:4b-q4_K_M      # the original hybrid release; the plain qwen3:4b tag cannot turn thinking off
   ollama create qwen3-4b-32k -f configs/qwen3-4b-32k.Modelfile
   ```

2. Set the server to hold one request at a time with a smaller context cache, then restart the Ollama app:

   ```sh
   launchctl setenv OLLAMA_FLASH_ATTENTION 1
   launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
   launchctl setenv OLLAMA_NUM_PARALLEL 1
   ```

   After the first request, `ollama ps` should show CONTEXT 32768 and 100% GPU. `launchctl setenv` lasts until the next reboot.

3. Run one question through every configuration with the `qwen3-4b-ollama` endpoint. Clues go to a separate file, so the test model's clues never mix with the real ones:

   ```sh
   .venv/bin/python -m harness.smoke configs/endpoints.toml qwen3-4b-ollama
   .venv/bin/python -m harness.clues configs/endpoints.toml qwen3-4b-ollama --limit 1 --review 1 --out data/local/clues.json
   caffeinate -is .venv/bin/python -m harness.run configs/local.toml      # caffeinate keeps the Mac awake
   .venv/bin/python -m harness.check runs/local
   .venv/bin/python -m harness.grade judge runs/local configs/endpoints.toml qwen3-4b-ollama
   .venv/bin/python -m harness.analyze runs/local
   ```

   On an M3 MacBook Air the run takes about 35 minutes and the judge about a minute. Expect the 4B model to search once or twice and give up, and expect its judgments to be unreliable (it sometimes copies the correct answer as the extracted one). That is fine: the test checks that every stage runs and records its tokens, not accuracy.

# Build Plan: Subagent Return Study

Token-based version · Oct 16, 2026

## Summary

- **We measure tokens in place of latency.** Every cost is a token count from the model server's usage report, split by phase, with generated and read tokens kept separate.
- **Total work:** about 240 hours across 36 tasks.
- **Available:** about 300 hours (4 people at roughly 10 hours a week until Dec 9), leaving about 60 hours of buffer.
- **Front-loaded:** about 155 of those hours fall before the Nov 5 mid-term, which means roughly 13 hours per person per week until then.
- **Simpler compute:** without timing runs we no longer need a dedicated GPU, and runs can go through a hosted API.

## What changes when we measure tokens

Token counts come straight from the model server's usage report, so they are exact and do not depend on hardware or how busy the server is.

- **Dropped:** the fixed-load latency runs, the timing breakdown, and the need for a dedicated GPU. Runs can now be packed at full speed or sent to a hosted API.
- **Kept:** the same split by phase, now counted in tokens.
- **Reported separately:** generated (output) tokens and read (input) tokens. Generating a token costs far more than reading one, so a single total would hide the trade-off between a subagent writing a summary and the orchestrator reading raw text.

| Metric | What it counts |
| --- | --- |
| Return size | Tokens the orchestrator receives from each subagent |
| Subagent search tokens | Input and output tokens across all search turns |
| Subagent writing tokens | Output tokens spent writing the return (zero for copied formats) |
| Orchestrator reading tokens | Input tokens of the orchestrator's call, including all reports |
| Orchestrator answer tokens | Output tokens of the orchestrator's decision |
| Extra-round tokens | Everything spent in re-delegation rounds |
| Totals per question | Total input and total output tokens |
| Tokens per correct answer | Output and input tokens divided by correct answers |

The proposal still frames the study around latency: the title, H2, H3 and the success target. Task F1 updates them; one option for the target is "within 2 points of the 1,000-token summary's accuracy with at least 20% fewer output tokens."

## Task plan

### A. Setup and data (16 hours)

| # | Task | Steps | Output | Hours |
| --- | --- | --- | --- | --- |
| A1 | Repo and environment | Create the git repo and folder layout; pin Python and package versions; add one test command | A shared repo everyone runs the same way | 3 |
| A2 | Data setup | Download and decrypt the BrowseComp-Plus questions; download the corpus and BM25 index; write loaders; keep decrypted questions out of git | Questions, corpus, labels and index ready | 4 |
| A3 | Model access | Pick a hosted API or self-hosted vLLM; set up the endpoint config; send one test call; confirm the usage report includes input, output and cached tokens | A working model endpoint | 3 |
| A4 | Clue extraction | Write the clue-listing prompt; run it on all 830 questions; hand-check 30; save the clue file; chart how many clues questions have | The clue file, and which N values the data supports | 6 |

### B. Harness (65 hours)

| # | Task | Steps | Output | Hours |
| --- | --- | --- | --- | --- |
| B1 | Shared interfaces | Define the config (format, N, seed, model), the per-call token record (role, phase, input, output, cached and reasoning tokens) and the per-question result file | One contract everyone builds against | 4 |
| B2 | Model client | Call any OpenAI-compatible endpoint; read the usage report; retry on errors; tag each call with role and phase | Every call logged with exact token counts | 5 |
| B3 | Search tool | Wrap the BM25 index; split documents into passages with IDs; add search and open actions; cap text shown per document; record what each subagent saw | The tool subagents call, plus evidence tracking | 8 |
| B4 | Planner | Prompt the orchestrator to group clues into exactly N subtasks; fix wrong counts; record the actual N when a question has fewer clues | N under control with a natural split | 6 |
| B5 | Subagent search loop | Choose the action protocol; implement search, open and finish; enforce a step limit; flag key passages at finish; handle invalid actions | One search trajectory that feeds any format | 10 |
| B6 | Orchestrator decision | Read all reports; answer or re-delegate; cap the rounds; write the answer in BrowseComp grading style | Final answers and the re-delegation rate | 6 |
| B7 | Parallel execution | Run N subagents at once; limit questions in flight; respect API rate limits | Fast runs within quota | 3 |
| B8 | Shared search cache | Reuse the plan and first-round searches across all six formats | Only the return differs between formats; fewer tokens spent | 5 |
| B9 | Token accounting | Sum tokens by phase (plan, search, writing, return size, orchestrator reading and answer, extra rounds); compute tokens per correct answer | The cost results | 6 |
| B10 | Resumable runner | One result file per question; skip finished questions on restart; run a grid of format × N from one command | Runs survive interruptions | 4 |
| B11 | Offline tests | Build a mock model and a tiny corpus; test each module and one full question | Bugs caught before spending tokens | 8 |

### C. Return formats (29 hours)

| # | Task | Steps | Output | Hours |
| --- | --- | --- | --- | --- |
| C1 | Common rules | Reveal the format only at the subagent's final turn; use the same report wrapper for every format; count budgets with the model's tokenizer; turn thinking off for the writing turn; require passage IDs as citations | Only the content varies between formats | 3 |
| C2 | Written summaries (short, medium, long) | Write the prompt; enforce a hard token cap; require citations; check how often outputs hit the cap | Formats 1 to 3 | 5 |
| C3 | Structured claims | Define the JSON layout following PACT (what was searched, claims, evidence, status); validate it; decide what to do with invalid JSON | Format 4 | 5 |
| C4 | Extracted quotes | Copy the passages the subagent flagged; fall back when none are flagged; cap the size | Format 5, with zero writing tokens | 4 |
| C5 | Raw documents | Copy everything the subagent read, grouped by document; fix the order; cap the size | Format 6, the largest return | 3 |
| C6 | Budget calibration | Measure typical sizes from the released BrowseComp-Plus trajectories and the pilot; set caps so the formats span a useful range | Final budgets per format | 6 |
| C7 | Delivered evidence | Record which documents reached the orchestrator under each format | The evidence recall metric | 3 |

### D. Evaluation (19 hours)

| # | Task | Steps | Output | Hours |
| --- | --- | --- | --- | --- |
| D1 | Grader | Run the official BrowseComp-Plus judge; export answers in its format; hand-check 30 judgments | Trusted accuracy scores | 6 |
| D2 | Evidence recall | Compare retrieved and delivered documents against the labels | How much evidence each format loses | 3 |
| D3 | Analysis notebook | Compute every metric per configuration; run paired tests; draw accuracy against tokens and the token breakdown by phase | The results section | 10 |

### E. Experiments (31 hours)

Hours here are people's time to launch, monitor and review runs, not model time.

| # | Task | Steps | Output | Hours |
| --- | --- | --- | --- | --- |
| E1 | Integration test | Run 5 questions through every configuration; check token totals add up across phases | A harness that works end to end | 4 |
| E2 | Baseline | Run a single agent on 50 questions; compare with published accuracy | Setup validated; model choice confirmed | 3 |
| E3 | Pilot | Run 50 questions across every configuration; review failures; lock budgets and N values | Design locked for the mid-term | 8 |
| E4 | Main runs | 200 questions × every configuration × 2 seeds; key configurations on all 830; rerun failures | The main results | 12 |
| E5 | Free-choice check | Run a subset with the orchestrator choosing N under a cap | Whether results hold when N is not forced | 4 |

### F. Writing and coordination (80 hours)

| # | Task | Steps | Output | Hours |
| --- | --- | --- | --- | --- |
| F1 | Update the proposal for tokens | Revise the title, hypotheses H2 and H3, the metrics and the success target | A report that matches the study | 2 |
| F2 | Mid-term report and talk | Completed and open tasks, pilot results, problems and fixes, updated timeline, who did what; 5-minute talk | Nov 5 deliverable | 12 |
| F3 | Final report | Conference-style write-up: motivation, related work, method, results, discussion; proofread | Dec 9 deliverable | 30 |
| F4 | Final talk | 15-minute slides and two rehearsals | Talk between Dec 1 and Dec 8 | 10 |
| F5 | Code release | README, cleanup, and published result traces | Dec 9 deliverable | 6 |
| F6 | Team meetings | About 2.5 hours a week across the team for eight weeks | Coordination | 20 |

## Hours by area

&#91;embedded content: our estimates · sums of the task tables above\]

The final report is the primary deliverable, so writing gets the largest share. The buffer mostly protects the harness, the riskiest engineering: if it slips, the pilot and the mid-term slip with it.

## Order and parallel work

**Start with A1, B1 and C1.** They set the repo, the data contracts and the format rules that every other task builds on. Once they are agreed, four tracks run in parallel:

| Person | Track | Tasks | Hours |
| --- | --- | --- | --- |
| Person 1 | Agents and written formats | B4, B5, B6, B8, C2, C3 | 37 |
| Person 2 | Infrastructure and token accounting | A1, A3, B1, B2, B7, B9, B10, B11 | 36 |
| Person 3 | Data, search and grading | A2, B3, C6, D1, D2, E2 | 30 |
| Person 4 | Format rules and analysis | A4, C1, C4, C5, C7, D3 | 29 |
| Everyone | Runs and writing | E1, E3, E4, E5, F1 to F6 | 108 (about 27 each) |

Person 1 can build the search loop against the mock model from B11 while the endpoint and search tool are still in progress.

**By the mid-term (Nov 5):** all of A, B and C; D1 and D2; the integration test, baseline and pilot (E1 to E3); and F1 and F2.

**By the final report (Dec 9):** the main runs and free-choice check (E4, E5), the analysis (D3), and the final report, talk and code release (F3 to F5).

If the first three weeks get too heavy, two tasks can move after the mid-term without blocking the pilot: the shared search cache (B8) and budget calibration (C6), with the pilot run on default budgets.

## Decisions to settle

Each is small, but it changes what a task builds, so settle it before that task starts.

- **Action protocol (B5):** the model's built-in tool calling, or plain JSON actions? JSON is simpler and works the same across models; tool calling is closer to how real systems run.
- **Subagent context (B5):** does each subagent see only its subtask, or the full question too? Seeing the full question makes the split less clean.
- **Thinking tokens (B9, C1):** keep thinking on during search, and report thinking tokens as their own output category, since they can dwarf the visible text.
- **Cached input tokens (B9):** count every input token, and also report how many were served from cache, since repeated prefixes are cheaper in practice.
- **Size caps (C4, C5):** how large can extracted quotes and raw documents get?
- **Re-delegation (B6):** how many extra rounds are allowed: one, or two?
- **Model access (A3):** a hosted API or self-hosted vLLM. Both report token counts; the choice mostly affects cost and setup time.

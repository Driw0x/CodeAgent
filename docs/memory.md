[English](memory.md) | [Français](memory.fr.md) | [简体中文](memory.zh-CN.md)

<a id="codeagent--project-memory"></a>
# CodeAgent — Project Memory


[Project README](../README.md)


> Project summary: features, decisions, results, and limitations from M1 to M8.

<a id="état-actuel"></a>
## Current Status

**M1–M7 completed. M8 PASS within its original controlled-planning scope; two real autonomous scenarios validated with `OLLAMA_MAX_LOADED_MODELS=1`. Current HITL qualification remains partial. M9 not started.** See the [final validation report](#m8-validation) for evidence and scope.

CodeAgent analyzes Python projects, indexes their code, generates source-based responses with citation verification and grounding, maintains the index, and uses read-only tools via MCP.

M7 introduces persistent conversational memory and shared context budget. M8 adds conditional planning and controlled execution with replanning.

<a id="m1--lecture-et-extraction"></a>
## M1 — Code Reading and Extraction

The first milestone established the foundation of the project:

- recursive traversal of Python files;
- filtering of irrelevant directories;
- reading of content;
- management of relative paths;
- initial unit tests.

Key decision: maintain a simple, LLM-independent pipeline to ensure repository analysis remains deterministic.

<a id="m2--recherche-sémantique"></a>
## M2 — Semantic Search

The code was subsequently decomposed using Python AST into chunks primarily representing functions, classes, variables, and imports.

Each chunk preserves its origin:

- file;
- type;
- name;
- start and end lines.

The chunks are encoded with `all-MiniLM-L6-v2` and indexed in FAISS.

Semantic search was then improved with a lexical reranking applied to the candidates returned by FAISS.

Pipeline achieved:

```text
Projet Python
→ AST
→ chunks
→ embeddings
→ FAISS
→ candidats
→ reranking lexical
→ chunks pertinents
```

<a id="m3--llm-local-et-rag"></a>
## M3 — Local LLM and RAG

M3 added response generation over code.

Prior to integration, several local LLMs were compared using a common benchmark to select a model suitable for the project.

Model selected:

- `Qwen2.5-Coder 14B`
- executed locally with Ollama.

The final RAG pipeline is:

```text
Question
→ embedding
→ FAISS search
→ reranking
→ chunk selection
→ context construction
→ Qwen2.5-Coder 14B
→ answer with provenance
```

The context passed to the LLM retains the chunk metadata, enabling responses that indicate the files and lines used.

Example:

```text
app/parser/file_loader.py:6-9
```

<a id="m4--mémoire-projet"></a>
## M4 — Project Memory

M4 added persistent and incremental memory to the project to avoid fully reconstructing the index on each execution.

Features added:

- persistent saving of the FAISS index;
- saving of associated chunks;
- automatic reloading of existing index;
- history of questions and answers;
- creation of a project manifest;
- detection of added, modified, and deleted files;
- incremental updates of embeddings and index;
- retention of relative paths to identify project files.

The manifest associates each file with a SHA-256 hash of its content. Two successive project states can thus be compared to precisely determine the changes.

Update pipeline:

```text
Python project
→ file reading
→ current manifest
→ comparison with previous manifest
→ added / modified / deleted files
→ recompute affected chunks only
→ update FAISS index
→ save index and new manifest
```

When no changes are detected, CodeAgent directly reuses the already saved index.

The analyses are also recorded in a JSONL history containing the date, the question, and the generated response.

<a id="m5--rag-quality--grounding"></a>
## M5 — RAG Quality & Grounding

M5 has strengthened and evaluated the RAG pipeline with:

- a retrieval benchmark using Hit@K, Recall@K, Precision@K, and MRR;
- a hybrid dense and lexical search;
- an improved reranking and context selection;
- a strengthened prompt to reduce hallucinations;
- automatically verified citations `[Sx]`;
- a claim-by-claim verification of response support by sources.

On the final holdout of 16 questions, M5 improves `Hit@5` and `Recall@5` from `0.6667` to `0.8333`, as well as the response/abstention behavior of `0.6250` to `0.7500`. The `MRR@5` decreases from `0.5069` to `0.3917` and the grounding verification adds on average `2.54 s` to the pipeline.

<a id="m6--mcp--tool-use"></a>
## M6 — MCP & Tool Use

M6 adds controlled tool usage via MCP:

- read-only tools `read_file`, `search_code`, `list_files`, `get_git_diff` and `run_tests`;
- tool calling with `Devstral 24B`;
- a `LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM` loop;
- timeout, step limits, retry limits, duplicate call detection, and structured error handling;
- reuse of M5 grounding on responses based on tool observations;
- per-question tracking of Devstral / grounding tokens, steps, tool calls, step times, and individual tool times;
- a dedicated final generation from source observations, with citation validation and grounding;
- context reduction for tools using bounded search results, line-range targeted reading, and consolidation of redundant sources.

The final end-to-end benchmark of 20 tasks achieves `80 %` success rate, with `100 %` routing success, `100 %` tool selection precision, `100 %` no-tool precision, and `100 %` argument precision. The 20 tasks produce 13 accepted grounded responses, 4 rejections `unsupported`, and 0 `citation_error`. The average latency remains high (`131.05 s`) and is primarily driven by local LLMs, while the tools themselves are now fast.

On the fixed grounding benchmark of 40 balanced cases, `Qwen2.5-Coder 14B` achieves `97.5 %` accuracy, `100 %` recall on unsupported cases, 0 false accepts, and 1 false reject. `Devstral 24B` achieves `90 %` with 0 false accepts and 4 false rejects; thus, `Qwen2.5-Coder 14B` remains the default grounder. On the gating benchmark of 20 tasks, Devstral achieves `100 %` precision. Laya and Verdict achieve `75 %`, ModernBERT-NLI achieves `65 %`, and GLiNER2.5 achieves `45 %`. Jev could not be evaluated as the external API remained inaccessible despite a compliant call to the documentation. The Verdict -> Devstral cascade reduces latency by `42.9 %` but decreases precision from `100 %` to `93.3 %`; it is not retained.

The final optimization of the context reduces the multi-tool smoke test token count from approximately `10.5k` to `6.6k` without loss of grounding. `search_code` has also been optimized by avoiding traversal of ignored folders prior to filtering: its observed time in smoke tests drops from approximately `17.1 s` to `0.01-0.12 s`.

<a id="m7--memory--context-management"></a>
## M7 — Memory & Context Management

M7 completed for memory and context; the reliability of autonomous finals remains distinct.

- `ConversationMemory` preserves recent context and automatically compresses older turns into ordered and prioritized excerpts, without using an LLM for summarization.
- The append-only archive `history.jsonl` preserves original exchanges and observations; atomic checkpoints `sessions/<session_id>.json` enable recovery and reconstruction from the JSONL in case of corruption.
- Future version or ahead-of-archive checkpoints are rejected without overwriting. Conversational memory remains separate from the project index.
- `ContextBudget` shares context among memory, RAG, tools, and LLM, with conservative generation and byte estimation in UTF-8; a single indispensable element that is too large triggers an explicit rejection.
- The resolved project path and `--session` isolate conversations, including those between homonymous projects.
- Current sources take precedence over history; selection and compression preserve provenance, paths, lines, and values. Final outputs and grounding use the same retained sources.
- An old exchange is not proof of the current code: a modified source must be re-read. Errors only prove their diagnosis.

The [deterministic memory benchmark](../benchmarks/memory/results/m7.json) validates **9/9 recalls and 6/6 controls**: bounded context, recall, extractive fidelity, sources/grounding, retrieval, and absence of summarization model. Historical real tests preserve **7/7 retrieved `read_file` readings, but 0/7 fully autonomous finals**, particularly for omitted citations or retrieval; these figures are not a replay after M8 corrections. The retrieval of old exchanges remains lexical and linear; semantic search remains optional. A bounded summary may exclude excerpts, whose originals remain archived. A session assumes a single writer process, without automatic archive purging. Memory does not guarantee faithful model recall nor correctly sourced final outputs.

<a id="m8--planning--controlled-autonomous-execution"></a>
## M8 — Planning & Controlled Autonomous Execution

**PASS within the original milestone scope.** M8 adds conditional short plans, minimal execution state, real MCP steps, bounded recovery of remaining work, preserved progress, explicit stops and comparison against direct M6 execution. The original objectives are in `README.md` at Git `0bf0437`. Optional HITL has separate validation limits below; no general autonomous reliability is claimed.

<a id="m8-contract"></a>
### Architecture and contracts

- `app/agent/planning.py` owns lexical routing, plan validation, state and evidence certificates. `ToolLoop` reuses the unique M6 MCP loop, M7 memory and M5/M6 citations/grounding. Simple requests remain direct; `--planning auto` is default, `off` disables planning, `adaptive` remains experimental.
- The planner returns JSON `{goal, steps}` or `{"stop":"user_decision","reason":"..."}`. Plans contain 2–6 steps with exactly `id`, `objective`, `depends_on`, `success`; IDs are unique, dependencies point only backward, independent steps may use `[]`.
- Objectives preserve unambiguous request excerpts and file paths, with bounded verb case/accent normalization. All collection actions must be covered. Coordinated multi-file reads use the same full objective, one step per file and explicit `success.path` selections.
- `success` allows only `tool`, `kind`, `value`, `from_step`, `path`; tools are `read_file`, `search_code`, `list_files`, `get_git_diff`, `run_tests`; conditions are `contains`, `nonempty`, `observed`. Discovered paths must come from completed dependencies referenced by `from_step`; invented paths are rejected.
- `observed` records an actual result, including empty searches/diffs or failed tests, without certifying final success. Errors/timeouts cannot complete a step. A requested implementation requires its complete body, not a signature or template.
- Replanning replaces only remaining work and preserves completed steps/evidence. One validation repair per generation shares the replanning allowance. Invalid grouped calls receive one bounded reminder and never execute before validation. Retries, deduplication and all budgets remain shared.
- Sources retain original observation IDs, files, lines and arguments. Selection can renumber `[Sx]`; `source_mapping` links final citations to observations. Current evidence precedes history; human instructions never become evidence. Final generation and grounding use the same retained sources.
- `goal_achieved` requires completed collection, valid citations, grounding, requested format and coverage, with no unjustified abstention. Explicit pytest counts/return codes are compared with cited results. Collection, `COMPLETED` or `SUPPORTED` alone does not establish final success.

<a id="m8-hitl"></a>
### HITL and recovery

`--hitl off` is default. `on-failure` pauses after observable planning, step or finalization rejection. The same task retains MCP connection, sources, completed steps, errors, deduplication and counters; one M7 turn is archived at completion.

| Menu | Action |
|---|---|
| 1–4 | Inspect bounded/masked state; provide guidance; retry within quotas; stop with an unvalidated partial report. |
| 5–7 | Select an observed path; provide a project path while retaining other arguments; execute an existing read-only tool with validated JSON arguments. |
| 8 | Patch identifiable remaining steps, optionally remove one with `{"remove":true}`, and atomically validate the complete plan before action. |

Completed steps cannot be rewritten. Permissions, JSON Schema, project boundaries, dependencies, provenance, duplicate checks, citations and grounding remain enforced. Direct actions consume existing call/iteration quotas. Limits: three interventions per task, 2,000 characters per guidance, no reset. Human waiting counts toward the deadline; synchronous input checks expiry when input returns. Without a terminal, return `intervention_required`.

A rejected final allows one additional generation with the same sources and no new collection. A grounding exception allows one verification retry of the same candidate/sources, distinct from `UNSUPPORTED`. `resumed` proves authorization only: inspect later calls and the final outcome. Conversation restoration after restart does not resume an interrupted execution.

<a id="m8-corrections"></a>
### Causes and retained corrections

| Demonstrated cause | Retained correction |
|---|---|
| Verb casing/NFC/NFD accents, file names mistaken for actions, unnecessary dependencies | Bound action recognition without changing offsets/paths; permit independent steps; provide step diagnostics and exact multi-file repair excerpts. |
| Wrong test file, inconsistent process success, collection mistaken for completion | Validate requested pytest paths before execution/certification, require coherent process status and retain coverage/false-success checks. |
| Corrected search discarded; signature/template substituted for body | Preserve recovery observations and separate diagnostics from content; use `total_lines`, AST completeness and locally cited faithful code delivery. |
| Final prompt demanded an unasked remainder or tool narrative | Pass validated plan/evidence, request factual locally cited statements and reject unjustified abstention before grounding. |
| H: empty `/api/chat` despite a generated tool call | Raw replay exposed an out-of-step `read_file` lost by Ollama 0.40.8 parsing. Place the current step after observations; retain tool filtering and validation. |
| Empty-response metrics lost; Qwen loading timeout with Devstral resident | Record statistics before content rejection, retain `done_reason`; native `OLLAMA_MAX_LOADED_MODELS=1` on the measured machine, without an application model manager or timeout increase. |

<a id="m8-validation"></a>
### Validation and scope

**Final audit: 799 tests passed**, including **509 affected tests**; memory **6/6**, planning **6/6** across **16 scenarios / 32 executions**; syntax, imports, CLI options and `git diff --check` pass. No functional application/script change in this consolidation: historical real evidence remains applicable to the runtime; test files are reorganized. Deterministic tests use scripted models/grounding with real memory/stdio MCP and real pytest. They verify invariants, not Ollama autonomy.

The real final campaign on **23 October 2026** uses Ollama **0.40.8**, `devstral:24b`, `qwen2.5-coder:14b`, temperature **0**, seed **42**, context **16384**, generation limit **512**, HTTP timeout **300 s**, `keep_alive=30m`, `OLLAMA_MAX_LOADED_MODELS=1`. Shared budgets: **12 iterations, 6 steps, 2 replans, 16 MCP calls, 32768 tokens, 900 s**, tool timeout **60 s**. MCP is a real stdio subprocess; no model response or grounding verdict is scripted.

| Real scenario | Verified outcome |
|---|---|
| Main, `main-final`, HITL `on-failure` | **PASS AUTONOMOUS**, **3 MCP**, **248.454 s**, zero interventions/duplicates/final retries; one autonomous plan repair. Both files read completely, actual pytest, coverage/citations/format/grounding valid; independent review **13/13**. |
| Historical H unchanged, `H-final`, HITL `off` | **PASS AUTONOMOUS**, **4 MCP**, **274.461 s**, zero interventions/replans/duplicates. Implementation and tests read; `marker` returns 42 and `assert marker() == 42` correctly cited; independent review **13/13**. |

Exact main request:

> Lis app/agent/planning.py et tests/test_planning.py, puis exécute les tests de tests/test_planning.py. Fournis un bilan factuel en citant les fichiers consultés et le résultat réel des tests.

At that revision, reads cover lines **1–596** and **1–505**. MCP: `131 passed, 1 warning in 0.74s`, return code **0**, `passed=true`, `timed_out=false`, empty stderr. Independent pytest: `131 passed, 1 warning in 0.69s`. The existing `cache_dir` warning comes from disabling pytest's cache plugin. These are historical counts before test reorganization. Delivered citations S1/S2 identify both files, S3 the actual pytest result.

**HITL: PARTIAL qualification on the current revision.** Deterministic interruption/recovery tests pass. Historical **18 October** Ollama + in-memory MCP trials include **PASS ASSISTED F** (3 MCP, 2 interventions, **258.028 s**) and **PASS ASSISTED E2** (3 MCP, 3 preauthorized actions, **552.317 s**). Recorded chat decisions supply operator actions/guidance after snapshots, never replacement model responses/plans/finals. These bounded earlier-revision successes have partial reproducibility: later F trials and assisted H finalization fail. The October 23 assisted collection/refinalization preserves evidence but fails the final answer. Zero-intervention final successes do **not** validate a new complete interruption/resumption cycle on current code.

**Experimental decisions:** adaptive/global pilots each achieve **1/4** strict success (14541 / 28720 tokens): **KEEP GLOBAL**. Guided finalization delivers **3/10** in both variants for **10.44%** extra tokens: **KEEP CURRENT**, guidance/revision prototypes removed. Native JSON Schema improves raw JSON, not strict success (**1/4** each; 14836 / 16012 tokens, **+7.93%**): **KEEP CURRENT**, prototype/15 tests removed, no production `--planning-format`. Schema structure did not certify provenance/semantics; `format` with native `tools` was not used. Unexecuted holdouts are neither successes nor failures. Details: [adaptive review](../benchmarks/planning/results/autonomy-20261017-review.json), [finalization review](../benchmarks/planning/results/finalization-20261017-review.json), [structured review](../benchmarks/planning/results/structured-20261017-review.json), [structured protocol](../benchmarks/planning/structured-protocol.json).

<a id="m8-reproduction"></a>
### Reproduction and evidence

From the repository root, use installed dependencies/models and new outputs outside the repository:

```powershell
$py = '.venv/Scripts/python.exe'
$out = Join-Path $env:TEMP ('codeagent-m8-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
& $py -m pytest -q
& $py -m scripts.benchmark_memory --output (Join-Path $out 'memory.json')
& $py -m scripts.benchmark_planning --output (Join-Path $out 'planning.json')
& $py -m app.main --project . --planning auto --hitl on-failure --debug --session ('m8-' + [guid]::NewGuid())
```

Choose **2**, submit the exact request above. For H and optional assisted failure scenarios, see [planning instructions](../benchmarks/llm/README.md#memory-and-planning-protocols). Benchmark real runs require `--real`; `app.main` above uses Ollama directly without this flag. Scripted replays remain distinct. On the measured machine, start installed Ollama with process-local `$env:OLLAMA_MAX_LOADED_MODELS='1'`. Preserve existing services, restore initial state afterward and unload only models/processes created for testing. The historical temporary server was stopped, zero Ollama processes remained; no persistent setting changed.

Local final evidence, outside the repository:

```text
E:/AI/Codex/visualizations/2026/10/23/01a16a21-572d-7520-b967-5111912d7a82/m8-final
```

`main-final/`, `H-final/`: protocols, HTTP requests/responses, MCP traces, sources, timings, results, independent reviews. `tested-source/`, hashes, `session-changes.patch`: validated revision. `final-suite.txt`, `final-memory.json`, `final-planning.json`, `final-check-exits.json`: historical checks. `diagnostic.json`, `H-raw-exact.json`, `H-corrected/`: parser/loading failures. Archived `run_campaign.py`, `review_run.py`, `review_history.py` reproduce/review with new output directories. Local artifacts are not included in a clone.

Earlier proofs remain under `data/memory/m8-autonomy-20261017`, `m8-finalization-20261017`, `m8-structured-20261017` (including `prototype.zip`, `prototype-files.json`, `manifest.json`), `m8-hitl-20261017`, `m8-hitl-executable-20261018/final-delivery`, `m8-hitl-validation-20261018-853fc1c4/E-02`; the last two preserve assisted successes, decisions and reviews. October 23 failed audits remain in visualization folders `01a169cb-76e8-7e91-aadc-6510601947a4/m8-hitl-audit` and `01a16a07-8d01-7f31-9519-9b7b30be1a38/m8-final`. Historical failures are not rewritten as successes.

<a id="m8-limitations"></a>
### Remaining limits

The two successes validate those scenarios/configuration, not general autonomy or unseen-task generalization. Lexical routing, proof certificates and format checks are bounded; grounding may falsely accept/reject semantic claims. One initial plan still needs autonomous repair. Current recovery is verified deterministically; a complete current assisted Ollama cycle is not revalidated. Loading depends on hardware; a real grounding timeout without the one-model limit remains preserved. Human waiting may exhaust the wall-clock budget; synchronous input and execution restart remain limited as above. No second MCP loop, writing tool, framework or M9 implementation is added.

<a id="décisions-techniques-principales"></a>
## Main technical decisions

- local operation;
- Python AST for code structuring;
- `all-MiniLM-L6-v2` for embeddings;
- FAISS as vector index;
- lexical reranking after vector search;
- Ollama for local LLM execution;
- `Qwen2.5-Coder 14B` selected after benchmark;
- clear separation between retrieval and generation;
- systematic preservation of file/line provenance;
- local persistence of the FAISS index and chunks;
- SHA-256 manifest to detect project changes;
- incremental updates to recalculate only modified files;
- historical analysis records stored in JSONL;
- hybrid dense + lexical search;
- automatic validation of citations;
- grounding verification with the local LLM;
- MCP to expose and execute read-only tools;
- `Devstral 24B` retained for tool calling;
- `Qwen2.5-Coder 14B` retained for grounding M6 after dedicated benchmark;
- preloading Ollama and `keep_alive` to limit cold starts when models can remain resident;
- specialized routers evaluated but not integrated, Devstral remains more accurate.

<a id="état-du-projet-et-suite"></a>
## Project status and next steps

M1–M7 completed. M8 PASS in its original scope, with the validation limits above. M9 — Advanced RAG & Reliability remains future; further work requires measured benefit.

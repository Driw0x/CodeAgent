[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

<a id="codeagent"></a>
# CodeAgent


Personal project to create a local AI agent for Python code analysis and project memory.

<a id="présentation"></a>
## Introduction

CodeAgent is a local AI assistant designed to analyze Python projects and answer questions about their code.

The objective of the project is to create an agent capable of:

- automatically reading Python files in a project,
- indexing the code with embeddings,
- understanding the overall structure of the project,
- answering questions about the code,
- maintaining a simple memory of the project,
- using tools in a controlled manner when necessary,
- planning multi-step tasks when it provides real benefit.

The project operates entirely locally using a local LLM and a vector database.

<a id="statut-du-projet"></a>
### Project Status

**M1–M7 completed · M8 implemented · M9 future**

**M8 — Deterministic checks and two autonomous Ollama + MCP scenarios validated with `OLLAMA_MAX_LOADED_MODELS=1`.**

The core of CodeAgent is complete and functional: code analysis, hybrid search, RAG pipeline, source-based responses, index persistence, incremental updates, grounding, and controlled tool usage via MCP.

Memory and context budgets are covered by the [M7 synthesis](docs/memory.md#m7--memory--context-management). The [M8 synthesis](docs/memory.md#m8--planning--controlled-autonomous-execution) documents planning, validation scope and remaining limitations.

---

<a id="fonctionnalités-actuelles"></a>
## Current Features

<a id="analyse-du-code-python"></a>
### Python Code Analysis

* Recursive reading of Python files.
* Filtering of unnecessary directories (`.venv`, `__pycache__`, etc.).
* Extraction of file content.
* Displaying relative paths within the project.

<a id="parsing-ast"></a>
### AST Parsing

Automatic extraction of key code elements:

* Functions (`FunctionDef`)
* Classes (`ClassDef`)
* Variables (`Assign`, `AnnAssign`)
* Imports (`Import`)
* Specific imports (`ImportFrom`)

Each element is converted into a chunk containing:

```python
{
    "file": ...,
    "type": ...,
    "name": ...,
    "content": ...,
    "start_line": ...,
    "end_line": ...
}
```

<a id="recherche-sémantique"></a>
### Semantic Search

* Dividing the project into chunks.
* Generating embeddings with `all-MiniLM-L6-v2`.
* Storing vectors in a FAISS index.
* Generating embeddings for user queries.
* Searching for the k most relevant chunks.
* Semantic search of candidate chunks with FAISS.
* Hybrid dense and lexical search.
* Lexical reranking of candidate chunks.
* Selection of the most relevant chunks for the RAG context.

Examples of queries:

* "function that reads file content"
* "function that reads directory"

---

<a id="architecture"></a>
## Architecture

```text
app
├── agent
├── llm
├── memory
├── parser
├── rag
├── retrieval
├── tools
├── main.py
└── mcp_server.py
```

---

<a id="technologies-utilisées"></a>
## Technologies used

<a id="parsing"></a>
### Parsing

* Python AST

<a id="embeddings"></a>
### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

<a id="base-vectorielle"></a>
### Vector database

* FAISS

<a id="llm-local"></a>
### Local LLM

* Ollama
* Qwen2.5-Coder 14B
* Devstral 24B for tool calling

<a id="rag"></a>
### RAG

* FAISS
* Hybrid dense and lexical search
* Lexical reranking
* Context injection with file/line provenance
* Citations `[Sx]` and grounding verification

<a id="tests"></a>
### Tests

* Pytest

---

<a id="installation"></a>
## Installation

<a id="1-cloner-le-projet"></a>
### 1. Clone the project

```bash
git clone <repo>
cd CodeAgent
```

<a id="2-installer-les-dépendances-python"></a>
### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

<a id="3-installer-ollama"></a>
### 3. Install Ollama

Install Ollama on the machine, then verify that the installation works :

```bash
ollama --version
```

<a id="4-télécharger-le-llm-local"></a>
### 4. Download the local LLM

CodeAgent uses `qwen2.5-coder:14b` for RAG / grounding and `devstral:24b` for tool calling.

Download the model with :

```bash
ollama pull qwen2.5-coder:14b
ollama pull devstral:24b
```

Verify that the model is available :

```bash
ollama list
```

---

<a id="exécution-du-projet"></a>
## Execution of the project

From the root of the project, launch CodeAgent with :

```bash
python -m app.main
```

The program then asks a question about the project :

```text
Question : Comment les embeddings sont-ils ajoutés dans FAISS ?
```

CodeAgent offers a RAG mode and an Agent mode. The Agent mode reuses the MCP M6 loop, adds a plan M8 when the request contains explicit dependencies, preloads local models, and displays per question the Devstral / grounding tokens, step times, tool calls, their duration, grounding status, and response time.

To observe M8 in PowerShell, select **2** after this command :

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main --project . --session m8-validation --debug
```

`--planning off` compares with M6 directly. `--max-steps`, `--max-plan-steps`, `--max-replans`, `--max-tool-calls`, `--max-total-tokens`, `--max-seconds` and `--tool-timeout` bound the execution. The [M8 synthesis](docs/memory.md#m8--planning--controlled-autonomous-execution) presents the results and limitations of this planning.

<a id="assistance-humaine-m8-optionnelle"></a>
### Human oversight M8, optional

```powershell
python -m app.main --project . --session hitl-test --debug --hitl on-failure
```

Select **2** (Agent). After an autonomous attempt fails, HITL can pause to inspect evidence, correct a tool call or remaining plan, retry, or stop. Resumption preserves the task, completed work, evidence and global budgets. Human input is validated and never treated as source evidence; noninteractive execution returns `intervention_required`.

Deterministic tests cover interruption and resumption. The latest real autonomous scenarios needed no intervention, so those runs alone do not validate a HITL cycle. See the [HITL contract](docs/memory.md#m8-hitl) and [validation evidence](docs/memory.md#m8-validation).

To resume a conversation, use the same session identifier and the same project:

```bash
python -m app.main --session analyse-1
python -m app.main --project /chemin/vers/projet-python --session analyse-1 --context-tokens 16384
```

Choose a different `--session` for an independent conversation. The budget preserves recent exchanges, automatically compresses older ones into deterministic snippets, and selectively retrieves the JSONL archive. Original observations remain available; previous citation identifiers are not reused as current sources. The conservative estimate uses UTF-8 bytes, not a model tokenizer.

<a id="vérifications-sans-llm-réel"></a>
### LLM-free validations

```powershell
$m8CheckDirectory = Join-Path $env:TEMP ('codeagent-check-' + [guid]::NewGuid().ToString())
python -m pytest -q
$env:HF_HUB_OFFLINE = '1'
python -m scripts.benchmark_rag --output-dir (Join-Path $m8CheckDirectory 'rag')
```

See the [benchmark protocols](benchmarks/llm/README.md#memory-and-planning-protocols) for memory and planning checks with temporary output files.

The memory benchmark does not download anything. The RAG benchmark requires MiniLM already available locally (use `HF_HUB_OFFLINE=1` to disable network access). `pytest.ini` and `conftest.py` place pytest cache and `tmp_path` files into system temporary directories, with fixture cleanup after use.

---

<a id="documentation"></a>
## Documentation

- [Project memory, context, and planning — M1–M8, results and validation](docs/memory.md)
- [M5 — RAG Quality & Grounding](docs/rag_quality.md)

<a id="roadmap"></a>
# Roadmap

<a id="milestone-1--lecture-et-extraction-du-code"></a>
## Milestone 1 — Code reading and extraction

<a id="objectifs"></a>
### Objectives

* [x] Project initialization
* [x] Recursive file reading in Python
* [x] Path management
* [x] Ignore useless files
* [x] Content extraction
* [x] Unit test implementation

<a id="résultat"></a>
### Result

The project is capable of automatically traversing a Python repository and extracting the necessary information for analysis.

---

<a id="milestone-2--embeddings-et-recherche-sémantique"></a>
## Milestone 2 — Embeddings and semantic search

<a id="objectifs-1"></a>
### Objectives

* [x] Code chunking
* [x] AST parsing
* [x] Generation of embeddings
* [x] Vector indexing
* [x] Search for nearest neighbors
* [x] User query embedding
* [x] Functional semantic search

<a id="résultat-1"></a>
### Result

CodeAgent is capable of automatically retrieving the most relevant code portions from a natural language question.

---

<a id="milestone-3--rag-sur-le-code"></a>
## Milestone 3 — Code RAG

<a id="benchmark-llm"></a>
### LLM Benchmark

* [x] Define a common benchmark to compare local LLMs
* [x] Compare several local LLMs
* [x] Select and integrate a local LLM

<a id="pipeline-rag"></a>
### RAG Pipeline

* [x] Build the RAG pipeline
* [x] Inject retrieved chunks into the context
* [x] Answer questions about the project
* [x] Generate code explanations
* [x] Reference files and lines involved

<a id="exemple"></a>
### Example

Question :

> Where is the function that reads files defined?

Answer :

> The function `read()` is defined in `app/parser/file_loader.py`, lines 6-9.

Sources:
- `app/parser/file_loader.py:6-9`

---

<a id="milestone-4--mémoire-projet"></a>
## Milestone 4 — Project Memory

<a id="objectifs-2"></a>
### Objectives

* [x] Persistent storage of indexes
* [x] History of analyses
* [x] Incremental update of embeddings
* [x] Tracking project modifications

---

<a id="milestone-5--rag-quality--grounding"></a>
## Milestone 5 — RAG Quality & Grounding

<a id="objectifs-3"></a>
### Objectives

- [x] Build a RAG benchmark
- [x] Add Recall@K, Precision@K, and MRR
- [x] Evaluate the current retrieval as baseline
- [x] Add a hybrid dense + lexical search
- [x] Evaluate and improve reranking
- [x] Improve the context sent to the LLM
- [x] Strengthen the prompt against hallucinations
- [x] Add file and line citations
- [x] Automatically verify citations
- [x] Add a consistency check between source <=> response
- [x] Compare the final pipeline to baseline M4

<a id="résultat-2"></a>
### Result

On the final holdout of 16 questions, M5 improves `Hit@5` and `Recall@5` from `0.6667` to `0.8333`, as well as the response/abstention behavior of `0.6250` to `0.7500`. The observed trade-off is a `MRR@5` lower (`0.5069 → 0.3917`) and an average cost increase of `+2.54 s` due to grounding.

---

<a id="milestone-6--mcp--tool-use"></a>
## Milestone 6 — MCP & Tool Use

<a id="objectifs-4"></a>
### Objectives

- [x] Define a minimal interface for tools
- [x] Describe each tool with a name, a description, and a `JSON Schema`
- [x] Validate parameters before execution
- [x] Implement a first set of read-only tools: file reading, code search, file listing, Git diff inspection, and test execution
- [x] Let the LLM decide whether a tool is necessary and which one to use
- [x] Integrate tools via MCP
- [x] Set up the loop `LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM`
- [x] Add timeout, maximum number of steps, limited retries, and detection of identical tool calls `tool + arguments`
- [x] Validate tool results and return structured errors to the LLM
- [x] Reuse the grounding M5 for the final response when project sources are used
- [x] Evaluate tool use on a dedicated benchmark
- [x] Track tokens Devstral / grounding, steps, tool calls, time per step, and individual tool execution time by question
- [x] Optimize the tool context with bounded search, targeted reading, and consolidation of redundant sources
- [x] Evaluate grounding models separately on a dedicated holdout

<a id="implémentation"></a>
### Implementation

The initial scope remains read-only. Tools that modify project state will only be added after stabilization of the execution loop and must request explicit confirmation for high-risk operations.

The benchmark follows task success rate, invalid calls, retries, duplicated calls, tool errors, number of steps, grounding, tokens, and latency.

<a id="résultat-3"></a>
### Result

CodeAgent determines when a tool is necessary, selects and executes the appropriate tool via MCP, recovers from simple errors without uncontrolled loops, and generates a final response with sources before grounding validation.

On the final end-to-end benchmark of 20 tasks, routing, tool selection, no-tool, and argument handling achieve `100 %`, while end-to-end success achieves `80 %` (`13` grounded responses accepted, `4` rejections `unsupported`, `0` `citation_error`). The average is `2.10` steps and `1.15` tool calls per task; latency remains high at `131.05 s`, primarily due to local LLMs.

On a holdout grounding of 40 balanced cases, `Qwen2.5-Coder 14B` achieves `97.5 %` accuracy, `100 %` recall on unsupported cases, 0 false accept, and 1 false reject. `Devstral 24B` achieves `90 %` with 0 false accept and 4 false rejects; Qwen14 is therefore selected for grounding M6. Context optimization reduces the multi-tool smoke test by approximately `10.5k` to `6.6k` tokens, and optimization of `search_code` reduces its observed time by approximately `17.1 s` to `0.01-0.12 s`.

On the gating benchmark of 20 tasks, Devstral achieves `100 %` precision; Laya and Verdict `75 %`, ModernBERT-NLI `65 %`, and GLiNER2.5 `45 %`. Jev could not be evaluated as the external API remained inaccessible despite a compliant call to the documentation. A cascade Verdict -> Devstral tested on a holdout of 30 tasks reduces average latency by `42.9 %`, but achieves `93.3 %` precision against `100 %` for Devstral alone; it is therefore not retained.

<a id="future-improvement"></a>
### Future improvement

- Re-evaluate Jev when its API becomes accessible and consider future specialized routers / grounders only if they improve the precision/latency trade-off without increasing false accepts.

---

<a id="milestone-7--memory--context-management"></a>
## Milestone 7 — Memory & Context Management

<a id="objectifs-5"></a>
### Objectives

- [x] Maintain a short-term memory of the most recent conversation turns
- [x] Summarize and compress previous turns
- [x] Define a token budget for the context
- [x] Automatically trigger compression when the threshold is exceeded
- [x] Persist the summary between executions
- [ ] Optional: retrieve relevant past exchanges via semantic search
- [x] Separate sessions and projects, preserve decisions, constraints, errors, and provenance
- [x] Select structured observations before compression, retain originals for verification
- [x] Measure continuity, extractive fidelity, and context size on deterministic fixtures without an LLM

<a id="résultat-4"></a>
### Result

CodeAgent has a persistent, session- and project-isolated conversational memory with automatic compression and shared context budget. The operation is validated on deterministic fixtures, with **9/9 recall**. **M7 is complete** for memory and context; historical real tests recover **7/7 reads**, but only **0/7 fully autonomous final outcomes**.

---

<a id="milestone-8--planning--controlled-autonomous-execution"></a>
## Milestone 8 — Planning & Controlled Autonomous Execution

<a id="objectifs-6"></a>
### Objectives

- [x] Detect tasks that genuinely require multiple dependent steps
- [x] Generate a short plan only for such tasks
- [x] Maintain a minimal state: objective, completed steps, observations, failures, and remaining budget
- [x] Execute each step using the M6 tools loop
- [x] Replan only the remaining steps when an observation or error invalidates the plan
- [x] Preserve previously validated work during replanning
- [x] Stop execution when the objective is achieved, the budget is exhausted, or user input is required
- [x] Evaluate multi-step tasks by comparing the direct M6 loop with the M8 planning approach

<a id="résultat-5"></a>
### Result

Dependent tasks use a validated plan; simple tasks retain direct M6 execution. MCP observations verify steps, and replanning preserves completed work within the same budgets. Final answers are checked for requested deliverables, citations and grounding. The [M8 synthesis](docs/memory.md#m8--planning--controlled-autonomous-execution) centralizes the final results, reproduction settings and limitations. Native JSON Schema output was evaluated and not retained; the delivered planner validates its JSON contract in code.

---

<a id="milestone-9--advanced-rag--reliability"></a>
## Milestone 9 — Advanced RAG & Reliability

<a id="objectifs-proposés"></a>
### Proposed objectives

- [ ] Strengthen metadata, stable identifiers, versions, filters, and index/chunk consistency
- [ ] Compare BM25 and specialized rerankers against the existing lexical/RRF approach, using Hit@K, Recall@K, MRR, and nDCG
- [ ] Evaluate iterative retrieval and reformulation only when the initial retrieval fails
- [ ] Detect obsolete or contradictory sources and evaluate targeted corrections
- [ ] Compare an independent verifier/NLI against the existing grounding: false accepts, false rejects, P50/P95, and latency

<a id="future-improvements--research-ideas"></a>
### Future improvements / Research ideas

AST remains prioritized for Python. Recursive chunking, TreeRAG, GraphRAG, OCR, and multimodal approaches require a concrete corpus and a comparative protocol before integration. No M10 is currently justified. Multi-agent, independent Critic Agent, sophisticated reflection, and anti-hallucination fine-tuning remain deferred. FAISS and the current MCP loop are preserved. LangChain is not necessary; LangGraph will only be re-evaluated if resumptions and branches make the current workflows insufficient. No distributed or cloud system is planned.

---

<a id="état-actuel-du-projet"></a>
## Current project status

CodeAgent now has a complete pipeline enabling analysis of a Python project, indexing its code, retrieving relevant portions, generating context-aware responses using a local LLM, and maintaining the index up to date as the project evolves.

The RAG foundation is complete at M5 and tool use controlled via MCP is complete at M6. **M7 is complete** for memory and context management. The actual limits of routing and final generation are documented; they do not prove memory or source corruption in M7. M8 adds conditional planning and its observable controls; M9 remains future and will consolidate measured improvements in RAG and reliability.

<!-- multilingual-documentation -->

<a id="documentation-complémentaire"></a>
## Complementary documentation

- [benchmarks/llm/README.md](benchmarks/llm/README.md)

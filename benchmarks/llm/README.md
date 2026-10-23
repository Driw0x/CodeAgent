[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)


[Project README](../../README.md)

<a id="résultats-du-benchmark"></a>
## Benchmark results

Seven local models were evaluated using the same benchmark, the same Ollama runtime, and the same generation parameters.

|Model|Average latency|Average throughput|
| ----------------------- | --------------: | -------------: |
| `qwen2.5-coder:7b`      |3.22 s|86.71 tokens/s|
| `qwen2.5-coder:14b`     |4.13 s|45.21 tokens/s|
| `deepseek-coder-v2:16b` |5.85 s|36.26 tokens/s|
| `gemma3:12b`            |5.64 s|18.42 tokens/s|
| `devstral:24b`          |10.12 s|10.54 tokens/s|
| `qwen3-coder:30b`       |14.13 s|12.23 tokens/s|
| `llama3.1:8b`           |3.64 s|70.95 tokens/s|

<a id="modèle-sélectionné"></a>
## Selected model

The model selected for CodeAgent is:

```text
qwen2.5-coder:14b
```

The selection is based on benchmark performance measurements as well as manual validation of the test cases necessary for the RAG pipeline operation.

`qwen2.5-coder:7b` remains an interesting lightweight alternative when speed and resource consumption are prioritized.

The heavier models tested did not provide a sufficiently significant gain to justify their execution cost within the current CodeAgent context.

<a id="memory-and-planning-protocols"></a>
## Memory and planning protocols

Run from the repository root; generated reports go to a temporary directory:

```powershell
$benchmarkOutput = Join-Path $env:TEMP ('codeagent-bench-' + [guid]::NewGuid().ToString())
python -m scripts.benchmark_memory --output (Join-Path $benchmarkOutput 'memory.json')
python -m scripts.benchmark_planning --execution both --output (Join-Path $benchmarkOutput 'planning.json')
python -m scripts.benchmark_planning --real --execution auto --cases-file benchmarks/planning/autonomy-pilot.json --case H-tests --debug --output (Join-Path $benchmarkOutput 'planning-real.json')
```

Without `--real`, model decisions and grounding are scripted; MCP tools operate on temporary project files. These checks verify execution mechanics, evidence and budgets, not model quality. Planning modes: `off` uses M6 directly, `auto` uses M8 planning, `both` compares these modes; `adaptive` revises remaining steps locally, and `compare` compares `auto` with `adaptive`.

`--real` requires the existing Ollama service and installed models; it does not start the service. It uses real Ollama decisions and MCP tools in process on temporary fixtures, not the stdio transport. `--hitl on-failure` enables an interactive pause after a blockage; intervention is not forced. Automatic success checks require a separate factual review of actions, complete reads, answer, sources and grounding.

The JSON files in `benchmarks/planning/` define executable cases (`autonomy-pilot`, `autonomy-holdout`, `hitl-pilot`) or frozen experiment protocols (`finalization-protocol`, `structured-protocol`). Files under `results/` are generated reports or independent historical reviews, not runtime configuration. The native JSON Schema prototype was removed; `--planning-format` is unavailable in the delivered code. See [M8 contracts and evidence](../../docs/memory.md#m8--planning--controlled-autonomous-execution) for experiment decisions, real stdio validation and reproduction conditions.

[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="résultats-du-benchmark"></a>
## 基准测试结果

已使用相同的基准测试、相同的Ollama运行时和相同的生成参数，对七个本地模型进行了评估。

|模型|平均延迟|平均吞吐量|
| ----------------------- | --------------: | -------------: |
| `qwen2.5-coder:7b`      |3.22 秒|86.71 个标记/秒|
| `qwen2.5-coder:14b`     |4.13 秒|45.21 个标记/秒|
| `deepseek-coder-v2:16b` |5.85 秒|36.26 个标记/秒|
| `gemma3:12b`            |5.64 秒|18.42 个标记/秒|
| `devstral:24b`          |10.12 秒|10.54 个标记/秒|
| `qwen3-coder:30b`       |14.13 秒|12.23 个标记/秒|
| `llama3.1:8b`           |3.64 秒|70.95 个标记/秒|

<a id="modèle-sélectionné"></a>
## 选定模型

为CodeAgent选定的模型是：

```text
qwen2.5-coder:14b
```

该选择基于基准测试测得的性能以及对RAG管道运行所需测试用例的人工验证。

`qwen2.5-coder:7b`在速度和资源消耗优先的情况下仍是一个轻量级的有趣选择。

在当前CodeAgent的使用场景下，测试的更重模型并未带来足够显著的性能提升，以证明其运行成本是值得的。

<a id="memory-and-planning-protocols"></a>
## 记忆与规划协议

从仓库根目录运行；生成报告写入临时目录：

```powershell
$benchmarkOutput = Join-Path $env:TEMP ('codeagent-bench-' + [guid]::NewGuid().ToString())
python -m scripts.benchmark_memory --output (Join-Path $benchmarkOutput 'memory.json')
python -m scripts.benchmark_planning --execution both --output (Join-Path $benchmarkOutput 'planning.json')
python -m scripts.benchmark_planning --real --execution auto --cases-file benchmarks/planning/autonomy-pilot.json --case H-tests --debug --output (Join-Path $benchmarkOutput 'planning-real.json')
```

未使用 `--real` 时，模型决策与来源依据核验采用脚本，MCP 工具操作临时项目文件。这些检查验证执行机制、证据与预算，不衡量模型质量。规划模式：`off` 直接使用 M6，`auto` 使用 M8 规划，`both` 比较这两种模式；`adaptive` 在本地修订剩余步骤，`compare` 比较 `auto` 与 `adaptive`。

`--real` 需要现有 Ollama 服务与已安装模型，不会启动服务。Ollama 决策与进程内 MCP 工具调用均为真实执行，但使用临时 fixture，不经过 stdio 传输。`--hitl on-failure` 在阻塞后启用交互暂停，不强制人工介入。自动成功检查仍需独立核对事实，包括动作、完整读取、回答、来源与来源依据核验。

`benchmarks/planning/` 下的 JSON 定义可执行案例（`autonomy-pilot`、`autonomy-holdout`、`hitl-pilot`）或固定实验协议（`finalization-protocol`、`structured-protocol`）。`results/` 中的文件是生成报告或独立历史评审，不是运行配置。原生 JSON Schema 原型已移除；交付代码不提供 `--planning-format`。实验决策、真实 stdio 验证与复现条件见 [M8 合约与证据](../../docs/memory.zh-CN.md#m8--planning--controlled-autonomous-execution)。

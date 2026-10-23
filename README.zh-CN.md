[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

<a id="codeagent"></a>
# CodeAgent
一个用于Python代码分析和项目记忆的本地AI代理个人项目。

<a id="présentation"></a>
## 介绍

CodeAgent 是一个本地AI助手，专为分析Python项目代码并回答关于其代码的问题而设计。

本项目的目标是创建一个能够：

- 自动读取项目中的Python文件，
- 使用嵌入（embeddings）对代码进行索引，
- 理解项目的整体结构，
- 回答关于代码的问题，
- 保持一个简单的项目记忆，
- 在必要时以受控方式使用工具，
- 在真正带来效益时规划多步骤任务。

该项目完全在本地运行，依赖本地大语言模型（LLM）和向量数据库。

<a id="statut-du-projet"></a>
### 项目状态

**M1–M7 已完成 · M8 已实现 · M9 未来计划**

**M8 — 确定性检查与两个 Ollama + MCP 自主场景已在 `OLLAMA_MAX_LOADED_MODELS=1` 配置下验证。**

CodeAgent 的核心基础已完工并可运行：代码分析、混合搜索、RAG管道、源码响应、索引持久化、增量更新、事实验证以及通过MCP实现工具的受控使用。

[M7 汇总](docs/memory.zh-CN.md#m7--memory--context-management)介绍记忆与上下文预算。[M8 汇总](docs/memory.zh-CN.md#m8--planning--controlled-autonomous-execution)记录规划机制、验证范围与剩余限制。

---

<a id="fonctionnalités-actuelles"></a>
## 当前功能

<a id="analyse-du-code-python"></a>
### Python代码分析

* 递归读取Python文件。
* 过滤无用文件夹（`.venv`, `__pycache__` 等）。
* 提取文件内容。
* 显示项目相对路径。

<a id="parsing-ast"></a>
### AST解析

自动提取代码中的主要元素：

* 函数（`FunctionDef`）
* 类（`ClassDef`）
* 变量（`Assign`, `AnnAssign`）
* 导入（`Import`）
* 特定导入（`ImportFrom`）

每个元素被转换为一个包含：

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
### 语义搜索

* 将项目分割为多个片段（chunks）。
* 使用 `all-MiniLM-L6-v2` 生成嵌入（embeddings）。
* 将向量存储在FAISS索引中。
* 为用户查询生成嵌入。
* 检索最相关的k个片段。
* 使用FAISS进行候选片段的语义搜索。
* 结合密集与词典搜索的混合检索。
* 对候选片段进行词典重排序。
* 选择最相关的片段作为RAG上下文。

示例查询：

* "读取文件内容的函数"
* "读取目录的函数"

---

<a id="architecture"></a>
## 架构

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
## 所使用技术

<a id="parsing"></a>
### 解析

* Python AST

<a id="embeddings"></a>
### 嵌入（embeddings）

* 句子变换模型
* all-MiniLM-L6-v2

<a id="base-vectorielle"></a>
### 基础向量

* FAISS

<a id="llm-local"></a>
### 本地大语言模型

* Ollama
* Qwen2.5-Coder 14B
* Devstral 24B 用于工具调用

<a id="rag"></a>
### 检索增强生成

* FAISS
* 混合密集+词法检索
* 词法重排序
* 结合文件/行来源的上下文注入
* 引用 `[Sx]` 及 grounding 验证

<a id="tests"></a>
### 测试

* Pytest

---

<a id="installation"></a>
## 安装

<a id="1-cloner-le-projet"></a>
### 1. 克隆项目

```bash
git clone <repo>
cd CodeAgent
```

<a id="2-installer-les-dépendances-python"></a>
### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

<a id="3-installer-ollama"></a>
### 3. 安装 Ollama

在机器上安装 Ollama，然后验证安装是否成功 :

```bash
ollama --version
```

<a id="4-télécharger-le-llm-local"></a>
### 4. 下载本地大语言模型

CodeAgent 使用 `qwen2.5-coder:14b` 进行 RAG / grounding，以及使用 `devstral:24b` 进行工具调用。

通过以下命令下载模型 :

```bash
ollama pull qwen2.5-coder:14b
ollama pull devstral:24b
```

验证模型是否可用 :

```bash
ollama list
```

---

<a id="exécution-du-projet"></a>
## 项目执行

从项目根目录启动 CodeAgent :

```bash
python -m app.main
```

程序随后会要求对项目提出一个问题 :

```text
Question : Comment les embeddings sont-ils ajoutés dans FAISS ?
```

CodeAgent 提供 RAG 模式和 Agent 模式。Agent 模式复用 MCP M6 循环，当请求中包含明确依赖时，会添加 M8 计划，预先加载本地模型，并按问题逐个显示 Devstral / grounding 的 token 数、各步骤耗时、工具调用详情、调用时长、grounding 状态以及响应时间。

要在 PowerShell 中观察 M8，执行该命令后选择 **2** :

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main --project . --session m8-validation --debug
```

`--planning off` 与 M6 直接对比。`--max-steps`、`--max-plan-steps`、`--max-replans`、`--max-tool-calls`、`--max-total-tokens`、`--max-seconds` 和 `--tool-timeout` 限制了执行过程。[M8 综合分析](docs/memory.zh-CN.md#m8--planning--controlled-autonomous-execution) 展示了该计划的成果与局限性。

<a id="assistance-humaine-m8-optionnelle"></a>
### 可选的人类辅助 M8

```powershell
python -m app.main --project . --session hitl-test --debug --hitl on-failure
```

选择 **2**（Agent）。自主尝试失败后，HITL 可暂停执行，以检查证据、修正工具调用或剩余计划、重试或停止。恢复保留同一任务、已完成工作、证据与全局预算。人工输入仍须验证，且不构成来源证据；非交互执行返回 `intervention_required`。

确定性测试覆盖中断与恢复。最近的真实自主场景均无需人工介入，因此这些运行本身不能验证 HITL 周期。参见 [HITL 合约](docs/memory.zh-CN.md#m8-hitl)与[验证证据](docs/memory.zh-CN.md#m8-validation)。

要重新开始对话，请使用相同的会话标识符和相同的项目：

```bash
python -m app.main --session analyse-1
python -m app.main --project /chemin/vers/projet-python --session analyse-1 --context-tokens 16384
```

选择另一个`--session`以开启独立对话。预算会保留最近的交互内容，自动压缩旧内容为确定性摘要，并选择性地恢复JSONL存档。原始观察结果仍然可用；旧的引用标识符不会被重新用作当前来源。保守估计使用UTF-8字节，而非模型的分词器。

<a id="vérifications-sans-llm-réel"></a>
### 无需LLM真实运行的验证

```powershell
$m8CheckDirectory = Join-Path $env:TEMP ('codeagent-check-' + [guid]::NewGuid().ToString())
python -m pytest -q
$env:HF_HUB_OFFLINE = '1'
python -m scripts.benchmark_rag --output-dir (Join-Path $m8CheckDirectory 'rag')
```

记忆与规划检查及临时输出文件的用法见[基准测试协议](benchmarks/llm/README.zh-CN.md#memory-and-planning-protocols)。

内存基准测试不会下载任何内容。RAG基准测试需要本地已安装的MiniLM（使用`HF_HUB_OFFLINE=1`禁用网络）。`pytest.ini`和`conftest.py`会将pytest缓存和`tmp_path`文件放置在系统临时目录中，并在使用后清理fixture。

---

<a id="documentation"></a>
## 文档

- [项目内存、上下文和规划 — M1–M8，结果与验证](docs/memory.zh-CN.md)
- [M5 — RAG质量与来源依据核验](docs/rag_quality.zh-CN.md)

<a id="roadmap"></a>
# 路线图

<a id="milestone-1--lecture-et-extraction-du-code"></a>
## 里程碑1 — 代码的读取与提取

<a id="objectifs"></a>
### 目标

* [x] 项目初始化
* [x] 递归读取Python文件
* [x] 路径管理
* [x] 忽略无用文件
* [x] 内容提取
* [x] 单元测试搭建

<a id="résultat"></a>
### 结果

该项目能够自动遍历Python仓库并提取分析所需的必要信息。

---

<a id="milestone-2--embeddings-et-recherche-sémantique"></a>
## 里程碑 2 — 嵌入与语义搜索

<a id="objectifs-1"></a>
### 目标

* [x] 将代码划分为块
* [x] 解析抽象语法树
* [x] 生成嵌入向量
* [x] 构建向量索引
* [x] 查找最近邻
* [x] 对用户查询进行嵌入
* [x] 实现语义搜索功能

<a id="résultat-1"></a>
### 结果

CodeAgent 能够根据自然语言问题自动检索最相关的代码片段。

---

<a id="milestone-3--rag-sur-le-code"></a>
## 里程碑 3 — 代码上的 RAG

<a id="benchmark-llm"></a>
### LLM 性能基准测试

* [x] 定义一个通用基准以比较本地 LLM
* [x] 比较多个本地 LLM
* [x] 选择并集成一个本地 LLM

<a id="pipeline-rag"></a>
### RAG 流程

* [x] 构建 RAG 流程
* [x] 将检索到的代码块注入上下文
* [x] 对项目问题进行回答
* [x] 生成代码解释
* [x] 引用相关文件和行号

<a id="exemple"></a>
### 示例

问题：

> 函数读取文件是在哪里定义的？

回答：

> 函数 `read()` 在 `app/parser/file_loader.py` 中定义，第 6 到第 9 行。

来源：
- `app/parser/file_loader.py:6-9`

---

<a id="milestone-4--mémoire-projet"></a>
## 里程碑 4 — 项目记忆

<a id="objectifs-2"></a>
### 目标

* [x] 持久化保存索引
* [x] 保存分析历史记录
* [x] 实现嵌入向量的增量更新
* [x] 跟踪项目变更

---

<a id="milestone-5--rag-quality--grounding"></a>
## 里程碑 5 — RAG 质量与事实验证

<a id="objectifs-3"></a>
### 目标

- [x] 构建一个 RAG 基准测试
- [x] 增加 Recall@K、Precision@K 和 MRR
- [x] 评估当前检索性能作为基线
- [x] 增加密集检索与词典检索的混合搜索
- [x] 评估并优化重排序
- [x] 提升发送给 LLM 的上下文质量
- [x] 加强提示以减少幻觉
- [x] 添加文件与行号引用
- [x] 自动验证引用内容
- [x] 增加对响应的源文件 <=> 的验证
- [x] 将最终流程与 M4 基线进行对比

<a id="résultat-2"></a>
### 结果

在最终 16 个问题的验证集上，M5 将 `Hit@5` 和 `Recall@5` 的性能从 `0.6667` 提升至 `0.8333`，同时优化了响应/回避行为，从 `0.6250` 改进到 `0.7500`。观察到的权衡是更弱的 `MRR@5`（`0.5069 → 0.3917`），以及由于事实验证带来的平均额外开销 `+2.54 s`。

---

<a id="milestone-6--mcp--tool-use"></a>
## 里程碑 6 — MCP 与工具使用

<a id="objectifs-4"></a>
### 目标

- [x] 定义工具的最小接口
- [x] 为每个工具定义名称、描述和一个 `JSON Schema`
- [x] 执行前验证工具参数
- [x] 实现一个只读工具集：文件读取、代码搜索、文件列表、Git 差分检查和测试执行
- [x] 让大语言模型决定是否需要使用工具以及选择哪个工具
- [x] 通过MCP集成工具
- [x] 建立`LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM`循环
- [x] 添加超时、最大步骤数、重试限制以及重复调用检测（`tool + arguments`）
- [x] 验证工具结果，并向大语言模型返回结构化错误信息
- [x] 当使用项目源时，复用M5的 grounding 以生成最终回答
- [x] 在专用基准上评估工具使用情况
- [x] 跟踪Devstral/grounding的token数、步骤数、工具调用次数、每步耗时以及单个工具的耗时
- [x] 通过有界搜索、精准阅读和合并冗余源来优化工具上下文
- [x] 在专用保留集上独立评估grounding模型性能

<a id="implémentation"></a>
### 实现

初始范围仅支持只读。任何修改项目状态的工具将在执行循环稳定后才被添加，并且对于高风险操作必须明确请求确认。

基准测试跟踪任务成功率、无效调用、重试、重复调用、工具错误、步骤数、grounding、token数和延迟。

<a id="résultat-3"></a>
### 结果

CodeAgent在判断是否需要工具时做出决策，通过MCP选择并执行合适的工具，避免无控制循环，捕获简单错误，并在grounding验证前生成最终源引用回答。

在最终20个任务的端到端基准测试中，路由、工具选择、无工具决策及参数配置均达到`100 %`，端到端成功率达到`80 %`（`13`个被接受的源引用回答，`4`个拒绝，`unsupported`个被拒绝，`0`个被拒绝，`citation_error`个被拒绝）。每任务平均步骤数为`2.10`，平均工具调用次数为`1.15`；平均延迟仍较高，达到`131.05 s`，主要由于本地大语言模型的延迟。

在包含40个平衡案例的grounding保留集中，`Qwen2.5-Coder 14B`准确率达到`97.5 %`，对未支持案例的召回率达到`100 %`，错误接受数为0，错误拒绝数为1。`Devstral 24B`准确率达到`90 %`，错误接受数为0，错误拒绝数为4；因此选择Qwen14作为M6的grounding模型。上下文优化将多工具烟雾测试的token数从`10.5k`减少至`6.6k`，而`search_code`的优化使其观察到的运行时间从`17.1 s`减少至`0.01-0.12 s`。

在20个任务的门控基准测试中，Devstral达到`100 %`的准确率；Laya和Verdict达到`75 %`，ModernBERT-NLI达到`65 %`，GLiNER2.5达到`45 %`。Jev无法评估，因为尽管调用符合文档要求，外部API仍不可访问。在30个案例的保留集上测试Verdict到Devstral的级联流程，将平均延迟降低至`42.9 %`，但其准确率仅为`93.3 %`，相较于单独的Devstral（`100 %`），因此未被采纳。

<a id="future-improvement"></a>
### 未来改进

- 当Jev的API可访问时重新评估其性能，并仅当新路由/grounding模型能在不增加错误接受的前提下提升准确率与延迟平衡时才考虑引入。

---

<a id="milestone-7--memory--context-management"></a>
## 里程碑7 — 内存与上下文管理

<a id="objectifs-5"></a>
### 目标

- [x] 保留最近几轮对话的短期记忆
- [x] 概括并压缩旧对话轮次
- [x] 为上下文设定令牌预算
- [x] 当阈值被超过时，自动触发压缩
- [x] 在两次执行之间持久化摘要
- [ ] 可选：通过语义搜索检索相关历史对话
- [x] 分离会话与项目，保存决策、约束、错误和来源
- [x] 在压缩前选择结构化观察结果，保留原始数据以供验证
- [x] 在无LLM的测试用例上持续测量连贯性、提取保真度和上下文大小

<a id="résultat-4"></a>
### 结果

CodeAgent 拥有持久、按会话和项目隔离的对话记忆，具备自动压缩和共享上下文预算功能。其运行已在确定性测试用例上验证，**9/9 回忆准确**。**M7 已完成**，在记忆与上下文方面；历史实际测试中实现了 **7/7 读取**，但仅实现了 **0/7 完全自主最终执行**。

---

<a id="milestone-8--planning--controlled-autonomous-execution"></a>
## 里程碑 8 —— 计划与受控自主执行

<a id="objectifs-6"></a>
### 目标

- [x] 检测确实需要多步骤依赖的任务
- [x] 仅针对这些任务生成简短计划
- [x] 保持最小状态：目标、已完成步骤、观察结果、失败情况和剩余预算
- [x] 使用 M6 工具循环执行每一步
- [x] 仅当观察结果或错误使计划无效时，重新规划剩余步骤
- [x] 在重新规划时保留已验证的工作
- [x] 当目标达成、预算耗尽或需要用户决策时，停止执行
- [x] 通过比较 M6 直接循环与 M8 计划，评估多步骤任务

<a id="résultat-5"></a>
### 结果

具有依赖关系的任务使用经过验证的计划；简单任务继续使用 M6 直接循环。MCP 观察结果用于验证步骤，重新规划在同一预算内保留已完成工作。最终回答须检查所需交付内容、引用与来源依据。最终结果、复现配置与限制集中记录于 [M8 汇总](docs/memory.zh-CN.md#m8--planning--controlled-autonomous-execution)。原生 JSON Schema 输出经过评估后未被保留；交付的规划器在代码中验证 JSON 合约。

---

<a id="milestone-9--advanced-rag--reliability"></a>
## 里程碑 9 — 高级 RAG 与可靠性

<a id="objectifs-proposés"></a>
### 目标提案

- [ ] 加强元数据、稳定标识符、版本、过滤器以及索引/分块的一致性
- [ ] 比较 BM25 与专用重排序器相对于现有词法/RRF 方法，使用 Hit@K、Recall@K、MRR 和 nDCG 评估
- [ ] 仅当初始检索失败时，才进行迭代检索和重表述
- [ ] 检测过时或矛盾的来源，并评估针对性修正
- [ ] 比较独立验证器/NLI 与现有锚定机制：误接受、误拒绝、P50/P95 以及延迟

<a id="future-improvements--research-ideas"></a>
### 未来改进 / 研究想法

对于 Python 代码，仍优先使用 AST。递归分块、TreeRAG、GraphRAG、OCR 和多模态功能需要先建立具体语料与比较评估协议，再考虑集成。目前没有理由新增 M10。多智能体、独立 Critic Agent、复杂 Reflexion 及抗幻觉微调仍暂缓。保留 FAISS 与当前 MCP 循环。无需 LangChain；仅当恢复执行和分支处理使现有工作流不足时，才重新评估 LangGraph。不计划引入分布式或云系统。

---

<a id="état-actuel-du-projet"></a>
## 项目当前状态

CodeAgent 现已具备完整流程，可分析 Python 项目、索引其代码、检索相关片段、利用本地 LLM 生成上下文化响应，并在项目演进时持续更新索引。

基础 RAG 已在 M5 完成，通过 MCP 控制的工具调用功能在 M6 完成。**M7 已完成**，涵盖内存与上下文管理。对路由和最终生成的实际限制已记录；这些限制并未证明 M7 存在内存或来源损坏。M8 增加了条件规划及其可观测控制；M9 仍为未来阶段，将整合 RAG 及其可靠性方面的可测量改进。

<!-- multilingual-documentation -->

## 更多文档

- [benchmarks/llm/README.md](benchmarks/llm/README.zh-CN.md)

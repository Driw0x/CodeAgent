<a id="codeagent"></a>

# CodeAgent

[Français](README.md) | [简体中文](README.zh-CN.md)
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

**M1–M7 已完成 · Devstral 自主最终回答的局限已有记录 · M8/M9 已规划**

CodeAgent 的核心基础已完工并可运行：代码分析、混合搜索、RAG管道、源码响应、索引持久化、增量更新、事实验证以及通过MCP实现工具的受控使用。

里程碑 1 至 7 已完成。记忆与上下文预算已经验证；Ollama 自主最终回答的局限仍记录于[项目汇总记录](docs/memory.zh-CN.md#limites-connues)。M8 将增加受控的多步骤规划。

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

CodeAgent 提供 RAG 模式和 Agent 模式。Agent 模式使用 M6 MCP 循环，预加载本地模型，并针对每个问题显示 Devstral / grounding 的 token 数、各阶段耗时、工具调用及其耗时、grounding 状态和响应时间。

要重新开始对话，请使用相同的会话标识符和相同的项目：

```bash
python -m app.main --session analyse-1
python -m app.main --project /chemin/vers/projet-python --session analyse-1 --context-tokens 16384
```

选择另一个`--session`以开启独立对话。预算会保留最近的交互内容，自动压缩旧内容为确定性摘要，并选择性地恢复JSONL存档。原始观察结果仍然可用；旧的引用标识符不会被重新用作当前来源。保守估计使用UTF-8字节，而非模型的分词器。

<a id="vérifications-sans-llm-réel"></a>

### 无需LLM真实运行的验证

```powershell
python -m pytest -q
python -m scripts.benchmark_memory
$env:HF_HUB_OFFLINE = '1'
python -m scripts.benchmark_rag --output-dir (Join-Path $env:TEMP 'codeagent-m7-rag-final')
```

内存基准测试不会下载任何内容。RAG基准测试需要本地已安装的MiniLM（使用`HF_HUB_OFFLINE=1`禁用网络）。`pytest.ini`和`conftest.py`会将pytest缓存和`tmp_path`文件放置在系统临时目录中，并在使用后清理fixture。

---

<a id="documentation"></a>

## 文档

- [项目记忆与上下文管理 — M1–M7、结果和验证](docs/memory.zh-CN.md)
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

CodeAgent 保留近期上下文，在超出预算时自动压缩较早的历史，并可通过持久化摘要恢复对话。

<a id="résultat-automatisé"></a>

### 自动化结果

在三段各含 63 轮交互的确定性对话中，M7 在压缩和恢复后找回了全部 9 条预期信息；每轮重置历史的对照结果为 0/9。最大估算上下文为 6 271，输入上限为 6 444；完整回放将达到 48 957。摘录保真度、来源和错误恢复通过基准测试标准，且没有调用 LLM 生成摘要。这些测试替身不衡量真实模型的质量或延迟。最终验证复现了 **220 项测试通过**，数量与参考测试活动一致。使用 Ollama 时，参考对话路由通过 **3/3**，固定 MCP 来源的 FAISS 最终回答通过 **3/3**。`read_file` 技术恢复成功 **7/7**，但完整自主最终回答仍为 **0/7**。未进行实际检查的 FAISS 问题可能导致拒答。**M7 已完成**记忆与上下文范围的工作；自主生成质量、grounding 局限和 Ollama 内存压力仍记录于[结果与局限](docs/memory.zh-CN.md#tests-et-benchmarks)。[复现命令](docs/memory.zh-CN.md#commandes-de-reproduction)区分测试替身与真实模型。复现的 3/3 路由结果不保证内容忠实：空会话中的回忆会编造事实。历史语义检索仍为可选项；词法检索复用现有分词器。

---

<a id="milestone-8--planning--controlled-autonomous-execution"></a>

## 里程碑 8 —— 计划与受控自主执行

<a id="objectifs-6"></a>

### 目标

- [ ] 检测确实需要多个依赖步骤的任务
- [ ] 仅为这些任务生成简短计划
- [ ] 维护最小状态：目标、已完成步骤、观测、失败和剩余预算
- [ ] 使用 M6 工具循环执行每个步骤
- [ ] 仅在观测或错误使计划失效时重新规划剩余步骤
- [ ] 在重新规划时保留已经验证的工作
- [ ] 在目标达到、预算耗尽或需要用户决策时停止执行
- [ ] 通过比较 M6 直接循环与 M8 规划来评估多步骤任务

<a id="implémentation-prévue"></a>

### 计划中的实现

目标循环保持简单：`Goal -> Plan optionnel -> Step -> Tool -> Observation -> Next Step / Replan -> Final Answer`。

简单请求不使用规划，继续直接使用 M6 循环。

高级观测外置及其按需检索仍以可测量的需求为前提。

<a id="résultat-attendu"></a>

### 预期结果

CodeAgent 预期能够受控地执行多步骤开发任务，跟踪进度，并从简单失败中恢复，无需无谓地重做已经完成的工作。

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

RAG 基础在 M5 完成，经 MCP 的受控工具使用在 M6 完成。**M7 已完成**记忆和上下文管理。真实路由与最终生成的局限已有记录；它们并不证明 M7 记忆或来源发生损坏。M8 将加入条件规划；M9 将汇总经测量验证的 RAG 及其可靠性改进。

<!-- multilingual-documentation -->

## 更多文档

- [LLM 基准测试](benchmarks/llm/README.zh-CN.md)

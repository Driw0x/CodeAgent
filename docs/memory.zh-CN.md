<a id="codeagent--project-memory"></a>

# CodeAgent — 项目记忆

[Français](memory.md) | [简体中文](memory.zh-CN.md)

[项目 README](../README.zh-CN.md)

> 项目汇总记录：M1–M6 已完成，M7 已完成记忆和上下文范围的工作；自主能力的局限已有记录。

<a id="état-actuel"></a>

## 当前状态

里程碑 1、2、3、4、5 和 6 已完成。

RAG 基础在 M5 完成，经 MCP 的受控工具使用在 M6 完成。

CodeAgent 当前能够分析 Python 项目、为代码建立索引、检索与问题相关的代码片段、生成有来源的回答、验证引用和 grounding、维护项目记忆，并通过 MCP 在受控循环中使用只读工具。

M7 增加持久化对话会话、抽取式压缩，以及 RAG、MCP 观测和 Ollama 请求共享的预算。M7 的结果、局限和协议汇总如下。

<a id="m1--lecture-et-extraction"></a>

## M1 — 读取与提取

第一个里程碑为项目奠定了基础：

- 对Python文件进行递归遍历；
- 过滤无用文件夹；
- 读取内容；
- 相对路径的管理；
- 初步单元测试。

重要决策：保持流水线简单且独立于大语言模型，以确保仓库分析的确定性。

<a id="m2--recherche-sémantique"></a>

## M2 — 语义搜索

随后，代码使用Python的AST进行了切分，主要生成代表函数、类、变量和导入的片段。

每个片段保留其来源信息：

- 文件；
- 类型；
- 名称；
- 起始和结束行号。

这些片段使用`all-MiniLM-L6-v2`进行编码，然后在FAISS中进行索引。

检索随后通过应用于FAISS返回候选结果的词法重排序得到了优化。

获得的流水线：

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

## M3 — 本地大语言模型与RAG

M3增加了对代码的响应生成功能。

在集成之前，多个本地大语言模型通过共同基准进行了对比，以选择适合项目的模型。

选定的模型：

- `Qwen2.5-Coder 14B`
- 使用Ollama在本地执行。

最终的RAG流水线是：

```text
Question
→ embedding
→ recherche FAISS
→ reranking
→ sélection des chunks
→ construction du contexte
→ Qwen2.5-Coder 14B
→ réponse avec provenance
```

传递给大语言模型的上下文保留了分块的元数据，从而能够生成指出所使用文件和行的响应。

示例：

```text
app/parser/file_loader.py:6-9
```

<a id="m4--mémoire-projet"></a>

## M4 — 项目记忆

M4为项目增加了持久且增量的记忆，以避免每次执行都重新构建整个索引。

新增功能：

- FAISS索引的持久保存；
- 相关分块的保存；
- 现有索引的自动重新加载；
- 问题与回答的历史记录；
- 项目宣言的创建；
- 新增、修改和删除文件的检测；
- 嵌入和索引的增量更新；
- 保留相对路径以识别项目文件。

清单将每个文件映射到其内容的 SHA-256 哈希。由此可比较项目的两个连续状态，准确确定修改内容。

更新流水线：

```text
Projet Python
→ lecture des fichiers
→ manifeste courant
→ comparaison avec le manifeste précédent
→ fichiers ajoutés / modifiés / supprimés
→ recalcul uniquement des chunks concernés
→ mise à jour de l’index FAISS
→ sauvegarde de l’index et du nouveau manifeste
```

当未检测到任何修改时，CodeAgent会直接重用已保存的索引。

分析结果也记录在包含日期、问题和生成回答的JSONL历史记录中。

<a id="m5--rag-quality--grounding"></a>

## M5 — RAG质量与锚定

M5通过以下方式增强了并评估了RAG管道：

- 使用Hit@K、Recall@K、Precision@K和MRR的检索基准测试；
- 混合密集检索与词法检索；
- 改进的重排序和上下文选择；
- 增强的提示以减少幻觉；
- 自动验证的`[Sx]`引文；
- 逐条验证回答对源材料的支持情况。

在最终的16个问题测试集上，M5将`Hit@5`和`Recall@5`的性能从`0.6667`提升至`0.8333`，并将`0.6250`的回答/回避行为从`0.7500`改进。`MRR@5`从`0.5069`提升至`0.3917`，且锚定验证在管道中平均增加了`2.54 s`。

<a id="m6--mcp--tool-use"></a>

## M6 — MCP与工具使用

M6通过MCP引入了受控的工具使用：

- 只读工具`read_file`、`search_code`、`list_files`、`get_git_diff`和`run_tests`；
- 使用`Devstral 24B`进行工具调用；
- `LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM`循环；
- 超时、步骤限制、重试限制、重复调用检测和结构化错误处理；
- 在基于工具观察的回答中重用M5的锚定结果；
- 按问题跟踪Devstral/锚定的token数量、步骤数、工具调用、每步骤时间以及各工具的单独时间；
- 从源观察结果生成最终回答，并进行引文验证和锚定；
- 通过限制检索结果、按行范围读取和合并冗余源来减少工具上下文。

最终的20个任务端到端基准测试达到了`80 %`的成功率，其中路由成功率为`100 %`，工具选择精度为`100 %`，无工具精度为`100 %`，参数精度为`100 %`。20个任务产生了13个被接受的锚定回答，4个拒绝`unsupported`，0个`citation_error`。平均延迟仍然较高（`131.05 s`），主要由本地LLM决定，而工具本身现在已变得快速。

在包含40个平衡案例的固定锚定基准测试中，`Qwen2.5-Coder 14B`达到了`97.5 %`的准确率，`100 %`的未支持案例召回率，0个误接受，1个误拒绝。`Devstral 24B`达到了`90 %`，且0个误接受，4个误拒绝；因此`Qwen2.5-Coder 14B`仍然是默认的锚定器。在包含20个任务的门控基准测试中，Devstral达到了`100 %`的精度。Laya和Verdict达到了`75 %`，ModernBERT-NLI达到了`65 %`，GLiNER2.5达到了`45 %`。Jev未能被评估，因为尽管调用符合文档要求，外部API仍不可访问。Verdict到Devstral的级联流程降低了延迟`42.9 %`，但将精度从`100 %`降至`93.3 %`；该流程未被采纳。

最终上下文优化将多工具烟测试的令牌数量从`10.5k`减少到`6.6k`，且不损失来源依据核验能力。`search_code`也已优化，通过避免在过滤前遍历被忽略的文件夹：其在烟测试中的观察时间从大约`17.1 s`降至`0.01-0.12 s`。

<a id="m7--memory--context-management"></a>

## M7 — 内存与上下文管理

**M7 已完成其功能范围内的工作。**记忆、预算、压缩、恢复、隔离和来源保留均已验证。此阶段的完成不保证所有自主最终回答都可靠。M8 和 M9 仍为后续阶段。

<a id="architecture-et-persistance"></a>

### 架构与持久化

项目记忆（`index.faiss`、`chunks.json`、`manifest.json`）与对话分离。现有 `history.jsonl` 归档增加可选的 `session_id` 和 `observations` 字段，同时保留 `timestamp`、`question` 和 `answer`。没有会话标记的旧分析仍可读取，但不会自动导入新对话。

`ConversationMemory` 仅恢复请求的会话。原子检查点 `sessions/<session_id>.json` 保存摘要、压缩计数和已处理轮次的边界。v1 检查点从 JSONL 重建，以恢复被重新确认的决策；新检查点为 v2。损坏状态会被重建；未来版本或领先于归档的检查点会被拒绝，且不会被覆盖。近期交互从归档重建，不额外持久化复制原始文本。

摘要包含标注角色、顺序和优先级的摘录。决策、约束、路径、错误和假设仍可辨认。不调用摘要模型。相关的旧信息可通过流式扫描归档进行词法检索。历史引用与当前轮次的来源明确区分；grounding 继续仅验证当前来源。

`ContextBudget` 以保守估算统计消息、工具 schema 和观测：每个 UTF-8 字节计一个 token，每条消息计 32 单位，另为 framing 保留 256 单位。生成有独立的 `num_predict` 预留。`--context-tokens`（或 `CODEAGENT_CONTEXT_TOKENS`，默认 16 384）同时设置预算和 Ollama 的 `num_ctx`。若某个必需元素过大，则在发送前明确报错，不任意截断。

较大的工具结果按完整对象或完整行缩减，同时保留路径、行号和对应值。原始结果留在归档中。最终回答和 grounding 使用相同的来源选择。`--project` 也隔离不同项目的索引与会话，并将项目传递给 stdio MCP 服务器。

保留策略：智能体上下文默认保留最近四轮原始交互，选择更早的轮次，摘要上限为记忆预算的一半；先淘汰低优先级摘录，再淘汰更旧的摘录。只追加归档从不自动清理。词法检索仍线性扫描归档。持久化会话假定只有一个写进程；并发写入和高级磁盘保留策略仍不在当前本地范围内。

<a id="réalisations-m71-à-m76"></a>

### M7.1 至 M7.6 的实现

| 步骤 | 实现 | 主要文件 | 验证 |
|---|---|---|---|
| M7.1 | 有序轮次、近期上下文、决策和约束、会话与项目隔离 | `app/memory/conversation.py` | 角色、修正、尚未压缩的旧轮次和无污染 |
| M7.2 | 系统、记忆、RAG、工具及 schema 的共享预算，回答预留和显式 `num_ctx` | `app/memory/context.py`, `app/llm/local_llm.py` | UTF-8、精确边界、联网前拒绝、多条观测 |
| M7.3 | 自动抽取式压缩、优先级、去重和淘汰 | `app/memory/conversation.py` | 精确摘录、ORION/VEGA/ORION 时序、假设不被当作事实 |
| M7.4 | 复用 JSONL 归档、原子检查点、恢复和重建 | `app/memory/history.py`, `app/memory/conversation.py` | 重启、不完整归档、损坏和未来版本 |
| M7.5 | 定向词法检索，在缩减前进行结构化选择 | `app/memory/context.py`, `app/agent/tool_loop.py`, `app/rag/pipeline.py` | 有效 JSON、完整行与对象、精确值以及与 grounding 共用的来源 |
| M7.6 | RAG/Agent 集成、CLI 和确定性 M6/M7 基准测试 | `app/main.py`, `scripts/benchmark_memory.py` | 内存内与 stdio MCP、RAG 后续问题、预算、恢复及错误恢复 |

`--session` 选择对话；`--project` 选择项目。解析后的路径决定哈希记忆目录：两个同名项目仍相互隔离。项目显式传递给 stdio MCP。明确的 RAG 问题保持原样；简单的代词后续问题可以复用上一问题。完整语义改写和旧交互的语义检索仍被推迟。

<a id="sources-et-corrections-vérifiées"></a>

### 来源与已验证的修正

处理链为 `observation MCP originale → source → sélection M7 → prompt final → citations → grounding`。`build_tool_source` 保留路径、实际边界、参数和结果。缩减操作使用副本。`consolidate_sources` 移除冗余来源；`fit_sources` 将保留来源重新编号为连续序号。grounding 接收的正是这份最终列表。错误只能证明其诊断，不能证明请求的内容。历史数据不会成为当前代码的证据；修改后的来源必须重新读取。

最近一次验证修正了三个已经证实的 M7 问题：

- ORION/VEGA/ORION 的重新确认曾丢失最后一次事件。现在最后一次出现保留其轮次，v1 检查点会被重建。
- 五条观测可能超出预算：可用 6 444，却估算为 9 133。当前共享预算预留系统、调用、schema 和 JSON 转义；五个 ID、值和行仍全部保留。
- 可选历史曾在 60 轮交互后挤掉恢复读取的内容。现在当前来源优先于历史；`source_valid` 和 `tool_error_recovered` 再次通过。

此前已经交付的相关修正继续保留，本次最终验证没有新增循环变更：

- 五种工具中预期的 `ValueError` 诊断以 `ToolError` 传递。起始位置超过 EOF 时会被拒绝，空文件亦如此；过大的结束位置会被限制。
- 实际执行产生的错误、异常和超时保留在恢复回答的最终来源中。
- `TaskGroup` 的叶级异常按类型和消息显示并持久化；原始异常组被重新抛出。历史 `analyse-1` 异常仍原因未知，尚未复现。
- 部分拒答中的事实仍会被验证。没有成功读取时，虚构恢复成功的说法会被转为拒答。
- 只有在验证所有 claims 后，才补上单一来源的缺失引用。多个来源之间不自动分配引用；无效引用仍被拒绝。
- 对于明确询问项目代码但没有进行当前检查的请求，系统会拒答。此保护仍为启发式；间接指代依赖路由器。
- Ollama 默认 URL 使用 `127.0.0.1`。显式设置的 URL 仍受尊重。

`--debug` 或 `CODEAGENT_DEBUG=1` 记录上下文、路由回答、原始候选回答、最终提示词、选择前后的来源、ID 映射、claims 和各请求指标。`missing_references`、`invalid_references` 和 `verifier_rejected_claim` 区分拒绝原因；`supported=null` 表示异常，而非否定判定。可选诊断只保存在本地 JSONL，不重新注入对话。完整 traceback 留在终端。

此前已有的相关清理：移除停用的 embedding 别名与注释及 `compact_text` 截断，复用现有 reranking，将 `anyio_backend` 测试夹具集中管理。未删除历史测试或有用夹具。审查未找到已证明整块无用的模块。pytest 缓存和 `tmp_path` 放在系统临时目录；拒绝将 `--basetemp` 指向仓库。`data/memory/`、用户索引和会话留在本地，受 Git 忽略并予以保留。

<a id="tests-et-benchmarks"></a>

### 测试与基准测试

2026 年 10 月 14 日的参考验证：**220 项测试通过**。历史进展：最初 133 项，增加 FAISS 增量测试后 134 项，之后为 172、188、198 和 220 项。最后净增的 22 项主要验证时序、预算、来源选择和引用。没有人为固定计数。未配置额外 lint、type-check 或 build。

仓库最终验证复现 **220 passed，耗时 6,58 s**；清理后为 **220 passed，耗时 5,46 s**，没有失败或忽略项，测试数量变化为 **0**。记忆基准复现六项标准：M7 **9/9 回忆成功**，每轮重置为 **0/9**，最大值 **6 271**，上限 **6 444**，完整回放 **48 957**，零次 LLM 摘要调用。`benchmarks/memory/results/m7.json` 保存该次测量与测试替身的延迟。本次最终验证移除了脚本中未使用的 `estimate_tokens` 导入。

<a id="protocole-déterministe-et-résultats-historiques"></a>

#### 确定性协议与历史结果

`deterministic-memory-v1` 使用三段预定义的 63 轮对话（含 60 轮次要交互），每段三次回忆，中途恢复一次，再在文件缺失错误后读取一个 600 行文件。内存内 MCP 客户端和工具为真实实现；回答与 grounding LLM 为测试替身。六项标准要求回忆全部成功、摘录精确、遵守预算、使用当前来源、从错误恢复且不使用摘要模型。这些标准不衡量 Ollama 的理解能力。

首次验证保留为精确测量值：

| 测量 | M6：每轮重置为零 | M7 |
|---|---:|---:|
| 成功的字面回忆 | 0/9 | 9/9 |
| 最大估算上下文 | 5 354 | 5 945 |
| 输入上限 | 6 444 | 6 444 |
| 完整回放，单独比较大小 | 最多 48 957 | 最多 48 957 |
| 每段对话的压缩次数 | 0 | 62 |
| 对话结束时已压缩的轮次 | 0 | 65 |
| 所选旧摘录的保真度 | 没有回忆 | 在测试夹具上完全一致 |
| 有来源的最终回答 / 测试替身 grounding | 3/3 | 3/3 |
| 已恢复的工具错误 | 3/3 | 3/3 |
| 每段对话的测试替身 LLM 调用 | 8 | 8 |
| LLM 摘要调用 | 0 | 0 |
| 每段对话的工具调用 | 2 | 2 |
| 每例平均延迟 | 5,77–8,43 ms | 29,44–42,18 ms |
| 每例 P50 | 1,07–3,16 ms | 19,27–22,16 ms |
| 每例 P95（测量 4 个问题） | 20,10–26,71 ms | 63,55–104,18 ms |
| 每个模型的实测 token | 未测量 | 未测量 |

最新一次验证修正了预算分配，达到 6 271/6 444；中间阶段的 `m7_robustness.json` 测得 6 418/6 444，仍为 9/9 回忆成功且六项标准均为真。约 88 % 的缩减是 M7 与完整回放相比，而非与无记忆的小上下文相比。M6 对照使用当前工具和预算模拟每轮重置为零，并非执行 M6 的历史提交。测试替身延迟来自每段对话中测量的四个问题，不能证明模型延迟改善。归档仍线性扫描；没有数据规模测量就不增加索引。

<a id="régression-rag-et-grounding-historique"></a>

#### RAG 回归与历史 grounding

在固定的 **25 个问题 / 133 个 chunks** 语料上，M7 前后的结果为：

| 测量 | M4 基线 | M5 重排序/上下文 |
|---|---:|---:|
| Hit@5 | 0,7200 | 0,8400 |
| Recall@5 | 0,7200 | 0,8200 |
| MRR@5 | 0,5233 | 0,5440 |
| 平均上下文，字符数 | 未测量 | 2 389,36 |
| 丢失的相关来源 | 未测量 | 0 |

`benchmarks/rag/results/audit_baseline/` 和 `benchmarks/memory/results/m7_rag_regression/` 的三项汇总指标完全一致。最终验证在相同的 25 个问题、133 个 chunks 上，以 MiniLM 离线模式复现结果：Hit@5 0,84、Recall@5 0,82、MRR@5 0,544，未丢失相关来源。该协议与历史 M5 的 111/167 chunks 结果以及 16 个问题的留出集不同。RAG 基准使用本地 MiniLM，不衡量 Ollama 的生成或 grounding。

真实 Qwen 留出集 `m7_grounding_holdout.json` 保留 **39/40**、精确率 **0,975**、零次错误接受、一次错误拒绝：`grounding_output_supported_01`。来源明确要求 `SUPPORTED`/`UNSUPPORTED`，并支持被拒绝的 claim。本次重跑测得：预加载墙钟时间 **173,207 s**，预热后均值 **2,853 s**，中位数 **2,560 s**，P95 **4,961 s**，40 项检查总计 **114,136 s**，零网络错误。首次尝试在 300 s 后预加载失败，零个评估案例，确切原因仍未知。`preload` 将缺失的耗时返回为零，这不等于测得加载耗时为零。该 holdout 未在最后一次来源验证中重跑，也未为取得 40/40 而修改。

<a id="campagnes-ollama-et-autonomie"></a>

#### Ollama 验证与自主能力

`m7-final-v2` 协议的 S1–S8 标准和窗口位于 `benchmarks/memory/final_cases.json`，由 `scripts/benchmark_m7_final.py` 执行。真实对话、新进程中的重启、预定义压缩历史、固定 MCP 来源回放以及自主路径是不同的测量。下表保留完成的运行及其失败，排除被中断的尝试。

| 真实场景 | 已完成运行 | 成功 | 失败 | 原因 | 状态 |
|---|---:|---:|---:|---|---|
| S1 三个中间问题后的记忆 | 3 | 3 | 0 | 两条事实精确，零工具，`not_used` | 此前两次验证及最终代码已验证 |
| S2 压缩与修正 | 3 | 2 次事实复核 | 1 次最终回答被拒绝 | 首次回忆编造引用；后两次保留 ORION、精确值和否定约束 | 澄清后两次回忆忠实；保留首次失败 |
| S3 新 Python 进程中的持久化 | 3 | 3 | 0 | 相同归档与会话；两条事实精确，零工具 | 已验证，含最终代码 |
| S4 另一会话的隔离 | 3 | 3 次事实复核 | 0 | 没有 ASTER-8617；明确说明代码与约束缺失 | 已验证，含最终代码 |
| S5 固定 MCP 来源的 FAISS 最终回答，v1 | 2 | 0 | 2 | 生成带引用，随后 Qwen 在 300 s 超时 | 保留历史验证 |
| S5 固定 MCP 来源的 FAISS 最终回答，v2 | 3 | 3 | 0 | 事实精确，每个最终回答验证四条 claims，引用有效 | 仅在这些来源上验证；不推断自主检查能力 |
| S5 加入保护前的自主路径 | 3 + 3 | 0 | 6 | 尽管要求检查，仍无工具调用；未证明项目实际用法 | Devstral 路由局限；随后为继承分支加上保护 |
| S5 加入保护后的自主路径 | 1 | 0 个有用回答 | 1 次拒答 | 无证据的候选回答被替换为拒答；零工具 | 保护已验证；仍未得到自主检索 |
| S6 固定 MCP 观测下的恢复最终回答，v1 | 2 | 0 | 2 | 缺少恢复引用；或遗漏恢复，随后 Qwen 超时 | 保留历史验证 |
| S6 固定 MCP 观测下的恢复最终回答，v2 | 3 | 0 | 3 | 诊断有依据，但遗漏恢复；两个修正段落无引用 | 来源存在；生成不可靠 |
| S6 自主路径，首次 v2 验证 | 3 | 0 个完整最终回答 | 3 | 三次真实读取 200–226；一个最终回答无引用，两个遗漏恢复 | 传输与恢复 3/3；生成 0/3 |
| S6 澄清指令后的自主路径 | 3 | 0 个完整最终回答 | 3 | 三次真实读取 220–226；代码块与孤立引用，在 Qwen 前被拒绝 | 传输与恢复 3/3；生成 0/3 |
| S6 加入保护后的自主路径 | 1 | 0 个完整最终回答 | 1 | 成功读取 220–226；同样的孤立引用和拒绝 | 不声称引用已修正 |
| S7 仅对话回忆 | 3 | 3 | 0 | 零工具，事实正确，`not_used` | 已验证；两次验证中的额外路由均为 3/3 |
| S8 无依据的 claim，真实 Qwen | 1 | 1 | 0 | 关于 Kubernetes 的断言在 201,359 s 后以 `UNSUPPORTED` 拒绝 | 验证一次；未重新调节验证器 |

七次 S6 自主运行均真实恢复了 `read_file`，但通过的**完整最终回答为 0/7**。固定来源的 FAISS v2 最终回答通过 **3/3**，这不证明自主检查能力。v1（retriever 1–90）与 v2（70–115，含 `index.search`）窗口不同，不据此推导前后改善。S2 压缩使用预定义历史，再由真实 Ollama 回答；其他回忆为对话式。保留了两类字面匹配的误判：S2 的 `0,545` 和“不得增加任何依赖”，以及 S4 的“既无代码也无约束”。事实复核将它们与 S2 首次真实的引用拒绝区分。最终代码的 S2 返回 ORION、0.545 和正确约束，压缩 12 次且零工具；`real_memory_final.json` 中仍保留其改述导致的 `success: false`。

首个调用的路由，数据集 `routing_cases.json`：历史依次为 **1/3**、**2/3**、**3/3**，保存在 `m7_routing_before.json`、`m7_routing_after.json`、`m7_routing_final.json`。已知及缺失信息的回忆均无工具；混合请求选择 `get_git_diff`。最终代码复验也通过 **3/3**，延迟为 **8,408/9,648/6,816 s**。这不衡量工具执行或最终回答质量，也不声称延迟改善。最终验证同样复现 **3/3**、零错误，延迟 **8,1779/8,9203/6,5370 s**，均值 **7,8784 s**。协议仍只评估首个调用；未重跑 S5/S6 或 Qwen 留出集。但空会话案例返回了未出现在任何传入历史中的“测试代码 `test_1`”和“优先级 `P1`”。无工具路由正确，内容却是编造的。已知回忆找回 ORCHIDEE-9274 和绿松石色。这一新增生成局限既不是会话污染，也不能证明回忆忠实；3/3 严格限定为路由得分。

历史 MCP 子集 `no_tool_01`、`read_file_03`、`recovery_01`、`multi_step_01` 在最后的引用修正前取得路由/参数 **4/4**、端到端 **2/4**。它不能代替 M6 的 20 项任务。被拒绝的 `multi_step_01` 为 S2 读取中的定义引用了 S1 搜索。之后，引言“以下是第 1 至 40 行的摘要：”被作为 claim 拒绝，这是 M5 提取局限，未增加特例绕过。一次旧恢复回放曾被接受（六条 claims，**86,347 s**，其中最终生成 **71,725 s**，grounding **14,619 s**），但接受了关于原始分数而非 RRF 排名的不精确表述。它既不验证自主决策，也不证明验证器普遍准确。

在相同 MCP 来源和候选回答下比较 M6/M7：内容、路径和引用保持完整；正确引用的候选回答在两者中均被接受，多来源但无引用的候选回答均被拒绝。此比较中的 grounder 为测试替身。没有证明 Devstral 的遗漏由压缩造成。部分拒答检查确实修正了继承的 S9 接受缺陷。生成式修复实验得到 **0/3 完整修复**（**17,699/17,517/17,446 s**），此前首次通用尝试为 **22,376 s** 且没有修正。该实验已从交付代码移除。

<a id="performances-ollama-mesurées"></a>

#### Ollama 实测性能

观测参数：`keep_alive=30m`、`num_ctx=16384`、`num_predict=512`、温度 0、seed 42、HTTP 超时 300 s；Agent：五步，每种工具重试一次，工具超时 60 s。先预加载 Qwen，再预加载 Devstral；预热失败仍允许会话继续。本次最终验证没有强制卸载模型，也未改变模型、grounding、超时或 GPU 设置。

| 回放，此阶段不含工具耗时 | 最终生成 | Grounding | 墙钟时间 | 结果 |
|---|---:|---:|---:|---|
| FAISS 1 | 36,042 s | 300,034 s | 336,078 s | 超时 |
| FAISS 2 | 15,107 s | 300,002 s | 315,111 s | 超时 |
| read_file 1 | 47,454 s | 无调用 | 47,456 s | 缺少引用 |
| read_file 2 | 20,048 s | 300,012 s | 320,061 s | 超时；遗漏恢复 |
| FAISS v2, 1 | 157,167 s | 10,680 s | 167,849 s | 已接受，四条 claims 有依据 |
| FAISS v2, 2 | 33,564 s | 6,587 s | 40,152 s | 已接受，相同事实与来源 |
| FAISS v2, 3 | 33,468 s | 4,798 s | 38,267 s | 已接受，相同事实与来源 |
| read_file v2, 1 | 20,270 s | 1,302 s | 21,574 s | 诊断被接受，遗漏恢复 |
| read_file v2, 2 | 61,830 s | 无调用 | 61,832 s | 缺少恢复引用 |
| read_file v2, 3 | 61,191 s | 无调用 | 61,193 s | 缺少恢复引用 |

FAISS v2 首次提示词评估为 **121 140,123 ms**，加载 **471,416 ms**，生成 **34 663,701 ms**；第 2/3 次提示词为 **283,549/285,094 ms**，生成为 **33 132,572/33 020,726 ms**。初始成本主要来自提示词，而非 157 s 的加载。S8：Qwen 加载 **193 261,843 ms**，提示词 **7 617,799 ms**，生成 **269,208 ms**，墙钟时间 **201,359 s**。这不能反推此前超时的原因。

澄清后的 S6 自主运行：**98,767/97,551/98,094 s**，工具 **0,103/0,013/0,013 s**，MCP setup **0,740/0,726/0,703 s**，引用被拒绝后没有 grounding。加入保护后：**99,700 s**，agent **60,473 s**，最终生成 **38,244 s**，工具 **0,030 s**，setup **0,757 s**，其余约 **0,196 s** 包括传输关闭与本地处理。不同验证的路径和输出不同，不声称端到端加速。最初回放汇总中的 `tokens=0` 是报告缺陷：精确计数位于 `diagnostics.requests`，脚本通过 `_refresh_token_stats` 为后续测量修正。

唯一隔离测得的网络改善：通过 `localhost` 发送三次 GET `/api/ps` 耗时 **2,055/2,034/2,032 s**，通过 IPv4 为 **0,0161/0,000935/0,000627 s**。历史 Ollama 0.40.3 服务器监听 IPv4；本机 `localhost` 首先解析为 IPv6。Devstral 声明为 10 % CPU / 90 % GPU，Qwen 为 100 % GPU；声明的 VRAM 大小为 **15 705 047 040** 和 **12 031 780 453 字节**，合计约 25,8 GiB，而 RX 7800 XT 的物理显存为 16 GB。这不证明两者同时驻留。一次验证中可用 RAM 达到 **588 860 KiB**，下一次为 **708 092 KiB**（692 MiB）；观测到 Qwen runner 被替换。内存压力已证实，分页与超时的因果关系未测量。17:07 可访问的日志不对应相关请求。预热阻塞在 `socket.readinto` 的原因仍未知；后续在模型已加载时完成了路由与回忆验证。中断的运行不计入结果。

<a id="limites-connues"></a>

### 已知局限

交付状态中没有已证实而未解决的 M7 bug；测试不保证完全没有缺陷。抽取式记忆保留数据，但可能从摘要中淘汰某条摘录；原始归档仍可用。过大的不可拆分句子保留在归档中，需要定向请求。词法回忆不是通用语义回忆。一个会话假定只有一个写进程；并发和高级磁盘保留仍被推迟。FAISS 索引与 manifest 不具备事务性，重命名仍按删除再添加处理；`ast.unparse` 会规范化文本，即使保留原始行号。

自主能力局限继续保留：FAISS 在未实际检查时可能拒答；真实 `read_file` 恢复 **7/7**，完整最终回答 **0/7**；还存在遗漏或错配引用、遗漏恢复、范围或路径决策不稳定。间接指代可能超出保护范围。没有历史的回忆也可能在无工具路由正确时编造回答，例如最终复验中的 `test_1`/`P1`。无记忆污染不等于没有这种幻觉。记忆不能代替当前来源或来源验证。

独立于 M7 的 grounding 局限：引言与代码块被当作 claims、一个段落的引用被应用到多个句子、错误拒绝 `grounding_output_supported_01`，以及在缺少精确证据时接受“有效”这一限定词和原始分数组合。`final_cases.json` 中的 `m9_regressions` 保留 `weighted_raw_scores`、`exact_rrf_ranks` 和历史错误拒绝案例。阶段结束验证没有用 Qwen 执行这些案例；**39/40** 保持不变。这些记录说明留待 M9 的局限，并不表示启动 M8 或 M9，也不放宽 grounding。模型、本机同时驻留能力和硬件资源限制性能；未经 grounding 而被拒绝的结果不能与通过 grounding 的回答直接比较。

<a id="commandes-de-reproduction"></a>

### 复现命令

从根目录激活安装了 `requirements.txt` 的环境：Windows 使用 `.\.venv\Scripts\Activate.ps1`；Linux/WSL 使用 `source .venv/bin/activate`。顺序执行；复验输出保存在系统临时目录。测试与记忆基准不需要网络或 Ollama。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest -q
python -m scripts.benchmark_memory
$env:HF_HUB_OFFLINE = '1'
python -m scripts.benchmark_rag --output-dir (Join-Path $env:TEMP 'codeagent-m7-rag-final')
python -m app.main --help
python -m scripts.benchmark_memory --help
python -m scripts.benchmark_m7_final --help
python -m compileall -q app scripts tests conftest.py
git diff --check
```

RAG 基准要求 MiniLM 已在本地可用。不要将其离线模式与 embedding 测试替身混淆。针对性检查如下，数量取决于当前测试套件状态：

```powershell
python -m pytest -q tests/test_conversation.py -k "recent_roles or sessions_projects or compaction or reaffirmed or checkpoint"
python -m pytest -q tests/test_conversation.py tests/test_memory_integration.py -k "budget or bounded or large_sources or observation or source or citation"
python -m pytest -q tests/test_memory_integration.py
python -m pytest --cache-show
```

套件还包括历史 parser、embedding、增量索引、manifest、retrieval、引用、grounding、循环、计时和工具优化测试。临时目录测试验证缓存与 `tmp_path` 位于仓库外，且旧 pytest 目录不存在。编译时若要避免本地字节码，将 `PYTHONPYCACHEPREFIX` 指向系统临时路径。

使用 Ollama 时，保留已安装的 `qwen2.5-coder:14b` 和 `devstral:24b`。检查 `ollama list`，必要时在另一终端启动 `ollama serve`，然后依次运行各项测量。测试期间不要并发第二个 Ollama 生成。完整 holdout 按其冷启动协议有意卸载并预加载 Qwen。

```powershell
ollama ps
Invoke-RestMethod http://127.0.0.1:11434/api/ps | ConvertTo-Json -Depth 8
python -m scripts.benchmark_tool_routing --dataset benchmarks/memory/routing_cases.json --output (Join-Path $env:TEMP 'codeagent-m7-routing-final.json')
python -m scripts.benchmark_m7_final --mode memory --output (Join-Path $env:TEMP 'codeagent-m7-real-memory-final.json')
python -m scripts.benchmark_m7_final --mode finals --repetitions 3 --output (Join-Path $env:TEMP 'codeagent-m7-fixed-finals.json')
python -m scripts.benchmark_m7_final --mode autonomous --repetitions 3 --output (Join-Path $env:TEMP 'codeagent-m7-autonomous-final.json')
python -m scripts.benchmark_m7_final --mode unsupported --output (Join-Path $env:TEMP 'codeagent-m7-unsupported-final.json')
python -m scripts.benchmark_grounding_models --models qwen2.5-coder:14b --output (Join-Path $env:TEMP 'codeagent-m7-grounding-holdout.json')
```

脚本的字面标记仍须辅以对候选回答及各条 claim 的复核。超时是异常，绝不等于 `UNSUPPORTED` 判定。在诊断硬件前保留 `/api/ps` 状态、Ollama 指标、RAM/VRAM、墙钟计时和 `--debug` 日志。

<a id="validation-manuelle"></a>

### 手动验证

创建专用会话；日志位于仓库外，应用记忆仍在 `data/memory/`：

```powershell
python -m app.main --project . --session m7-personal-final --debug 2>&1 |
    Tee-Object -FilePath (Join-Path $env:TEMP 'codeagent-m7-personal-final.log')
```

选择 **2**（Agent），每个问题单独占一行：

1. `Dans cette session, retiens le code ASTER-8617 et la contrainte aucune dépendance ajoutée. Réponds brièvement.`
2. 提出三个中间问题，然后发送 `Quel code et quelle contrainte ai-je mentionnés dans cette session ?`
3. 输入空行退出，停止进程，重新执行相同命令并再次询问。之后更换 `--session`：事实应不存在，工具调用为零。
4. `Explique-moi brièvement le rôle de FAISS dans CodeAgent.` 检查实际执行的工具、来源、引用、grounding 和事实；拒答不等于检查成功。
5. `Lis app/retrieval/faiss_retriever.py des lignes 227 à 277. Si la plage est invalide, corrige-la.` 检查错误、真实恢复的读取，以及引用诊断和恢复的最终回答；不得虚构动作或用额外问题掩盖失败。

回忆应包含 ASTER-8617 和精确约束，零工具，`Grounding: not_used`。技术最终回答应有来源支持的事实和 `Grounding: passed`。retriever 当前长 226 行：起始 227 会被拒绝，过大的结束位置会被限制；该文件变更时应复核协议。上述 `memory`、`finals`、`autonomous`、`unsupported` 模式分别复现连续性/压缩/恢复/隔离、固定来源的最终回答、自主选择，以及拒绝编造的 claim。

针对两个同名项目，在仓库外创建目录：

```powershell
$validationRoot = python -c "import tempfile; print(tempfile.mkdtemp(prefix='codeagent-m7-projects-'))"
$projectA = Join-Path $validationRoot 'a/same'
$projectB = Join-Path $validationRoot 'b/same'
New-Item -ItemType Directory -Force -Path $projectA,$projectB | Out-Null
python -m app.main --project $projectA --session m7-project-isolation --debug
python -m app.main --project $projectB --session m7-project-isolation --debug
python -m app.main --project $projectA --session m7-project-isolation --debug
```

选择 2。在 A 中记录 `PROJET-A-6842` 和“不得增加任何依赖”，然后退出。在 B 中要求回忆：应不存在。在 A 的新进程中：两条事实应存在。即使项目与会话同名，检查点也必须不同。测试 RAG 后续问题时选择 **0**，询问 `Où est définie read_dir ?`，再问 `Que fait cette fonction ?`；仍须使用当前来源并验证 grounding。`tests/test_memory_integration.py` 中的 stdio 测试不依赖模型，复现无效范围后再有效读取。

<a id="conservation-des-preuves"></a>

### 证据保留

仓库中的测试夹具、数据集和 JSON 均予保留，包括 1/3 和 2/3 的路由、39/40 留出集，以及两套前后 RAG 语料。默认记忆命令仅重生成 `benchmarks/memory/results/m7.json`；首次验证的数字已保留于上文。额外复验输出位于仓库外。

最近一次来源验证的详细转录均保留且未删除，位于 `E:\AI\Codex\visualizations\2026\10\14\01a13ba8-ed1f-7023-b685-925cddea4da1`：`memory.json`、`m6_comparison.json`、`real_memory.json`、`real_memory_after.json`、`real_memory_final.json`、`finals.json`、`finals_after.json`、`autonomous.json`、`autonomous_after.json`、`autonomous_guard.json`、`unsupported.json`、`citation_repair_real.json`、`citation_repair_real_after.json`、`routing.json`、`routing_final.json`、`resources.json`、`resume_validation.json` 和 `rag/`。此前验证仍位于 `E:\AI\Codex\visualizations\2026\10\14\01a13b6e-3ccd-7ae1-8a01-2139da70a6c3`。这些本地文件不会包含在 push 中；本文保留有用测量、失败、局限及其命令。不把历史成功称为本次新复现的成功。本次最终验证新增的 RAG 与路由输出位于 `E:\AI\Codex\visualizations\2026\10\14\01a13c4d-cb46-7350-b684-666f26f69d24`（`rag/`、`routing.json`）。

<a id="décisions-techniques-principales"></a>

## 主要技术决策

- 本地运行；
- 使用Python AST结构化代码；
- `all-MiniLM-L6-v2`用于嵌入；
- 使用FAISS作为向量索引；
- 检索向量后进行词法重排序；
- 使用Ollama执行本地LLM；
- `Qwen2.5-Coder 14B`在基准测试后被保留；
- 明确区分检索与生成；
- 文件/行来源的系统性保存；
- FAISS索引和分块的本地持久化；
- SHA-256清单用于检测项目变更；
- 增量更新以仅重新计算已修改的文件；
- 分析历史记录以JSONL格式存储；
- 混合密集+词典检索；
- 引用的自动验证；
- 使用本地LLM验证来源依据核验性；
- MCP用于暴露和执行只读工具；
- `Devstral 24B`被保留用于工具调用；
- `Qwen2.5-Coder 14B`被保留用于M6来源依据核验验证，基于专项基准测试；
- 预加载Ollama和`keep_alive`以限制冷启动，当模型可驻留时；
- 已评估专用路由器但未集成，Devstral仍更精确。

<a id="état-du-projet-et-suite"></a>

## 项目状态与后续计划

RAG 基础在 M5 完成，受控工具使用在 M6 完成。

M7 已完成记忆与上下文范围内的工作，其自主能力和硬件局限见上文。后续计划是 M8 — Planning & Controlled Autonomous Execution，再到 M9 — Advanced RAG & Reliability。专用 grounding 仍以测得收益为前提。

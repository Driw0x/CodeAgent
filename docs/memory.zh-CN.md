[English](memory.md) | [Français](memory.fr.md) | [简体中文](memory.zh-CN.md)

[项目 README](../README.zh-CN.md)

<a id="codeagent--project-memory"></a>
# CodeAgent — 项目记忆

> 项目记忆：功能、决策、结果及M1至M8的限制。

<a id="état-actuel"></a>
## 当前状态

**M1–M7 已完成。M8 在原始受控规划范围内为 PASS；在 `OLLAMA_MAX_LOADED_MODELS=1` 下验证两个真实自主场景。当前 HITL 资格验证为部分完成。M9 尚未开始。**

CodeAgent分析Python项目，索引其代码，生成带引用验证和事实锚定的响应，维护索引，并通过MCP使用只读工具。

M7引入了持久化的对话记忆和共享上下文预算。M8增加了条件规划和受控执行，并支持重新规划。

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
问题
→ 嵌入向量
→ FAISS 检索
→ 重排序
→ 选择代码块
→ 构建上下文
→ Qwen2.5-Coder 14B
→ 带来源信息的回答
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
Python 项目
→ 读取文件
→ 当前清单
→ 与上一份清单比较
→ 新增／修改／删除的文件
→ 仅重新计算受影响的代码块
→ 更新 FAISS 索引
→ 保存索引与新清单
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

在最终的16个问题测试集上，M5将`Hit@5`和`Recall@5`的性能从`0.6667`提升至`0.8333`，并将回答/回避行为得分从`0.6250`提升至`0.7500`。`MRR@5`从`0.5069`下降至`0.3917`，且锚定验证在管道中平均增加了`2.54 s`。

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

**M7已完成**，用于内存和上下文管理；自主最终结果的可靠性保持独立。

- `ConversationMemory`保留最近的上下文，并自动将旧轮次压缩为有序且优先的摘要，无需LLM进行总结。
- 只追加式存档`history.jsonl`保存原始的交互和观察记录；原子检查点`sessions/<session_id>.json`可在发生损坏时实现恢复和从JSONL文件重新构建。
- 拒绝未来版本或超前于存档的检查点，且不覆盖原有内容。对话记忆与项目索引保持分离。
- `ContextBudget` 在记忆、RAG、工具与 LLM 之间共享上下文预算，预留生成空间，并以 UTF-8 字节数进行保守估算；某个必需元素过大时会明确拒绝。
- 项目解析路径和`--session`隔离了对话，包括同名项目之间的对话。
- 当前源优先于历史记录；选择与压缩过程保留了来源、路径、行和值。最终结果与来源依据核验均使用相同的保留源。
- 旧交互不构成当前代码的证明：修改过的源必须重新审查。错误仅证明其诊断内容。

[确定性内存基准测试](../benchmarks/memory/results/m7.json) 验证了 **9/9 回忆和 6/6 控制测试**：上下文边界、回忆、提取保真度、源/来源依据核验、检索和无摘要模型。历史真实测试保留了 **7/7 读取`read_file`的原始内容，但0/7完整的自主最终结果**，尤其在引用或遗漏检索方面；这些数据并非M8修正后的重跑结果。旧交互的检索仍为词法和线性方式；语义搜索仍为可选。有限摘要可排除部分片段，但原始内容仍被存档。会话假设仅一个写作者进程，且不自动清除存档。记忆无法保证模型的准确回忆或最终结果的正确来源。

<a id="m8--planning--controlled-autonomous-execution"></a>
## M8 — 规划与受控自主执行

**原始里程碑范围内 PASS。** M8 提供条件短计划、最小状态、真实 MCP 步骤、只恢复剩余工作、保留进度、明确停止及与 M6 直接执行对比。原始目标见 Git `0bf0437` 的 `README.md`。可选 HITL 的验证范围另列如下；不宣称通用自主可靠性。

<a id="m8-contract"></a>
### 架构与契约

- `app/agent/planning.py` 负责词法路由、计划验证、状态和证据认证。`ToolLoop` 复用唯一 M6 MCP 循环、M7 记忆及 M5/M6 引用/grounding。简单请求保留直接路径；`--planning auto` 为默认，`off` 关闭规划，`adaptive` 仍为实验模式。
- 规划器返回 JSON `{goal, steps}` 或 `{"stop":"user_decision","reason":"..."}`。计划含 2–6 步，每步仅有 `id`、`objective`、`depends_on`、`success`；ID 唯一，依赖只能向前，独立步骤允许 `[]`。
- 目标保留请求中无歧义的摘录和文件路径，仅有界规范化动词大小写及重音。覆盖全部收集动作。协调多文件读取保留完整目标，每文件一步，明确指定 `success.path`。
- `success` 只允许 `tool`、`kind`、`value`、`from_step`、`path`；工具为 `read_file`、`search_code`、`list_files`、`get_git_diff`、`run_tests`；条件为 `contains`、`nonempty`、`observed`。发现的路径必须来自 `from_step` 指定的已完成依赖；拒绝虚构路径。
- `observed` 记录真实结果，包括空搜索/diff、测试失败，不证明最终任务成功。错误或超时不能完成步骤。请求实现时必须读取完整函数体，签名或模板不能代替。
- 重新规划只替换剩余工作，保留已完成步骤及证据。每次生成至多一次验证修复，共用重规划配额。无效批量调用仅获一次有界提醒，验证前绝不执行。重试、去重及预算始终共享。
- 来源保留原始观测 ID、文件、行号及参数。选择可重排 `[Sx]`；`source_mapping` 连接最终引用与观测。当前证据优先于历史；人工指令从不成为证据。最终生成与 grounding 使用相同保留来源。
- `goal_achieved` 要求收集完成、引用有效、grounding 通过、格式/覆盖符合请求，且无无理由弃答。明确 pytest 数量及返回码与所引用结果对照。仅有收集、`COMPLETED` 或 `SUPPORTED` 不能证明最终成功。

<a id="m8-hitl"></a>
### HITL 与恢复

`--hitl off` 为默认。`on-failure` 在计划、步骤或最终回答出现可观察拒绝后暂停。同一任务保留 MCP 连接、来源、完成步骤、错误、去重和计数器，结束时只归档一个 M7 轮次。

| 菜单 | 动作 |
|---|---|
| 1–4 | 检查有界且遮蔽的状态；提供指导；在配额内重试；停止并给出未验证部分报告。 |
| 5–7 | 选择已观测路径；提供项目路径并保留其他参数；以已验证 JSON 参数执行现有只读工具。 |
| 8 | 修改可识别的剩余步骤，可用 `{"remove":true}` 删除一步；执行前原子验证完整计划。 |

已完成步骤不可重写。权限、JSON Schema、项目边界、依赖、来源、去重、引用及 grounding 继续生效。直接动作消耗现有调用/迭代配额。每任务最多三次干预，每条指导最多 2000 字符，不重置预算。人工等待计入总时限；同步输入在返回时检查过期。无终端时返回 `intervention_required`。

被拒绝的最终回答允许使用相同来源额外生成一次，不重新收集。grounding 异常允许对同一候选/来源重验一次，与 `UNSUPPORTED` 区分。`resumed` 仅证明授权，须检查后续调用及最终结果。重启后恢复对话不等于恢复中断执行。

<a id="m8-corrections"></a>
### 原因与保留修正

| 已证实原因 | 保留修正 |
|---|---|
| 动词大小写/NFC/NFD 重音、文件名被当作动作、不必要依赖 | 限定识别且不改变位置/路径；允许独立步骤；给出逐步诊断和准确的多文件修复摘录。 |
| 错误测试文件、矛盾进程状态、收集被当作完成 | 执行/认证前验证请求 pytest 路径，要求一致状态，保留覆盖及误成功检查。 |
| 修正后搜索被丢弃；签名/模板替代函数体 | 保留恢复观测，分开诊断/内容；使用 `total_lines`、AST 完整性及就地引用的准确代码。 |
| 最终提示要求虚构的未完成部分或工具叙述 | 传递已验证计划/证据，要求带本地引用的事实句，grounding 前拒绝无理由弃答。 |
| H 生成了工具调用但 `/api/chat` 为空 | 原始重放揭示 Ollama 0.40.8 解析丢失不属于当前步骤的 `read_file`。把当前步骤放在观测后，保留工具过滤/验证。 |
| 空响应统计丢失；Devstral 驻留时 Qwen 加载超时 | 拒绝内容前记录统计，保留 `done_reason`；测量机器使用原生 `OLLAMA_MAX_LOADED_MODELS=1`，不增加应用模型管理器或超时。 |

<a id="m8-validation"></a>
### 验证与范围

**最终审计：799 个测试通过**，其中 **509 个相关测试**；记忆 **6/6**，规划 **6/6**，涵盖 **16 场景 / 32 次执行**；语法、导入、CLI 选项及 `git diff --check` 通过。本次整合不改变应用/脚本功能；历史真实证据仍适用于运行时代码，测试文件已重组。确定性测试使用脚本化模型/grounding、真实内存/stdio MCP 和真实 pytest，验证不变量，不衡量 Ollama 自主性。

**2026 年 10 月 23 日**最终真实运行使用 Ollama **0.40.8**、`devstral:24b`、`qwen2.5-coder:14b`，温度 **0**，种子 **42**，上下文 **16384**，生成上限 **512**，HTTP 超时 **300 s**，`keep_alive=30m`，`OLLAMA_MAX_LOADED_MODELS=1`。共享预算：**12 次迭代、6 步、2 次重规划、16 次 MCP、32768 tokens、900 s**，工具超时 **60 s**。MCP 是真实 stdio 子进程，未脚本替换模型响应或 grounding 结论。

| 真实场景 | 验证结果 |
|---|---|
| 主请求 `main-final`，HITL `on-failure` | **PASS AUTONOMOUS**，**3 次 MCP**，**248.454 s**，零干预/重复/最终重试；一次自主计划修复。完整读取两文件并真实 pytest，覆盖/引用/格式/grounding 有效；独立审查 **13/13**。 |
| 未改动的历史 H，`H-final`，HITL `off` | **PASS AUTONOMOUS**，**4 次 MCP**，**274.461 s**，零干预/重规划/重复。实现及测试已读取，正确引用 `marker` 返回 42 和 `assert marker() == 42`；独立审查 **13/13**。 |

准确主请求如下：

> Lis app/agent/planning.py et tests/test_planning.py, puis exécute les tests de tests/test_planning.py. Fournis un bilan factuel en citant les fichiers consultés et le résultat réel des tests.

该版本读取行范围 **1–596**、**1–505**。MCP：`131 passed, 1 warning in 0.74s`，返回码 **0**，`passed=true`、`timed_out=false`、stderr 为空。独立 pytest：`131 passed, 1 warning in 0.69s`。原有 `cache_dir` 警告由禁用 pytest 缓存插件导致。这些是测试重组前的历史数量。交付引用 S1/S2 标识两文件，S3 标识真实 pytest 结果。

**HITL：当前版本资格验证为 PARTIAL。** 确定性暂停/恢复测试通过。**10 月 18 日** Ollama + 内存 MCP 的历史运行包括 **PASS ASSISTED F**（3 次 MCP、2 次干预、**258.028 s**）与 **PASS ASSISTED E2**（3 次 MCP、3 个预授权动作、**552.317 s**）。已记录聊天决策在快照后提供操作/指导，从不替换模型响应、计划或最终回答。这些旧版本的有限成功复现性不完整，后续 F 及 H 辅助最终回答失败。10 月 23 日辅助收集/最终重试保留证据，但最终回答失败。最后两个零干预成功**不能**验证当前代码新的完整暂停/恢复周期。

**实验决策：**自适应/全局试验严格成功均为 **1/4**（14541 / 28720 tokens），**KEEP GLOBAL**。引导最终回答两变体均交付 **3/10**，额外 **10.44%** tokens，**KEEP CURRENT**，移除引导/修订原型。原生 JSON Schema 改善原始 JSON，不改善严格成功率（各 **1/4**；14836 / 16012 tokens，**+7.93%**），**KEEP CURRENT**，移除原型及 15 个测试；生产无 `--planning-format`。结构不证明来源/语义，未同时使用 `format` 与原生 `tools`。未执行留出集不计成功或失败。详情：[自主审查](../benchmarks/planning/results/autonomy-20261017-review.json)、[最终回答审查](../benchmarks/planning/results/finalization-20261017-review.json)、[结构化审查](../benchmarks/planning/results/structured-20261017-review.json)、[结构化协议](../benchmarks/planning/structured-protocol.json)。

<a id="m8-reproduction"></a>
### 复现与证据

仓库根目录使用已安装依赖/模型，将新结果写到仓库外：

```powershell
$py = '.venv/Scripts/python.exe'
$out = Join-Path $env:TEMP ('codeagent-m8-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
& $py -m pytest -q
& $py -m scripts.benchmark_memory --output (Join-Path $out 'memory.json')
& $py -m scripts.benchmark_planning --output (Join-Path $out 'planning.json')
& $py -m app.main --project . --planning auto --hitl on-failure --debug --session ('m8-' + [guid]::NewGuid())
```

选择 **2**，提交上述准确请求。H 和可选辅助失败场景见[规划说明](../benchmarks/llm/README.zh-CN.md#memory-and-planning-protocols)。基准脚本的真实运行需要 `--real`；上述 `app.main` 直接使用 Ollama，无此选项。脚本重放须明确区分。在测量机器上以进程级 `$env:OLLAMA_MAX_LOADED_MODELS='1'` 启动已安装 Ollama。保留既有服务，之后恢复初始状态，仅卸载测试创建的模型/进程。历史临时服务器已停止，剩余 Ollama 进程为零，未改持久设置。

仓库外的最终本地证据：

```text
E:/AI/Codex/visualizations/2026/10/23/01a16a21-572d-7520-b967-5111912d7a82/m8-final
```

`main-final/`、`H-final/` 保存协议、HTTP 请求/响应、MCP 轨迹、来源、耗时、结果和独立审查。`tested-source/`、哈希、`session-changes.patch` 标识验证版本。`final-suite.txt`、`final-memory.json`、`final-planning.json`、`final-check-exits.json` 保存历史检查。`diagnostic.json`、`H-raw-exact.json`、`H-corrected/` 保留解析/加载失败。归档 `run_campaign.py`、`review_run.py`、`review_history.py` 用新输出目录复现/审查。克隆不包含本地制品。

更早证据仍在 `data/memory/m8-autonomy-20261017`、`m8-finalization-20261017`、`m8-structured-20261017`（含 `prototype.zip`、`prototype-files.json`、`manifest.json`）、`m8-hitl-20261017`、`m8-hitl-executable-20261018/final-delivery`、`m8-hitl-validation-20261018-853fc1c4/E-02`；最后两目录保留辅助成功、决策及审查。10 月 23 日失败审计在可视化目录 `01a169cb-76e8-7e91-aadc-6510601947a4/m8-hitl-audit`、`01a16a07-8d01-7f31-9519-9b7b30be1a38/m8-final`。历史失败不会被改写为成功。

<a id="m8-limitations"></a>
### 剩余限制

两次成功验证对应场景/配置，不证明通用自主能力或未见任务泛化。词法路由、证据认证和格式检查有限，grounding 可能误接受/拒绝语义声明。首个计划仍需自主修复。当前恢复通过确定性验证，尚未重验完整当前 Ollama 辅助周期。加载依赖硬件，无单模型限制的真实 grounding 超时仍保留。人工等待可耗尽总时间，同步输入和重启恢复仍受上述限制。未增加第二 MCP 循环、写入工具、框架或 M9 实现。

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

M1–M7 已完成。M8 在原始范围内 PASS，验证限制见上。M9 — Advanced RAG & Reliability 仍为未来工作；后续变化须有可测量收益。

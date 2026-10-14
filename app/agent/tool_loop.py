import asyncio
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from app.llm import LocalLLM
from app.memory.context import ContextBudget, fit_sources
from app.memory.conversation import ConversationMemory
from app.memory.history import describe_error
from app.rag.citations import extract_citations, is_abstention, validate_citations
from app.rag.grounding import claims_have_citations, extract_claims, verify_claim
from app.rag.prompt import SYSTEM_PROMPT, format_chunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOL_MODEL = "devstral:24b"
DEFAULT_GROUNDING_MODEL = os.getenv("CODEAGENT_GROUNDING_MODEL", "qwen2.5-coder:14b")
DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_RETRIES = 1
DEFAULT_TOOL_TIMEOUT = 60.0
DEFAULT_SEARCH_RESULTS = 10
READ_WINDOW_BEFORE = 15
READ_WINDOW_AFTER = 35

TOOL_SYSTEM_PROMPT = """Tu es un agent spécialisé dans l'analyse de projets de code.

Tu peux utiliser les tools disponibles lorsque cela est nécessaire.

Règles :
- Utilise les observations des tools comme sources factuelles.
- Une observation réussie peut contenir un identifiant de source [S1], [S2], etc.
- Toute affirmation technique ou factuelle fondée sur une observation doit citer immédiatement la source correspondante au format exact [Sx].
- Ne cite jamais une source qui n'a pas été fournie.
- N'invente pas de fichier, fonction, comportement, état du projet ou fonctionnalité.
- Ne transforme pas un objectif, une roadmap ou une fonctionnalité prévue en fonctionnalité déjà implémentée.
- Ne déduis pas une information qui n'est pas explicitement supportée par les observations.
- Si les sources ne permettent pas de répondre, indique : "Je ne peux pas le déterminer à partir des sources disponibles."
- Si aucun tool n'est nécessaire, réponds normalement sans citation.
- Pour une question exclusivement sur les informations mentionnées dans cette conversation ou session, utilise uniquement son historique, sans tool projet.
- Les décisions, contraintes, identifiants et valeurs fournis par l'utilisateur sont des faits conversationnels : rappelle-les sans citation [Sx], même après compression. Applique leur dernière correction. Les numéros de tours ne sont pas des identifiants de sources.
- Pour expliquer le rôle d'un composant dans le projet ou décrire son code actuel, inspecte le projet avec les tools dans ce tour. Sans observation actuelle, ne présente pas de fait sur le projet.
- Si ces informations sont absentes de l'historique, indique que tu ne les retrouves pas dans cette session. Ne les cherche pas dans les fichiers, Git ou une autre session.
- Si la question combine un rappel conversationnel et une inspection explicite du projet, utilise les tools pour cette inspection.
- N'annonce pas une inspection à venir : appelle le tool demandé dans ce même tour avant de conclure.
- Utilise search_code pour localiser précisément un symbole avant de lire un gros fichier.
- Dès que les observations disponibles suffisent pour répondre, arrête d'appeler des tools.
- Après utilisation des tools, termine par un message bref : la réponse sourcée sera générée séparément à partir des observations.
- N'appelle pas un tool uniquement pour reconfirmer une information déjà obtenue.
"""


def build_server_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
        env={key: os.environ[key] for key in ("CODEAGENT_PROJECT_PATH", "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE") if key in os.environ},
    )


def mcp_tool_to_ollama(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def tool_result_content(result) -> str:
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    if texts:
        return "\n".join(texts)
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)
    return ""


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def tool_call_key(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)


def _collect_json_search_hints(value, hints: list[tuple[str, int]]):
    if isinstance(value, dict):
        path = value.get("path") or value.get("file")
        line = value.get("line") or value.get("line_number") or value.get("start_line")
        if isinstance(path, str):
            try:
                line_number = int(line)
            except (TypeError, ValueError):
                line_number = None
            if line_number is not None and line_number > 0:
                hints.append((normalize_path(path), line_number))
        for child in value.values():
            _collect_json_search_hints(child, hints)
    elif isinstance(value, list):
        for child in value:
            _collect_json_search_hints(child, hints)


def extract_search_hints(content: str) -> list[tuple[str, int]]:
    hints = []

    try:
        parsed = json.loads(content)
        _collect_json_search_hints(parsed, hints)
    except (json.JSONDecodeError, TypeError):
        pass

    pattern = re.compile(r"(?P<path>[A-Za-z0-9_./\\-]+\.py):(?P<line>\d+)")

    for match in pattern.finditer(content):
        hints.append((normalize_path(match.group("path")), int(match.group("line"))))

    unique = []
    seen = set()

    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        unique.append(hint)

    return unique


def question_requests_full_file(question: str) -> bool:
    text = question.casefold()
    markers = (
        "fichier entier",
        "fichier complet",
        "tout le fichier",
        "contenu entier",
        "contenu complet",
        "full file",
        "whole file",
        "entire file",
    )
    return any(marker in text for marker in markers)


def optimize_tool_arguments(
    name: str,
    arguments: dict,
    question: str,
    search_hints: dict[str, list[int]],
) -> dict:
    optimized = dict(arguments)

    if name == "search_code":
        optimized.setdefault("max_results", DEFAULT_SEARCH_RESULTS)
        return optimized

    if name != "read_file" or question_requests_full_file(question):
        return optimized

    if "start_line" in optimized and "end_line" not in optimized:
        try:
            start_line = int(optimized["start_line"])
            optimized["end_line"] = start_line + READ_WINDOW_BEFORE + READ_WINDOW_AFTER
        except (TypeError, ValueError):
            pass
        return optimized

    if "start_line" in optimized or "end_line" in optimized:
        return optimized

    path = optimized.get("path")

    if not isinstance(path, str):
        return optimized

    path = normalize_path(path)
    lines = search_hints.get(path)

    if not lines:
        return optimized

    line = lines[0]
    optimized["start_line"] = max(1, line - READ_WINDOW_BEFORE)
    optimized["end_line"] = line + READ_WINDOW_AFTER

    return optimized


def build_tool_source(name: str, arguments: dict, content: str) -> dict:
    data = {}

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            data = parsed
    except (json.JSONDecodeError, TypeError):
        pass

    hints = extract_search_hints(content) if name == "search_code" else []

    if name == "search_code" and hints:
        file = hints[0][0]
    else:
        file = data.get("path") or arguments.get("path") or f"tool/{name}"

    if not isinstance(file, str):
        file = f"tool/{name}"

    file = normalize_path(file)

    if name == "search_code" and hints:
        start_line = hints[0][1]
    else:
        start_line = data.get("start_line", arguments.get("start_line", 1))

    if not isinstance(start_line, int) or start_line < 1:
        start_line = 1

    end_line = data.get("end_line", arguments.get("end_line"))

    if not isinstance(end_line, int) or end_line < start_line:
        end_line = start_line

    observation = (
        f"Tool: {name}\n"
        f"Arguments: {json.dumps(arguments, ensure_ascii=False)}\n"
        f"Result:\n{content}"
    )
    source = {
        "file": file,
        "type": "tool",
        "name": name,
        "content": observation,
        "start_line": start_line,
        "end_line": end_line,
    }

    if hints:
        paths = sorted({path for path, _ in hints})
        lines_by_path = {}
        for path, line in hints:
            lines_by_path.setdefault(path, []).append(line)
        source["search_paths"] = paths
        source["search_lines"] = lines_by_path

    return source


def consolidate_sources(sources: list[dict]) -> list[dict]:
    read_sources = [source for source in sources if source.get("name") == "read_file"]
    result = []

    for source in sources:
        if source.get("name") != "search_code":
            result.append(source)
            continue

        search_paths = source.get("search_paths", [])

        if len(search_paths) != 1:
            result.append(source)
            continue

        path = normalize_path(search_paths[0])
        lines = source.get("search_lines", {}).get(path, [])

        if not lines:
            result.append(source)
            continue

        covered = False

        for read_source in read_sources:
            read_path = normalize_path(str(read_source.get("file", "")))
            if read_path != path:
                continue
            start_line = read_source.get("start_line", 1)
            end_line = read_source.get("end_line", start_line)
            if all(start_line <= line <= end_line for line in lines):
                covered = True
                break

        if not covered:
            result.append(source)

    return result


def build_tool_answer_prompt(question: str, sources: list[dict]) -> str:
    # Show failures before the recovered results, without changing citation IDs.
    ordered = sorted(enumerate(sources, start=1), key=lambda item: item[1]["type"] != "tool_error")
    context = "\n\n---\n\n".join(
        format_chunk(source, source_id)
        for source_id, source in ordered
    )
    successful = ", ".join(f"[S{index}]" for index, source in enumerate(sources, 1) if source["type"] != "tool_error") or "aucun"
    failed = ", ".join(f"[S{index}]" for index, source in enumerate(sources, 1) if source["type"] == "tool_error") or "aucun"
    return f"""CONTEXTE (observations d'appels déjà exécutés) :

{context}

BILAN DES APPELS EXÉCUTÉS :
Appels réussis, données effectivement retournées : {successful}.
Appels échoués, diagnostics uniquement : {failed}.

QUESTION:

{question}

INSTRUCTION:
Réponds à toutes les parties explicites de la question en utilisant uniquement les sources disponibles.
Chaque affirmation factuelle doit être accompagnée de la source [Sx] qui la supporte.
Une observation en erreur prouve seulement l'échec et son diagnostic.
Après récupération, indique la correction exécutée et le résultat obtenu, sans proposer d'appel supplémentaire.
Une plage initiale invalide n'annule pas les données d'une lecture réussie. Rapporte les bornes effectivement lues et cite aussi le paragraphe décrivant cette correction.
Réponds brièvement : résume les données récupérées et cite chaque paragraphe.
"""


def normalize_single_source_citations(answer: str, source_count: int) -> str:
    if source_count != 1 or not answer.strip() or not extract_claims(answer):
        return answer
    paragraphs = [
        paragraph.strip()
        for paragraph in answer.strip().split("\n\n")
        if paragraph.strip()
        and paragraph.strip() != "[S1]"
    ]
    normalized = []
    for paragraph in paragraphs:
        if not extract_citations(paragraph):
            paragraph = f"{paragraph} [S1]"
        normalized.append(paragraph)
    return "\n\n".join(normalized)


class ToolLoop:
    def __init__(
        self,
        llm: LocalLLM | None = None,
        grounding_llm: LocalLLM | None = None,
        grounding_model: str = DEFAULT_GROUNDING_MODEL,
        server=None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        memory=None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be greater than or equal to 1.")

        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")

        if tool_timeout <= 0:
            raise ValueError("tool_timeout must be greater than 0.")

        self.llm = llm or LocalLLM(model=DEFAULT_TOOL_MODEL)
        self.grounding_llm = grounding_llm or LocalLLM(model=grounding_model)
        self.server = server or build_server_parameters()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.tool_timeout = tool_timeout
        self.budget = ContextBudget.for_llm(self.llm)
        self.memory = memory if memory is not None else ConversationMemory(max_tokens=max(512, self.budget.input_limit // 3))
        self.debug = os.getenv("CODEAGENT_DEBUG") == "1"
        self.last_debug = {}
        self.last_trace = []
        self.last_steps = 0
        self.last_grounding = "not_used"
        self.last_token_stats = {}
        self.last_stage_stats = {}
        self.last_stage_times = {}
        self._reset_metrics()

    @staticmethod
    def _usage_snapshot(llm) -> dict:
        usage = getattr(llm, "usage_stats", {}) or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    def _reset_metrics(self):
        if hasattr(self.llm, "reset_usage_stats"):
            self.llm.reset_usage_stats()

        if hasattr(self.grounding_llm, "reset_usage_stats"):
            self.grounding_llm.reset_usage_stats()

        self.last_token_stats = {
            "devstral_prompt": 0,
            "devstral_completion": 0,
            "grounding_prompt": 0,
            "grounding_completion": 0,
            "total": 0,
        }
        self.last_stage_stats = {
            "agent": {"prompt_tokens": 0, "completion_tokens": 0},
            "final": {"prompt_tokens": 0, "completion_tokens": 0},
            "grounding": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.last_stage_times = {
            "mcp_setup": 0.0,
            "agent": 0.0,
            "tools": 0.0,
            "final": 0.0,
            "grounding": 0.0,
        }

    def _measure_llm(self, stage: str, llm, call):
        before = self._usage_snapshot(llm)
        start = perf_counter()

        try:
            return call()
        finally:
            seconds = perf_counter() - start
            self.last_stage_times[stage] += seconds
            after = self._usage_snapshot(llm)
            if self.debug:
                self.last_debug.setdefault("requests", []).append({
                    "stage": stage, "model": getattr(llm, "model", None), "seconds": seconds,
                    "tokens": {key: after[key] - before[key] for key in before},
                    "ollama": getattr(llm, "last_request_stats", None) if after != before else None,
                })
            self.last_stage_stats[
                stage
            ]["prompt_tokens"] += (
                after["prompt_tokens"]
                - before["prompt_tokens"]
            )
            self.last_stage_stats[
                stage
            ]["completion_tokens"] += (
                after["completion_tokens"]
                - before["completion_tokens"]
            )

    def _refresh_token_stats(self):
        devstral = getattr(self.llm, "usage_stats", {}) or {}
        grounding = getattr(self.grounding_llm, "usage_stats", {}) or {}
        devstral_prompt = devstral.get("prompt_tokens", 0)
        devstral_completion = devstral.get("completion_tokens", 0)
        grounding_prompt = grounding.get("prompt_tokens", 0)
        grounding_completion = grounding.get("completion_tokens", 0)
        self.last_token_stats = {
            "devstral_prompt": devstral_prompt,
            "devstral_completion": devstral_completion,
            "grounding_prompt": grounding_prompt,
            "grounding_completion": grounding_completion,
            "total": (
                devstral_prompt
                + devstral_completion
                + grounding_prompt
                + grounding_completion
            ),
        }

    def _finalize_answer(self, answer: str, source_chunks: list[dict]) -> str:
        claims = extract_claims(answer)
        citations = extract_citations(answer)
        # Missing references can be resolved deterministically only with one source,
        # and only after every claim has actually passed the existing verifier.
        repair = bool(claims) and any(not ids for _, ids in claims) and len(source_chunks) == 1
        if self.debug:
            self.last_debug.pop("reason", None)
            self.last_debug.update(candidate=answer, sources=source_chunks, claims=[])

        if is_abstention(answer) and not claims and not citations:
            self.last_grounding = "passed"
            if self.debug:
                self.last_debug["outcome"] = "abstention"
            return answer

        invalid = bool(citations) and not validate_citations(answer, len(source_chunks))
        if invalid or (not claims_have_citations(answer) and not repair):
            self.last_grounding = "citation_error"
            if self.debug:
                self.last_debug["reason"] = "invalid_references" if invalid else "missing_references"
            return "La réponse générée contient " "des citations invalides ou manquantes."

        supported = bool(claims)
        for claim, source_ids in claims:
            source_ids = source_ids or [1]
            before = self._usage_snapshot(self.grounding_llm)
            start = perf_counter()
            supported = None
            try:
                supported = self._measure_llm(
                    "grounding", self.grounding_llm,
                    lambda: verify_claim(claim, source_ids, source_chunks, self.grounding_llm),
                )
            finally:
                if self.debug:
                    after = self._usage_snapshot(self.grounding_llm)
                    self.last_debug["claims"].append({
                        "claim": claim, "source_ids": source_ids, "supported": supported,
                        "seconds": perf_counter() - start,
                        "tokens": {key: after[key] - before[key] for key in before},
                        "ollama": getattr(self.grounding_llm, "last_request_stats", None) if after != before else None,
                    })
            if not supported:
                break

        if not supported:
            self.last_grounding = "unsupported"
            if self.debug:
                self.last_debug["reason"] = "verifier_rejected_claim" if claims else "no_verifiable_claims"
                if repair:
                    self.last_debug["citation_repair"] = "rejected"
            return (
                "La réponse générée contient "
                "des affirmations non supportées "
                "par les sources."
            )

        self.last_grounding = "passed"
        if repair:
            answer = normalize_single_source_citations(answer, len(source_chunks))
        if self.debug:
            self.last_debug["outcome"] = "valid_sources"
            self.last_debug["accepted_answer"] = answer
            if repair:
                self.last_debug["citation_repair"] = "verified"
        return answer

    def _generate_final_answer(self, question: str, source_chunks: list[dict], observations: list[dict]) -> str:
        final_sources = consolidate_sources(source_chunks)
        # Preserve execution failures as evidence, after successful source IDs.
        tool_observations = [message for message in observations if message["role"] == "tool"]
        for trace, observation in zip(self.last_trace, tool_observations):
            if trace["status"] in {"error", "exception", "timeout"}:
                failure = build_tool_source(
                    trace["tool"], trace["arguments"], observation["content"],
                )
                failure["type"] = "tool_error"
                final_sources.append(failure)
        if self.debug:
            self.last_debug["sources_before_selection"] = final_sources
        history = self.memory.recall(question, self.budget.input_limit // 4)
        final_sources, prompt = fit_sources(
            question, final_sources, build_tool_answer_prompt, SYSTEM_PROMPT, self.budget, history, reduce_tools=True
        )
        if self.debug:
            self.last_debug.update(final_prompt=prompt, sources=final_sources,
                                   source_mapping={f"S{index}": source.get("observation_id")
                                                   for index, source in enumerate(final_sources, 1)})
            self.last_debug["final_context"] = {
                "estimated_tokens": self.budget.measure([
                    {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}
                ]), "input_limit": self.budget.input_limit,
                "history_included": bool(history and prompt.startswith(history)),
            }
        if not final_sources:
            self.last_grounding = "passed"
            return "Je ne peux pas le déterminer à partir des sources disponibles."
        answer = self._measure_llm(
            "final",
            self.llm,
            lambda: self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT),
        )
        return self._finalize_answer(answer, final_sources)

    async def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        observations = []
        self.last_debug = {}
        try:
            answer = await self._ask(question, observations)
        except Exception as error:
            self.memory.record(question, f"Erreur non résolue : {describe_error(error)}", observations, diagnostics=self.last_debug)
            raise
        self.memory.record(question, answer, observations, diagnostics=self.last_debug)
        return answer

    async def _ask(self, question: str, observations: list[dict]) -> str:

        self.last_trace = []
        self.last_steps = 0
        self.last_grounding = "not_used"
        self._reset_metrics()
        messages = observations
        seen_calls = set()
        retry_counts = {}
        source_chunks = []
        search_hints = {}

        try:
            mcp_start = perf_counter()

            async with Client(self.server) as client:
                listed = await client.list_tools()
                self.last_stage_times["mcp_setup"] += (perf_counter() - mcp_start)
                tools = [mcp_tool_to_ollama(tool) for tool in listed.tools]

                for _ in range(self.max_steps):
                    self.last_steps += 1
                    context = self.memory.prepare(question, TOOL_SYSTEM_PROMPT, self.budget, tools, messages)
                    if self.debug:
                        self.last_debug.setdefault("agent_contexts", []).append(context)
                    assistant_message = self._measure_llm(
                        "agent",
                        self.llm,
                        lambda: self.llm.chat(messages=context, tools=tools),
                    )
                    if self.debug:
                        self.last_debug.setdefault("agent_responses", []).append(assistant_message)
                    tool_calls = assistant_message.get("tool_calls") or []

                    if not tool_calls:
                        content = assistant_message.get("content", "",).strip()

                        if source_chunks:
                            return self._generate_final_answer(
                                question=question,
                                source_chunks=source_chunks,
                                observations=messages,
                            )

                        if not content:
                            raise RuntimeError("LLM returned neither " "content nor tool calls.")

                        if self.last_trace:
                            # No successful observation: never trust a generated claim of recovery.
                            self.last_grounding = "passed"
                            if self.debug:
                                self.last_debug["outcome"] = "abstention_tool_failure"
                            return "Je ne peux pas le déterminer à partir des sources disponibles."
                        if extract_citations(content):
                            self.last_grounding = "citation_error"
                            if self.debug:
                                self.last_debug.update(conversation_candidate=content, reason="invalid_conversation_references")
                            return "La réponse générée contient des citations invalides ou manquantes."
                        # ponytail: explicit project/inspection cues; indirect references still rely on the tool router.
                        project_name = Path(os.getenv("CODEAGENT_PROJECT_PATH") or PROJECT_ROOT).name
                        project_pattern = r"\b(?:projet|project)\b"
                        if project_name:
                            project_pattern += "|" + re.escape(project_name)
                        project = re.search(project_pattern, question, re.IGNORECASE)
                        conversation = re.search(r"\b(?:session|conversation|ai-je|avais-je|did I|have I)\b", question, re.IGNORECASE)
                        inspection = re.search(r"\b(?:lis|read|inspecte|inspect)\b", question, re.IGNORECASE)
                        target = project or re.search(r"[\w./\\-]+\.(?:py|md|json)\b|\b(?:read_file|search_code|get_git_diff|run_tests|list_files)\b", question, re.IGNORECASE)
                        technical = re.search(r"\b(?:rôle|role|composant|component|fonction|function|classe|class|code|implémentation|implementation)\b", question, re.IGNORECASE)
                        if (inspection and target) or (project and technical and not conversation):
                            self.last_grounding = "passed"
                            if self.debug:
                                self.last_debug.update(conversation_candidate=content, outcome="abstention_no_current_project_source")
                            return "Je ne peux pas le déterminer à partir des sources disponibles."
                        if self.debug:
                            self.last_debug["outcome"] = "conversation_without_documentary_citation"
                        return content

                    messages.append(assistant_message)
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        name = function.get("name")
                        arguments = function.get("arguments", {})

                        if not name:
                            raise RuntimeError("LLM returned a tool call " "without a name.")

                        if not isinstance(arguments, dict):
                            raise RuntimeError("LLM returned invalid " "tool arguments.")

                        arguments = optimize_tool_arguments(
                            name=name,
                            arguments=arguments,
                            question=question,
                            search_hints=search_hints,
                        )
                        trace = {
                            "tool": name,
                            "arguments": arguments,
                            "status": None,
                            "duration_seconds": 0.0,
                        }
                        self.last_trace.append(trace)
                        call_key = tool_call_key(name, arguments)

                        if call_key in seen_calls:
                            trace["status"] = "duplicate"
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "error": "duplicate_tool_call",
                                            "message": (
                                                "The same tool was already "
                                                "called with the same arguments."
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue

                        if retry_counts.get(name, 0) > self.max_retries:
                            trace["status"] = "retry_limit"
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "error": "retry_limit_reached",
                                            "tool": name,
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue

                        seen_calls.add(call_key)
                        tool_start = perf_counter()

                        try:
                            result = await asyncio.wait_for(
                                client.call_tool(name, arguments),
                                timeout=self.tool_timeout,
                            )
                        except TimeoutError:
                            retry_counts[name] = retry_counts.get(name, 0) + 1
                            trace["status"] = "timeout"
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "error": "tool_timeout",
                                            "tool": name,
                                            "message": (
                                                f"Tool '{name}' exceeded "
                                                f"{self.tool_timeout} seconds."
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue
                        except Exception as error:
                            retry_counts[name] = retry_counts.get(name, 0) + 1
                            trace["status"] = "exception"
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "error": "tool_exception",
                                            "tool": name,
                                            "message": str(error),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue
                        finally:
                            duration = perf_counter() - tool_start
                            trace["duration_seconds"] = duration
                            self.last_stage_times["tools"] += duration

                        if result.is_error:
                            retry_counts[name] = retry_counts.get(name, 0) + 1
                            trace["status"] = "error"
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "error": "tool_error",
                                            "tool": name,
                                            "message": tool_result_content(result),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue

                        trace["status"] = "success"
                        content = tool_result_content(result)

                        if name == "search_code":
                            for path, line in extract_search_hints(content):
                                search_hints.setdefault(path, []).append(line)

                        source_chunks.append(
                            build_tool_source(
                                name=name,
                                arguments=arguments,
                                content=content,
                            )
                        )
                        source_id = len(source_chunks)
                        source_chunks[-1]["observation_id"] = f"S{source_id}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "content": json.dumps(
                                    {
                                        "ok": True,
                                        "tool": name,
                                        "source": f"[S{source_id}]",
                                        "result": content,
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )

            if source_chunks:
                return self._generate_final_answer(question=question, source_chunks=source_chunks, observations=messages)

            raise RuntimeError(f"Maximum number of tool steps " f"reached ({self.max_steps}).")
        finally:
            self._refresh_token_stats()

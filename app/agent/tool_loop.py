import asyncio
import json
import os
import re
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

from mcp import Client, StdioServerParameters
from mcp.types import TextContent
from jsonschema import validate as validate_schema, ValidationError

from app.llm import LocalLLM
from app.memory.context import ContextBudget, fit_sources
from app.memory.conversation import ConversationMemory
from app.memory.history import describe_error
from app.rag.citations import CODE_PATTERN, extract_citations, is_abstention, mask_code, validate_citations
from app.rag.grounding import claims_have_citations, extract_claims, verify_claim
from app.rag.grounding import build_claim_prompt, GROUNDING_SYSTEM_PROMPT
from app.rag.prompt import SYSTEM_PROMPT, format_chunk, output_contract_satisfied, requests_source_code
from .hitl import MAX_GUIDANCE, MAX_INTERVENTIONS, terminal_intervention
from .planning import (ACTION_PATTERN, ExecutionStopped, PLAN_SYSTEM, READ_ONLY_TOOLS, PlanningExecution,
                       adaptive_requirements, answer_code_observed, answer_covers_actions, correct_remaining_plan, evidence_paths, observed_read_paths, planning_decision, requested_paths, result_data, validate_plan, verifies)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOL_MODEL = "devstral:24b"
DEFAULT_GROUNDING_MODEL = os.getenv("CODEAGENT_GROUNDING_MODEL", "qwen2.5-coder:14b")
DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_RETRIES = 1
DEFAULT_TOOL_TIMEOUT = 60.0
DEFAULT_SEARCH_RESULTS = 10
READ_WINDOW_BEFORE = 15
READ_WINDOW_AFTER = 35

M6_COMPLETION_RULES = """- Dès que les observations disponibles suffisent pour répondre, arrête d'appeler des tools.
- Après utilisation des tools, termine par un message bref : la réponse sourcée sera générée séparément à partir des observations."""

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
- Pour lire une implémentation, lis aussi son corps : la seule ligne def/class ne suffit pas. Si sa taille est inconnue, omets end_line pour obtenir la fenêtre de lecture prévue.
""" + M6_COMPLETION_RULES + """
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
        "contenu exact",
        "exact content",
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
        definitions = []
        for match in data.get("matches", []):
            if not isinstance(match, dict) or not isinstance(match.get("file"), str) or not isinstance(match.get("line"), int):
                continue
            definition = re.match(r"^\s*(?:(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)\b|([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=))", str(match.get("text", "")))
            if definition:
                definitions.append(dict(match, file=normalize_path(match["file"]), name=definition[1] or definition[2]))
        source["definition_matches"] = definitions
        # Resolve only exact symbol definitions in observed lines; imports/usages are not definitions.
        query = re.fullmatch(r"(?:(?:async\s+)?def\s+|class\s+)?([A-Za-z_]\w*)", str(arguments.get("query", "")).strip())
        if query:
            source["definition_paths"] = sorted({match["file"] for match in definitions if match["name"] == query[1]})

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


def build_tool_answer_prompt(question: str, sources: list[dict], *, assisted=False, plan=()) -> str:
    # Show failures before the recovered results, without changing citation IDs.
    ordered = sorted(enumerate(sources, start=1), key=lambda item: item[1]["type"] != "tool_error")
    context = "\n\n---\n\n".join(
        format_chunk(source, source_id)
        for source_id, source in ordered
    )
    successful = ", ".join(f"[S{index}]" for index, source in enumerate(sources, 1) if source["type"] != "tool_error") or "aucun"
    failed = ", ".join(f"[S{index}]" for index, source in enumerate(sources, 1) if source["type"] == "tool_error") or "aucun"
    progress = ("\nÉTAT MESURÉ DE LA COLLECTE (la réponse finale reste à vérifier) :\n"
                + json.dumps([{key: step[key] for key in ("id", "objective", "status")} for step in plan], ensure_ascii=False)
                + "\nLa boucle valide ces actions exécutées ; les fichiers lus n'ont pas à déclarer que la demande utilisateur est accomplie.\n") if plan else ""
    recovery = ""
    if failed != "aucun":
        recovery = ("Après récupération, indique la correction exécutée et le résultat obtenu, sans proposer d'appel supplémentaire.\n"
                    "Une plage initiale invalide n'annule pas les données d'une lecture réussie. Rapporte les bornes effectivement lues et cite aussi le paragraphe décrivant cette correction.")
    if assisted:
        recovery = ("Restitue le contenu et les faits effectivement trouvés qui répondent à la demande initiale. "
                    "Cite les observations de recherche et de lecture nécessaires aux différentes étapes. "
                    "Décris le déroulement des tools ou les corrections uniquement si la question demande explicitement ce compte rendu. "
                    "Ne remplace pas le résultat demandé par un constat d'exécution.")
    return f"""CONTEXTE (observations d'appels déjà exécutés) :

{context}

BILAN DES APPELS EXÉCUTÉS :
Appels réussis, données effectivement retournées : {successful}.
Appels échoués, diagnostics uniquement : {failed}.
{progress}

QUESTION:

{question}

INSTRUCTION:
Réponds directement en phrases factuelles, sans titre ni liste. Attache une citation [Sx] à chaque fait.
{recovery}
"""


def normalize_single_source_citations(answer: str, source_count: int) -> str:
    if source_count != 1 or not answer.strip() or not extract_claims(answer):
        return answer
    # Skip code spans before locating paragraph breaks. Only insert references;
    # never strip/rebuild code or move an existing citation between claims.
    tokens = list(CODE_PATTERN.finditer(answer))
    if any((token["ticks"] and not token["ticks_end"])
           or (token["tildes"] and not token["tildes_end"])
           or (token["inline"] and not token["inline_end"]) for token in tokens):
        return answer  # An unclosed span has no safe place for a citation.
    boundaries = [match.start() for match in re.finditer(r"\r?\n[ \t]*\r?\n", mask_code(answer))] + [len(answer)]
    normalized = []
    start = 0
    for end in boundaries:
        paragraph = answer[start:end]
        prose = paragraph
        for token in reversed(tokens):
            if start <= token.start() < end:
                prose = prose[:token.start() - start] + prose[token.end() - start:]
        local_claims = extract_claims(answer[:end])
        if paragraph.strip() and not extract_citations(prose) and local_claims and not local_claims[-1][1]:
            insertion = len(paragraph.rstrip())
            closing_fence = any(not token["inline"] and start <= token.start() < end
                                and not answer[token.end():start + insertion].strip() for token in tokens)
            if closing_fence:
                insertion = len(paragraph.rstrip("\r\n"))
            separator = ("\r\n" if "\r\n" in answer else "\n") if closing_fence else " "
            paragraph = paragraph[:insertion] + separator + "[S1]" + paragraph[insertion:]
        normalized.append(paragraph)
        start = end
    return "".join(normalized)


def is_pure_session_recall(question: str) -> bool:
    # ponytail: explicit retrospective wording only; ambiguous requests still use the router.
    past = re.search(
        r"\b(?:ai-je|avais-je|avons-nous)\s+(?:\w+\s+){0,2}"
        r"(?:indiqu\w*|demand\w*|choisi\w*|reten\w*|pris\w*|dit|donn\w*|mentionn\w*|enregistr\w*|défini\w*|précis\w*)\b"
        r"|\b(?:did|have)\s+(?:I|we)\s+(?:choose|chosen|say|said|tell|told|mention\w*|specif\w*|ask\w*|select\w*)\b"
        r"|\b(?:rappelle(?:-moi)?|remind me|recall)\b.*\b(?:ma|mon|mes|notre|nos|session|conversation|my|our)\b",
        question, re.IGNORECASE,
    )
    additional_task = ACTION_PATTERN.search(question) or re.search(
        r"\b(?:retiens|mémorise|note|souviens-toi|rappelle-toi|remember|record|écris|rédige|génère|ajoute|crée|calcule|traduis|résous|propose|write|create|add|calculate|translate|solve|suggest)\b",
        question, re.IGNORECASE,
    )
    return bool(past and not additional_task)


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
        planning="auto",
        max_plan_steps=6,
        max_replans=2,
        max_tool_calls=16,
        max_total_tokens=32768,
        max_seconds=900.0,
        hitl="off",
        intervention_handler=None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be greater than or equal to 1.")

        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")

        if tool_timeout <= 0:
            raise ValueError("tool_timeout must be greater than 0.")
        if planning not in {"auto", "off", "adaptive"} or not 2 <= max_plan_steps <= 6:
            raise ValueError("planning must be auto/off/adaptive; max_plan_steps must be between 2 and 6.")
        if max_replans < 0 or min(max_tool_calls, max_total_tokens, max_seconds) <= 0:
            raise ValueError("Execution budgets must be positive; max_replans must be nonnegative.")
        if hitl not in {"off", "on-failure"}:
            raise ValueError("hitl must be off/on-failure.")

        self.llm = llm or LocalLLM(model=DEFAULT_TOOL_MODEL)
        self.grounding_llm = grounding_llm or LocalLLM(model=grounding_model)
        self.server = server or build_server_parameters()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.tool_timeout = tool_timeout
        self.budget = ContextBudget.for_llm(self.llm)
        self.memory = memory if memory is not None else ConversationMemory(max_tokens=max(512, self.budget.input_limit // 3))
        self.planning = planning
        self.max_plan_steps = max_plan_steps
        self.max_replans = max_replans
        self.hitl = hitl
        self.intervention_handler = intervention_handler or terminal_intervention
        self._human_guidance = []
        self._active_sources = []
        self._verified_claims = set()
        self._human_paths = {}
        self.execution_limits = dict(max_tool_calls=max_tool_calls, max_total_tokens=max_total_tokens, max_seconds=max_seconds)
        self.last_execution = {}
        self.execution = None
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
            "planner": {"prompt_tokens": 0, "completion_tokens": 0},
            "agent": {"prompt_tokens": 0, "completion_tokens": 0},
            "final": {"prompt_tokens": 0, "completion_tokens": 0},
            "grounding": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.last_stage_times = {
            "planner": 0.0,
            "mcp_setup": 0.0,
            "agent": 0.0,
            "tools": 0.0,
            "final": 0.0,
            "grounding": 0.0,
        }

    def _measure_llm(self, stage: str, llm, call, context=None, tools=None):
        execution = self.execution
        reserve = 0
        if execution:
            budget = ContextBudget.for_llm(llm)
            reserve = budget.check(context, tools) + budget.response_tokens
            execution.check(reserve)
        before = self._usage_snapshot(llm)
        start = perf_counter()
        timeout = getattr(llm, "timeout", None)
        if execution and isinstance(timeout, (int, float)):
            llm.timeout = min(timeout, execution.state["budgets"]["seconds"])

        try:
            return call()
        except Exception:
            self._failed_stage = stage
            raise
        finally:
            if timeout is not None:
                llm.timeout = timeout
            seconds = perf_counter() - start
            self.last_stage_times[stage] += seconds
            after = self._usage_snapshot(llm)
            if execution:
                actual = sum(after[key] - before[key] for key in before)
                execution.used_tokens += actual if actual > 0 else reserve
                execution.refresh()
            if self.debug:
                self.last_debug.setdefault("requests", []).append({
                    "stage": stage, "model": getattr(llm, "model", None), "seconds": seconds,
                    "tokens": {key: after[key] - before[key] for key in before},
                    "token_measurement": "reported" if after != before else "unavailable",
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
            verification_key = json.dumps([claim, source_ids, source_chunks], sort_keys=True, ensure_ascii=False)
            if self.hitl != "off" and verification_key in self._verified_claims:
                supported = True
                continue
            before = self._usage_snapshot(self.grounding_llm)
            start = perf_counter()
            supported = None
            try:
                supported = self._measure_llm(
                    "grounding", self.grounding_llm,
                    lambda: verify_claim(claim, source_ids, source_chunks, self.grounding_llm),
                    context=[{"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                             {"role": "user", "content": build_claim_prompt(claim, source_ids, source_chunks)}],
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
            if self.hitl != "off":
                self._verified_claims.add(verification_key)

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

        if repair:
            answer = normalize_single_source_citations(answer, len(source_chunks))
            if not claims_have_citations(answer):
                self.last_grounding = "citation_error"
                if self.debug:
                    self.last_debug.update(reason="missing_references", citation_repair="rejected")
                return "La réponse générée contient des citations invalides ou manquantes."
        self.last_grounding = "passed"
        if self.debug:
            self.last_debug["outcome"] = "valid_sources"
            self.last_debug["accepted_answer"] = answer
            if repair:
                self.last_debug["citation_repair"] = "verified"
        return answer

    def _generate_final_answer(self, question: str, source_chunks: list[dict], observations: list[dict]) -> str:
        consolidated = consolidate_sources(source_chunks)
        failed_tools = {trace["tool"] for trace in self.last_trace if trace["status"] in {"error", "exception", "timeout"}}
        # A later read proves code, not the successful retry's arguments or execution.
        final_sources = [source for source in source_chunks if source in consolidated or source["name"] in failed_tools]
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
        system = SYSTEM_PROMPT + self._human_instruction()
        assisted = self.hitl != "off" and bool(self.last_execution["human"]["interventions"])
        builder = partial(build_tool_answer_prompt, assisted=assisted, plan=self.last_execution.get("plan", ()))
        history = self.memory.recall(question, self.budget.input_limit // 4)
        final_sources, prompt = fit_sources(
            question, final_sources, builder, system, self.budget, history, reduce_tools=True
        )
        self._final_sources = final_sources
        if self.debug:
            self.last_debug.update(final_prompt=prompt, sources=final_sources,
                                   user_goal=question, finalization_strategy="baseline",
                                   source_mapping={f"S{index}": source.get("observation_id")
                                                   for index, source in enumerate(final_sources, 1)})
            self.last_debug["final_context"] = {
                "estimated_tokens": self.budget.measure([
                    {"role": "system", "content": system}, {"role": "user", "content": prompt}
                ]), "input_limit": self.budget.input_limit,
                "history_included": bool(history and prompt.startswith(history)),
            }
        if not final_sources:
            self.last_grounding = "passed"
            return "Je ne peux pas le déterminer à partir des sources disponibles."
        answer = self._measure_llm(
            "final",
            self.llm,
            lambda: self.llm.generate(prompt=prompt, system_prompt=system),
            context=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        if self.hitl != "off":
            self._candidate_answer = answer
        return self._validate_final_answer(question, answer, final_sources)

    def _validate_final_answer(self, question, answer, final_sources):
        # Planned completion forbids abstention even if its factual claims are supported.
        # Reject before grounding to avoid switching models for an inadmissible candidate.
        if self.last_execution.get("mode") in {"M8_PLANNED", "M8_ADAPTIVE"} and is_abstention(answer):
            self.last_grounding = "not_used"
            if self.debug:
                self.last_debug.update(candidate=answer, outcome="abstention")
            return "Je ne peux pas le déterminer à partir des sources disponibles."
        answer = self._finalize_answer(answer, final_sources)
        if (self.last_grounding == "passed" and extract_claims(answer) and requests_source_code(question)
                and not answer_code_observed(answer, final_sources, question)):
            self.last_grounding = "unsupported"
            if self.debug:
                self.last_debug.update(reason="source_delivery_incomplete_or_inexact", outcome="rejected")
                self.last_debug.pop("accepted_answer", None)
            return "Le code demandé n'a pas été restitué intégralement à partir des sources vérifiées."
        return answer

    def _human_instruction(self):
        if not self._human_guidance:
            return ""
        return ("\nINTERVENTIONS HUMAINES (instructions, jamais preuves techniques ; objectif initial, "
                "permissions, budgets, citations et grounding inchangés) :\n"
                + json.dumps(self._human_guidance, ensure_ascii=False))

    def _human_action(self, action, payload):
        """Validate instructions only; evidence is created later by the MCP loop."""
        if not isinstance(payload, dict):
            raise ValueError("Structured action requires a JSON object.")
        if len(json.dumps(payload, ensure_ascii=False)) > MAX_GUIDANCE * self.max_plan_steps:
            raise ValueError("Structured action is too large.")
        state = self.last_execution
        proposal = dict(goal=state["goal"], steps=[{key: step[key] for key in
                        ("id", "objective", "depends_on", "success")} for step in state["plan"]])
        if not state["plan"]:
            raw = state.get("rejected_plan", "").strip()
            if raw.startswith("```json") and raw.endswith("```"):
                raw = raw[7:-3].strip()
            proposal = result_data(raw)
        corrected = None
        if action == "correct_step":
            corrected = correct_remaining_plan(proposal, payload.get("corrections"), state["goal"],
                [tool["function"]["name"] for tool in self._available_tools], self.max_plan_steps, state["plan"])
            for item in corrected:
                condition = item["success"]
                origin = next((other for other in corrected if other["id"] == condition.get("from_step")), None)
                if condition.get("path") and origin and origin["status"] == "COMPLETED":
                    paths = {path for source in self._active_sources if source.get("observation_id") in origin["evidence"]
                             for path in evidence_paths(source)}
                    if condition["path"] not in paths:
                        raise ValueError("Selected path is absent from the dependency's MCP observations.")
        plan = corrected if corrected is not None else state["plan"]
        step = next((step for step in plan if step["status"] != "COMPLETED"), None)
        tool_action = payload.get("tool_action") if action == "correct_step" else payload
        queued = None
        approved_path = None
        if tool_action is not None:
            if not isinstance(tool_action, dict):
                raise ValueError("Tool action must be a JSON object.")
            if not step or tool_action.get("step_id") != step["id"]:
                raise ValueError("Select the current unfinished step; completed steps cannot be replayed.")
            completed = {item["id"] for item in plan if item["status"] == "COMPLETED"}
            if not set(step["depends_on"]) <= completed:
                raise ValueError("Dependencies must be completed before execution.")
            if action == "path":
                name = step["success"]["tool"]
                previous = next((trace["arguments"] for trace in reversed(self.last_trace)
                                 if trace["tool"] == name and trace.get("step_id") == step["id"]), {})
                arguments = {**previous, "path": payload.get("path")}
            else:
                name, arguments = tool_action.get("name"), tool_action.get("arguments")
            tool = next((tool["function"] for tool in self._available_tools if tool["function"]["name"] == name), None)
            if not tool or name not in READ_ONLY_TOOLS:
                raise ValueError("Tool is not authorized.")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object.")
            # MCP schemas may omit additionalProperties; reject unknown parameters as well.
            schema = tool["parameters"]
            validate_schema(arguments, schema)
            if set(arguments) - set(schema.get("properties", {})):
                raise ValueError("Unknown tool arguments.")
            if name in {"read_file", "search_code", "list_files", "run_tests"}:
                from app.tools import project_tools
                path = arguments.get("path", "tests" if name == "run_tests" else ".")
                resolved = project_tools._resolve_project_path(path)
                if project_tools._is_ignored(resolved):
                    raise ValueError("Path is not authorized (ignored project directory).")
                path = resolved.relative_to(project_tools.PROJECT_ROOT.resolve()).as_posix()
                arguments = {**arguments, "path": path}
                if name in {"read_file", "run_tests"} and name == step["success"]["tool"]:
                    explicit = requested_paths(step["objective"])
                    if explicit and path not in explicit:
                        raise ValueError("Path conflicts with an explicit path in the objective.")
                    selected = step["success"].get("path")
                    if selected and path != selected:
                        raise ValueError("Path conflicts with the plan's selected evidence path.")
                    if name == "read_file":
                        approved_path = path
            if self._retry_counts.get(name, 0) > self.max_retries:
                raise ValueError("Tool retry budget exhausted; human input cannot reset it.")
            optimized = optimize_tool_arguments(name, arguments, state["goal"], self._search_hints)
            if tool_call_key(name, optimized) in self._seen_calls:
                raise ValueError("Duplicate tool call refused.")
            queued = dict(role="assistant", content="Human instruction; not evidence.",
                          tool_calls=[dict(function=dict(name=name, arguments=arguments))])
        if corrected is not None:
            state["plan"] = corrected
            self.execution.event("human_plan_validated", plan=corrected, preserved=state["completed"])
        if approved_path is not None:
            self._human_paths[step["id"]] = approved_path
        self._pending_human_action = queued
        self.execution.refresh()
        self.execution.event("human_action_validated", action=action, instruction=payload,
                             provenance="human_instruction", executable=queued is not None)

    def _intervene(self, reason, observations, sources, *, final=False, needs_replan=False):
        if self.hitl == "off":
            return False
        execution = self.execution
        execution.refresh()
        state = self.last_execution
        metrics = state["human"]
        budgets = state["budgets"]
        retry_exhausted = (reason in {"repeated_errors", "step_blocked"} and execution.current
                           and self._retry_counts.get(execution.current["success"]["tool"], 0) > self.max_retries
                           and budgets["replans"] <= 0)
        exhausted = (budgets["seconds"] <= 0 or budgets["tokens"] <= 0
                     or (not final and (budgets["iterations"] <= 0 or budgets["tool_calls"] <= 0))
                     or retry_exhausted)
        if exhausted:
            if retry_exhausted:
                state["errors"].append("Reprise refusée : quota de retries de l'outil atteint et aucun replanning disponible.")
                state["stop_reason"] = "repeated_errors"
            if budgets["seconds"] <= 0 or budgets["tokens"] <= 0 or (not final and (budgets["iterations"] <= 0 or budgets["tool_calls"] <= 0)):
                state["stop_reason"] = "global_budget"
                required = ["seconds", "tokens"] + ([] if final else ["iterations", "tool_calls"])
                state["errors"].append("Reprise refusée : budget épuisé (" +
                                       ", ".join(key for key in required if budgets[key] <= 0) + ").")
            execution.event("human_intervention", outcome="budget_exhausted", reason=reason)
            return False
        if metrics["interventions"] >= MAX_INTERVENTIONS:
            execution.event("human_intervention", outcome="intervention_limit", reason=reason)
            return False
        suggestion = ("Refinaliser avec les mêmes sources ; aucune nouvelle collecte."
                      if final else "Corriger la dépendance ou la condition indiquée par le validateur."
                      if reason == "invalid_plan" else
                      "Indiquer un chemin à vérifier ou une recherche autorisée ; aucune étape validée ne sera rejouée.")
        snapshot = dict(goal=state["goal"], reason=reason, error=state["errors"][-1] if state["errors"] else reason,
                        current_step=execution.current, completed=state["completed"], remaining=state["remaining"],
                        plan=state["plan"], budgets=budgets, last_tool=self.last_trace[-1] if self.last_trace else None,
                        evidence=[dict(id=s.get("observation_id"), tool=s["name"], file=s["file"]) for s in sources],
                        sources=sources, errors=state["errors"], raw_plan=state.get("rejected_plan"), suggestion=suggestion,
                        final=final, candidate=getattr(self, "_candidate_answer", None), grounding=self.last_grounding,
                        tools=self._available_tools,
                        tool_failures=dict(self._retry_counts), max_tool_failures=self.max_retries + 1,
                        observed_paths=sorted({path for source in sources for path in evidence_paths(source)}))
        # Do not expose mutable execution state to an input adapter.
        snapshot = json.loads(json.dumps(snapshot, ensure_ascii=False))
        start = perf_counter()
        try:
            action, guidance = self.intervention_handler(snapshot)
        except (EOFError, KeyboardInterrupt):
            action, guidance = "stop", ""
        finally:
            metrics["wait_seconds"] += perf_counter() - start
        execution.refresh()
        if action == "unavailable":
            state["stop_reason"] = "intervention_required"
            execution.event("human_intervention", outcome="unavailable", reason=reason)
            return False
        metrics["interventions"] += 1
        if action == "stop":
            state["stop_reason"] = "human_stopped"
            execution.event("human_intervention", outcome="stopped", reason=reason)
            return False
        structured = action in {"correct_step", "path", "tool"}
        try:
            if structured:
                if final:
                    raise ValueError("Finalization only permits retry or stop; no new collection.")
                self._human_action(action, guidance)
            elif action not in {"guidance", "retry"} or not isinstance(guidance, str) or (action == "guidance" and not 0 < len(guidance.strip()) <= MAX_GUIDANCE):
                raise ValueError(f"Provide between 1 and {MAX_GUIDANCE} guidance characters or select an action.")
            elif needs_replan and budgets["replans"] <= 0:
                raise ValueError("Replanning budget exhausted; correct the proposed steps directly.")
        except (ValueError, TypeError, ValidationError) as error:
            state["errors"].append("Intervention refusée : " + str(error))
            execution.event("human_intervention", outcome="invalid_guidance", reason=reason)
            return self._intervene(reason, observations, sources, final=final, needs_replan=needs_replan)
        if execution.state["budgets"]["seconds"] <= 0:
            state["stop_reason"] = "global_budget"
            execution.event("human_intervention", outcome="budget_exhausted", reason=reason)
            return False
        execution.event("human_intervention", outcome="guidance_provided" if action == "guidance" else "retry_requested",
                        reason=reason, guidance=guidance)
        self._last_human_action = action
        if not structured:
            self._human_guidance.append(guidance if action == "guidance" else suggestion)
        state["stop_reason"] = None
        if execution.current and execution.current["status"] == "FAILED":
            execution.current["status"] = "PENDING"
        execution.event("human_intervention", outcome="resumed", reason=reason,
                        completed=state["completed"], observations=state["observations"], budgets=state["budgets"])
        return True

    def _plan(self, question, tools, observations, replan=False):
        while True:
            try:
                return self._propose_plan(question, tools, observations, replan)
            except ExecutionStopped as error:
                if str(error) not in {"invalid_plan", "user_decision", "replan_limit"} or not self._intervene(
                        str(error), observations, self._active_sources, needs_replan=not bool(self.last_execution["plan"])):
                    raise
                if self._pending_human_action is not None or self._last_human_action == "correct_step":
                    return False
                if self.last_execution["plan"] and self.last_execution["replans"] >= self.max_replans:
                    self.execution.event("plan_retained", reason="replan_limit", plan=self.last_execution["plan"])
                    return False
                replan = True

    def _propose_plan(self, question, tools, observations, replan=False):
        execution = self.execution
        if replan:
            if self.last_execution["replans"] >= self.max_replans:
                execution.stop("replan_limit")
                raise ExecutionStopped("replan_limit")
            self.last_execution["replans"] += 1
        completed = [s for s in self.last_execution["plan"] if s["status"] == "COMPLETED"]
        feedback = []
        for attempt in range(2):
            system = PLAN_SYSTEM + "\nOutils disponibles : " + ", ".join(t["function"]["name"] for t in tools)
            system += "\nÉtat actuel : " + json.dumps({key: self.last_execution[key] for key in ("plan", "errors", "replans")}, ensure_ascii=False)
            system += self._human_instruction()
            context = self.memory.prepare(question, system, self.budget, active=[*observations, *feedback])
            if self.debug:
                self.last_debug.setdefault("planner_contexts", []).append(context)
            response = self._measure_llm("planner", self.llm, lambda: self.llm.chat(messages=context), context=context)
            if self.debug:
                self.last_debug.setdefault("planner_responses", []).append(response)
            try:
                raw = response.get("content", "").strip()
                if raw.startswith("```json") and raw.endswith("```"):
                    raw = raw[7:-3].strip()
                data = json.loads(raw)
                if isinstance(data, dict) and data.get("stop") == "user_decision" and data.get("reason"):
                    execution.stop("user_decision", str(data["reason"]))
                    raise ExecutionStopped("user_decision")
                plan = validate_plan(data, question, [t["function"]["name"] for t in tools], self.max_plan_steps, completed)
                break
            except (ValueError, TypeError) as error:
                if self.hitl != "off":
                    self.last_execution["rejected_plan"] = response.get("content", "")
                    execution.event("plan_rejected", raw=response.get("content", ""), error=str(error))
                if attempt or self.last_execution["replans"] >= self.max_replans:
                    execution.stop("invalid_plan", str(error))
                    raise ExecutionStopped("invalid_plan") from error
                # A single validation repair shares the existing replan/token/time budgets.
                self.last_execution["replans"] += 1
                self.last_execution["errors"].append("Plan refusé : " + str(error))
                execution.event("plan_repair", reason=str(error), count=self.last_execution["replans"])
                feedback = [{"role": "assistant", "content": response.get("content", "")},
                            {"role": "user", "content": (
                                f"Plan refusé par le validateur : {error}\n"
                                "Corrige uniquement ce plan. Renvoie l'objet JSON complet, sans ajouter d'objectif. "
                                "Conserve exactement les étapes déjà COMPLETED et leurs conditions. "
                                "Cette réparation unique reste soumise aux mêmes contraintes de validation."
                            )}]
        self.last_execution["plan"] = plan
        execution.refresh()
        execution.event("replanning" if replan else "plan", plan=plan, preserved=[s["id"] for s in completed], count=self.last_execution["replans"])

    def _verifies_step(self, step, source, sources):
        if self.planning == "adaptive":
            origins = [s for s in sources if s.get("observation_id") != source.get("observation_id")]
            paths = requested_paths(step["objective"])
            if step.get("required_path"):
                paths = {step["required_path"]}
            if (source["name"] == "read_file" and not paths
                    and re.search(r"\b(?:implémentation|implementation|fonction|function|classe|class)\b", step["objective"], re.IGNORECASE)
                    and not any(s.get("definition_matches") for s in origins)):
                return False
            if source["name"] == "read_file" and source["file"] not in (paths or observed_read_paths(step["success"], step["objective"], origins)):
                return False
            # Only observed definitions for the current target can certify its body.
            target = next((s["success"].get("value") for s in reversed(self.last_execution["plan"])
                           if s["status"] == "COMPLETED" and s["success"].get("value")), None)
            if target:
                origins = [dict(s, definition_matches=[m for m in s.get("definition_matches", []) if m["name"] == target]) for s in origins]
            return verifies(step["success"], source["name"], source["content"].partition("Result:\n")[2], step["objective"], origins)
        origin = next((s for s in self.last_execution["plan"] if s["id"] == step["success"].get("from_step")), None)
        origins = [s for s in sources if origin and s.get("observation_id") in origin["evidence"]]
        if origin:
            # The requested symbol comes from the validated objective, not the search query spelling.
            target = origin["success"].get("value", "")
            target = target if re.fullmatch(r"[A-Za-z_]\w*", target) else origin["objective"]
            origins = [dict(s, definition_matches=[match for match in s.get("definition_matches", [])
                       if re.search(rf"\b{re.escape(match['name'])}\b", target)]) for s in origins]
        if source["name"] == "run_tests":
            paths = requested_paths(step["objective"])
            if paths and source["file"] not in paths:
                return False
        if source["name"] == "read_file":
            approved = self._human_paths.get(step["id"])
            paths = {approved} if approved else requested_paths(step["objective"]) or {path for s in origins for path in evidence_paths(s)}
            if step["success"].get("path"):
                paths &= {step["success"]["path"]}
            if origin and not approved:
                paths &= observed_read_paths(step["success"], step["objective"], origins)
            if source["file"] not in paths:
                return False
        return verifies(step["success"], source["name"], source["content"].partition("Result:\n")[2], step["objective"], origins)

    def _planning_progress(self, question, tools, observations, sources):
        if self.planning == "adaptive":
            self._adaptive_progress(sources)
            return
        execution = self.execution
        replan = False
        for trace in self.last_trace[self._processed_traces:]:
            step = execution.current
            if step is None:
                break
            execution.event("tool", **trace)
            if trace["status"] == "success":
                source = next(s for s in sources if s["observation_id"] == trace["observation_id"])
                execution.state["observations"].append(source["observation_id"])
                if self._verifies_step(step, source, sources):
                    step.update(status="COMPLETED", evidence=[source["observation_id"]])
                    execution.event("step_completed", step_id=step["id"], evidence=step["evidence"], file=source["file"])
                else:
                    replan = trace.get("actor") != "human" or trace["tool"] == step["success"]["tool"]
                    diagnostic = "Condition non satisfaite dans les données retournées (ne prouve pas une absence dans le fichier entier)."
                    if trace["tool"] == "read_file":
                        diagnostic += " Pour une implémentation, inclure le corps ; élargir une lecture limitée à sa déclaration."
                    execution.state["errors"].append(diagnostic)
            elif trace["status"] in {"retry_limit", "duplicate"}:
                self._processed_traces = len(self.last_trace)
                execution.stop("repeated_errors", trace["status"])
                raise ExecutionStopped("repeated_errors")
            else:
                tool_messages = [message for message in observations if message.get("role") == "tool"]
                diagnostic = tool_messages[self.last_trace.index(trace)]["content"]
                execution.state["errors"].append(f"{trace['tool']}: {trace['status']}: {diagnostic}")
                failures = self._retry_counts.get(trace["tool"], 0)
                if failures > self.max_retries or (self.hitl != "off" and failures >= self.max_retries):
                    replan = True
            if replan:
                step["status"] = "FAILED"
                execution.event("step_failed", step_id=step["id"], reason=execution.state["errors"][-1])
                break
        self._processed_traces = len(self.last_trace)
        execution.refresh()
        if replan:
            if self.hitl != "off":
                execution.stop("insufficient_evidence")
                if not self._intervene("step_blocked", observations, sources):
                    raise ExecutionStopped(self.last_execution["stop_reason"] or "insufficient_evidence")
                replan = False
        if replan:
            revised = self._plan(question, tools, observations, replan=True)
            # A validated revised plan gets a bounded retry allowance. Seen calls stay blocked.
            for pending in self.last_execution["plan"]:
                if revised is not False and pending["status"] != "COMPLETED":
                    self._retry_counts.pop(pending["success"]["tool"], None)
        step = execution.current
        if step and step["status"] == "PENDING":
            if not set(step["depends_on"]) <= set(execution.state["completed"]):
                execution.stop("insufficient_evidence", "Dépendance non validée")
                raise ExecutionStopped("insufficient_evidence")
            step["status"] = "RUNNING"
            execution.state["current_step"] = step["id"]
            execution.event("step_started", step_id=step["id"], objective=step["objective"])
        elif step is None:
            execution.state["current_step"] = None

    def _adaptive_progress(self, sources):
        execution = self.execution
        for trace in self.last_trace[self._processed_traces:]:
            execution.event("tool", **trace)
            step = execution.current
            if trace["status"] in {"duplicate", "retry_limit"}:
                execution.stop("repeated_errors", trace["status"])
                raise ExecutionStopped("repeated_errors")
            if trace["status"] == "success":
                source = next(s for s in sources if s["observation_id"] == trace["observation_id"])
                execution.state["observations"].append(source["observation_id"])
                if step and self._verifies_step(step, source, sources):
                    step.update(status="COMPLETED", evidence=[source["observation_id"]])
                    execution.event("step_completed", step_id=step["id"], evidence=step["evidence"], file=source["file"])
                else:
                    execution.state["local_revisions"] += 1
                    execution.event("progress_pending", step_id=step["id"] if step else None,
                                    reason="Observation conservée ; obligation courante non prouvée.")
            else:
                execution.state["local_revisions"] += 1
                execution.state["errors"].append(f"{trace['tool']}: {trace['status']}; aucune récupération prouvée")
        self._processed_traces = len(self.last_trace)
        execution.refresh()
        step = execution.current
        execution.state["current_step"] = step["id"] if step else None
        if step and step["status"] == "PENDING":
            step["status"] = "RUNNING"
            execution.event("step_started", step_id=step["id"], objective=step["objective"])
        execution.event("progress_check", completed=execution.state["completed"],
                        remaining=execution.state["remaining"], budgets=execution.state["budgets"])

    def _finish_planned(self, question, sources, observations):
        verification_only = False
        retries = {"final": 0, "grounding": 0}
        while True:
            self._failed_stage = None
            try:
                answer = self._finish_planned_attempt(question, sources, observations, verification_only)
            except ExecutionStopped:
                raise
            except Exception as error:
                if self.hitl == "off" or self._failed_stage not in {"final", "grounding"}:
                    raise
                timeout = False
                cause = error
                while cause:
                    timeout |= isinstance(cause, TimeoutError) or "timed out" in str(cause).casefold()
                    cause = cause.__cause__ or cause.__context__
                self.last_grounding = ("timeout" if timeout else "exception") if self._failed_stage == "grounding" else "not_used"
                reason = ("grounding_timeout" if timeout else "grounding_exception") if self._failed_stage == "grounding" else (
                    "empty_candidate" if "empty" in str(error).casefold() else "final_exception")
                self.last_execution.setdefault("final_attempts", []).append(dict(
                    candidate=self._candidate_answer, grounding=self.last_grounding, reason=reason,
                    exception=dict(type=type(error).__name__, message=str(error)),
                    sources=[source.get("observation_id") for source in self._final_sources]))
                self.execution.stop(reason, describe_error(error))
                answer = self.execution.summary()
            if self.last_execution["stop_reason"] == "goal_achieved":
                return answer
            reason = self.last_execution["final_attempts"][-1].get("reason", "final_rejected") if self.hitl != "off" else "final_rejected"
            stage = "grounding" if reason in {"grounding_timeout", "grounding_exception"} else "final"
            if self.hitl != "off" and retries[stage] >= 1:
                self.execution.event("human_intervention", outcome="final_retry_limit", reason=reason)
                return self.execution.summary()
            if not self._intervene(reason, observations, sources, final=True):
                return self.execution.summary() if self.hitl != "off" else answer
            retries[stage] += 1
            verification_only = stage == "grounding"

    def _finish_planned_attempt(self, question, sources, observations, verification_only=False):
        if verification_only:
            answer = self._validate_final_answer(question, self._candidate_answer, self._final_sources)
        else:
            self._candidate_answer = None
            answer = self._generate_final_answer(question, sources, observations)
        if self.hitl != "off":
            candidate = self._candidate_answer
            reason = ("empty_candidate" if not candidate or not candidate.strip() else
                      "unjustified_abstention" if is_abstention(candidate) else
                      "missing_citations" if not extract_citations(candidate) else
                      "invalid_citations" if self.last_grounding == "citation_error" else
                      "unsupported" if self.last_grounding == "unsupported" else "final_rejected")
            self.last_execution.setdefault("final_attempts", []).append(dict(
                candidate=candidate, answer=answer, grounding=self.last_grounding, reason=reason,
                sources=[source.get("observation_id") for source in self._final_sources]))
        selected = [self._final_sources[index - 1] for index in set(extract_citations(answer))
                    if 1 <= index <= len(self._final_sources)]
        # A final must cite every collected objective, including search evidence consolidated into reads.
        covered = set()
        available = set()
        for source in sources:
            for final in self._final_sources if self.debug else selected:
                data = result_data(final["content"].partition("Result:\n")[2]) or {}
                if source.get("observation_id") == final.get("observation_id"):
                    required = [s for s in self.last_execution["plan"] if source["observation_id"] in s["evidence"]]
                    if all(self._verifies_step(s, final, sources) for s in required):
                        available.add(source["observation_id"])
                        if final in selected:
                            covered.add(source["observation_id"])
                elif source.get("name") == "search_code" and final.get("name") == "read_file" and final.get("type") == "tool":
                    paths = source.get("search_paths", [])
                    lines = source.get("search_lines", {}).get(final["file"], [])
                    excerpts = {item.get("line") for item in data.get("content_excerpts", []) if isinstance(item, dict)}
                    retained = not data.get("truncated") or set(lines) <= excerpts
                    if paths == [final["file"]] and lines and retained and all(final["start_line"] <= line <= final["end_line"] for line in lines):
                        available.add(source["observation_id"])
                        if final in selected:
                            covered.add(source["observation_id"])
        complete = all(set(s["evidence"]) <= covered for s in self.last_execution["plan"])
        actions = answer_covers_actions(self.last_execution["plan"], sources, self._final_sources, answer)
        conforms = output_contract_satisfied(question, answer)
        if self.planning == "adaptive":
            code_observed = answer_code_observed(answer, self._final_sources, question)
            code_present = (not output_contract_satisfied(question, "```python\nexample\n```")
                            or any(not span["inline"] for span in CODE_PATTERN.finditer(answer)))
            self.execution.event("final_check", objective_evidence_covered=complete, actions=actions,
                                 output_contract=conforms, code_observed=code_observed,
                                 requested_code_present=code_present, grounding=self.last_grounding)
            conforms = conforms and code_observed and code_present
        if self.debug:
            self.last_debug["verified_objectives"] = [
                dict(id=step["id"], objective=step["objective"], observation_ids=step["evidence"],
                     retained_verified=set(step["evidence"]) <= available,
                     locally_cited=set(step["evidence"]) <= covered)
                for step in self.last_execution["plan"]]
            self.last_debug["missing_evidence"] = [step["id"] for step in self.last_execution["plan"]
                                                    if not set(step["evidence"]) <= available]
            self.last_debug["validation"] = dict(objective_evidence_covered=complete, actions=actions,
                                                  output_contract=conforms, grounding=self.last_grounding,
                                                  independent_verdict=None)
        if self.last_grounding == "passed" and extract_claims(answer) and not is_abstention(answer) and complete and actions and conforms:
            self.execution.check()
            if self.hitl != "off":
                self.last_execution["final_attempts"][-1]["reason"] = None
            self.execution.stop("goal_achieved")
            if self.debug:
                self.last_debug["final_status"] = "accepted"
            return answer
        if self.debug:
            self.last_debug["final_status"] = "rejected"
        self.execution.stop("insufficient_evidence", f"Finale non validée : grounding={self.last_grounding}, couverture={complete}, actions={actions}, format={conforms}")
        if self.planning == "adaptive" and not conforms:
            return self.execution.summary()
        return self.execution.summary() + "\n\n" + answer

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
        mode, reason = planning_decision(question, self.planning)
        self.execution = PlanningExecution(question, mode, reason, debug=self.debug, max_iterations=self.max_steps,
                                           max_replans=self.max_replans, **self.execution_limits)
        self.last_execution = self.execution.state
        adaptive = mode == "M8_ADAPTIVE"
        planned = mode in {"M8_PLANNED", "M8_ADAPTIVE"}
        self._processed_traces = 0
        self._final_sources = []
        self._human_guidance = []
        self._human_paths = {}
        self._pending_human_action = None
        self._last_human_action = None
        self._candidate_answer = None
        self._verified_claims = set()
        self._available_tools = []
        if self.hitl != "off":
            self.last_execution["human"] = dict(interventions=0, wait_seconds=0.0)
        messages = observations
        seen_calls = set()
        self._seen_calls = seen_calls
        retry_counts = {}
        self._retry_counts = retry_counts
        source_chunks = []
        self._active_sources = source_chunks
        search_hints = {}
        self._search_hints = search_hints
        reminded_steps = set()
        continuation = []

        try:
            if self.memory.total_turns == 0 and is_pure_session_recall(question):
                if self.debug:
                    self.last_debug.update(outcome="empty_session_recall", memory={
                        "session_id": self.memory.session_id, "total_turns": self.memory.total_turns,
                    })
                return "Je ne retrouve aucune information antérieure dans cette session : son historique est vide."
            mcp_start = perf_counter()

            async with asyncio.timeout(self.execution.max_seconds), Client(self.server) as client:
                listed = await client.list_tools()
                self.last_stage_times["mcp_setup"] += (perf_counter() - mcp_start)
                tools = [mcp_tool_to_ollama(tool) for tool in listed.tools]
                if planned:
                    tools = [tool for tool in tools if tool["function"]["name"] in READ_ONLY_TOOLS]
                self._available_tools = tools
                if planned:
                    if adaptive:
                        self.last_execution["plan"] = adaptive_requirements(question)
                        self.last_execution["local_revisions"] = 0
                        if not self.last_execution["plan"] or len(self.last_execution["plan"]) > self.max_plan_steps:
                            self.execution.stop("insufficient_evidence", "Demande hors couverture lexicale ou limite d'objectifs du prototype.")
                            raise ExecutionStopped("insufficient_evidence")
                        self.execution.event("requirements", objectives=self.last_execution["plan"])
                    else:
                        self._plan(question, tools, messages)

                while self.last_steps < self.max_steps:
                    try:
                        self.execution.check()
                        active_tools = tools
                        system = TOOL_SYSTEM_PROMPT
                        if planned:
                            self._planning_progress(question, tools, messages, source_chunks)
                            step = self.execution.current
                            if step is None:
                                return self._finish_planned(question, source_chunks, messages)
                            system = system.replace(M6_COMPLETION_RULES,
                                "- L'étape courante exige un appel réel de son tool, même si les observations précédentes suffisent pour une réponse partielle.\n"
                                "- La synthèse sourcée est produite séparément après validation du plan ; la boucle contrôle le passage entre étapes et l'arrêt.")
                            system += "\nUne seule invocation de tool par réponse. Les dépendances ont été validées."
                            if self.last_execution["errors"]:
                                system += "\nDernier diagnostic d'exécution enregistré : " + json.dumps(self.last_execution["errors"][-1], ensure_ascii=False)
                                system += "\nPour reprendre une étape non validée, appelle son tool avec les arguments corrigés ; une promesse d'action ne suffit pas."
                            active_tools = [tool for tool in tools if tool["function"]["name"] == step["success"]["tool"]]
                            if adaptive:
                                active_tools = tools
                                system = TOOL_SYSTEM_PROMPT.replace(M6_COMPLETION_RULES, "La boucle contrôle la complétion ; une réponse textuelle ne valide aucun objectif.")
                                system += "\nExécution adaptative : choisis UN appel réel à partir des observations actuelles."
                                system += "\nTu peux chercher ou lister pour préparer une lecture, élargir un extrait insuffisant ou corriger une erreur."
                                system += "\nNe devine aucun chemin ; utilise seulement les chemins de la demande ou des observations réussies."
                                system += "\nIndique brièvement la raison du prochain appel dans content. Ne refais pas un appel déjà réussi."
                                system += "\nÉtat mesuré : " + json.dumps({key: self.last_execution[key] for key in ("goal", "plan", "errors", "budgets")}, ensure_ascii=False)
                        system += self._human_instruction()
                        self.last_steps += 1
                        self.execution.iterations = self.last_steps
                        human_action = self._pending_human_action is not None
                        current_instruction = ([{"role": "user", "content": (
                            "Objectif utilisateur inchangé. Exécute seulement cette étape de collecte : "
                            + json.dumps(step, ensure_ascii=False))}] if planned and not adaptive else [])
                        context = [] if human_action else self.memory.prepare(
                            question, system, self.budget, active_tools, [*messages, *continuation, *current_instruction])
                        continuation = []
                        if self.debug:
                            self.last_debug.setdefault("agent_contexts", []).append(context)
                        if human_action:
                            assistant_message = self._pending_human_action
                            self._pending_human_action = None
                        else:
                            assistant_message = self._measure_llm(
                                "agent",
                                self.llm,
                                lambda: self.llm.chat(messages=context, tools=active_tools),
                                context=context, tools=active_tools,
                            )
                        if self.debug:
                            self.last_debug.setdefault("agent_responses", []).append(assistant_message)
                        tool_calls = assistant_message.get("tool_calls") or []
                        if adaptive:
                            self.execution.event("decision_proposed", content=assistant_message.get("content", ""), tool_calls=tool_calls)

                        if not tool_calls or (planned and len(tool_calls) != 1):
                            content = assistant_message.get("content", "",).strip()
                            if planned:
                                try:
                                    decision = json.loads(content)
                                except ValueError:
                                    decision = {}
                                if isinstance(decision, dict) and decision.get("stop") == "user_decision":
                                    self.execution.stop("user_decision", str(decision.get("reason", "Décision manquante")))
                                elif self.execution.iterations >= self.max_steps:
                                    self.execution.stop("step_limit", "Limite d'itérations atteinte avant le rappel de l'étape courante.")
                                elif self.execution.tool_calls >= self.execution.max_tool_calls:
                                    self.execution.stop("global_budget", "Limite globale d'appels de tools atteinte avant le rappel de l'étape courante.")
                                elif step["id"] not in reminded_steps:
                                    reminded_steps.add(step["id"])
                                    self.execution.event("tool_reminder", step_id=step["id"],
                                                         reason="grouped_tool_calls" if tool_calls else "missing_tool_call")
                                    # Rejected calls have no MCP results; never replay an orphan call group.
                                    continuation = ([] if tool_calls else [assistant_message]) + [{"role": "user", "content": (
                                        f"L'étape {step['id']} reste non validée. Cette réponse n'est pas une invocation unique autorisée. Aucun appel de cette réponse n'a été exécuté. "
                                        f"Appelle maintenant {step['success']['tool']} avec des arguments fondés sur la demande et les observations. "
                                        "Une seule invocation réelle, pour l'étape courante ; respecte ses dépendances et le diagnostic fourni. "
                                        "Après ce rappel unique, une réponse sans invocation unique arrêtera l'exécution sans valider l'étape."
                                    )}]
                                    continue
                                else:
                                    self.execution.stop("insufficient_evidence", f"Après un rappel, le modèle n'a pas fourni l'observation nécessaire à l'étape {step['id']}.")
                                raise ExecutionStopped(self.last_execution["stop_reason"])

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

                        if not planned:
                            messages.append(assistant_message)
                        for tool_call in tool_calls:
                            function = tool_call.get("function", {})
                            name = function.get("name")
                            arguments = function.get("arguments", {})

                            if not name:
                                raise RuntimeError("LLM returned a tool call " "without a name.")

                            if not isinstance(arguments, dict):
                                raise RuntimeError("LLM returned invalid " "tool arguments.")
                            if planned and not adaptive and not human_action:
                                condition = step["success"]
                                if name != condition["tool"]:
                                    self.execution.stop("insufficient_evidence", "Tool hors étape refusé : " + name)
                                    raise ExecutionStopped("insufficient_evidence")
                                if name in {"read_file", "run_tests"}:
                                    target_paths = requested_paths(step["objective"])
                                    if condition.get("path"):
                                        target_paths &= {condition["path"]}
                                    if (target_paths or (name == "read_file" and not condition.get("from_step"))) and normalize_path(str(arguments.get("path", ""))) not in target_paths:
                                        self.execution.stop("insufficient_evidence", "Le chemin proposé ne correspond pas à l'objectif de l'étape courante.")
                                        raise ExecutionStopped("insufficient_evidence")
                                if condition.get("from_step"):
                                    origin = next(s for s in self.last_execution["plan"] if s["id"] == condition["from_step"])
                                    origins = [source for source in source_chunks if source.get("observation_id") in origin["evidence"]]
                                    paths = sorted(observed_read_paths(condition, step["objective"], origins))
                                    if len(paths) != 1:
                                        self.execution.stop("user_decision", "Chemin observé absent ou ambigu ; sélectionner explicitement un chemin validé.")
                                        raise ExecutionStopped("user_decision")
                                    if normalize_path(str(arguments.get("path", ""))) != normalize_path(paths[0]):
                                        self.execution.stop("insufficient_evidence", "Le chemin proposé ne correspond pas à la dépendance observée.")
                                        raise ExecutionStopped("insufficient_evidence")

                            if adaptive and name not in {tool["function"]["name"] for tool in active_tools}:
                                self.execution.stop("insufficient_evidence", "Tool non autorisé refusé : " + name)
                                raise ExecutionStopped("insufficient_evidence")
                            if adaptive and name in {"read_file", "search_code", "list_files"}:
                                known = requested_paths(question) | {path for source in source_chunks for path in evidence_paths(source)}
                                allowed = known | {str(parent).replace("\\", "/") for path in known for parent in Path(path).parents}
                                target = normalize_path(str(arguments.get("path", ".")))
                                if target not in allowed | {""}:
                                    self.execution.stop("insufficient_evidence", "Chemin non observé refusé : " + target)
                                    raise ExecutionStopped("insufficient_evidence")

                            if planned:
                                messages.append(assistant_message)
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
                            if planned:
                                trace["step_id"] = step["id"]
                                if human_action:
                                    trace["actor"] = "human"
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
                            self.execution.check()
                            if self.execution.tool_calls >= self.execution.max_tool_calls:
                                trace["status"] = "tool_budget"
                                self.execution.stop("global_budget", "Limite globale d'appels de tools atteinte")
                                raise ExecutionStopped("global_budget")
                            self.execution.tool_calls += 1
                            tool_start = perf_counter()

                            try:
                                result = await asyncio.wait_for(
                                    client.call_tool(name, arguments),
                                    timeout=min(self.tool_timeout, self.last_execution["budgets"]["seconds"]),
                                )
                            except TimeoutError:
                                retry_counts[name] = retry_counts.get(name, 0) + 1
                                if name in READ_ONLY_TOOLS - {"run_tests"}:
                                    seen_calls.discard(call_key)
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
                                if name in READ_ONLY_TOOLS - {"run_tests"} and isinstance(error, (ConnectionError, TimeoutError)):
                                    seen_calls.discard(call_key)
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
                                # MCP may hide unexpected exceptions behind a generic tool error.
                                # A bounded retry is safe for reads, never for run_tests.
                                if name in READ_ONLY_TOOLS - {"run_tests"} and re.search(
                                    r"\b(?:transient|temporary|temporarily|unavailable|connection reset|connection refused|Error executing tool)\b",
                                    tool_result_content(result), re.IGNORECASE,
                                ) and not re.search(r"\b(?:not exist|invalid|must|permission|outside|cannot|validation)\b",
                                                    tool_result_content(result), re.IGNORECASE):
                                    seen_calls.discard(call_key)
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
                            trace["observation_id"] = f"S{source_id}"
                            if planned:
                                trace["file"] = source_chunks[-1]["file"]
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

                    except ExecutionStopped as failure:
                        reason = str(failure)
                        if (not planned or reason not in {"user_decision", "insufficient_evidence", "repeated_errors"}
                                or not self._intervene(reason, messages, source_chunks)):
                            raise
                        # The existing validated plan, completed evidence, seen calls and counters remain intact.
                        continuation = [{"role": "user", "content": self._human_instruction()}]
                        continue
                    except RuntimeError as error:
                        if not planned or self.hitl == "off" or "empty chat response" not in str(error):
                            raise
                        self.execution.stop("empty_decision", describe_error(error))
                        if not self._intervene("empty_decision", messages, source_chunks):
                            raise ExecutionStopped(self.last_execution["stop_reason"])
                        continue

            if planned:
                self._planning_progress(question, tools, messages, source_chunks)
                if self.execution.current is None:
                    return self._finish_planned(question, source_chunks, messages)
                self.execution.stop("step_limit", "Limite d'itérations M6 atteinte avant validation du plan")
                raise ExecutionStopped("step_limit")
            if source_chunks:
                return self._generate_final_answer(question=question, source_chunks=source_chunks, observations=messages)

            raise RuntimeError(f"Maximum number of tool steps " f"reached ({self.max_steps}).")
        except ExecutionStopped:
            return self.execution.summary()
        except TimeoutError:
            self.execution.stop("global_budget", "Délai global atteint, initialisation MCP comprise")
            return self.execution.summary()
        except Exception as error:
            pending = [error]
            while pending:
                nested = pending.pop()
                if isinstance(nested, ExecutionStopped):
                    return self.execution.summary()
                if isinstance(nested, BaseExceptionGroup):
                    pending.extend(nested.exceptions)
            if not planned:
                self.execution.stop("execution_error", describe_error(error))
                raise
            reason = "context_budget" if "Context budget" in str(error) else "execution_error"
            self.execution.stop(reason, describe_error(error))
            return self.execution.summary()
        finally:
            self._refresh_token_stats()
            if not planned and not self.last_execution["stop_reason"]:
                self.execution.stop("direct_response")
            if self.hitl != "off":
                human = self.last_execution["human"]
                success = self.last_execution["stop_reason"] == "goal_achieved"
                events = self.last_execution["events"]
                resumed = next((index for index, event in enumerate(events)
                                if event.get("outcome") == "resumed"), None)
                recovered = resumed is not None and (success or any(
                    event["event"] == "step_completed" for event in events[resumed + 1:]))
                human.update(autonomous_success=success and human["interventions"] == 0,
                             assisted_success=success and human["interventions"] > 0,
                             recovery_success=recovered,
                             completed_objectives=len(self.last_execution["completed"]),
                             tool_calls=self.execution.tool_calls, final_grounding=self.last_grounding)
                self.last_debug["execution"] = self.last_execution
            self.execution.refresh()
            if self.debug:
                self.last_debug["execution"] = self.last_execution
                self.execution.event("summary", completed=self.last_execution["completed"], remaining=self.last_execution["remaining"],
                                     tool_calls=self.execution.tool_calls, replans=self.last_execution["replans"],
                                     stop_reason=self.last_execution["stop_reason"], grounding=self.last_grounding,
                                     tokens=self.last_token_stats if self.last_token_stats["total"] else "unavailable",
                                     budgets=self.last_execution["budgets"])

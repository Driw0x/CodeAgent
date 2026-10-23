"""Bounded evidence plans; tool execution stays in the M6 loop."""

import ast
import json
import re
import unicodedata
from time import perf_counter

from app.rag.grounding import _parse_fragment, extract_claims
from app.rag.citations import CODE_PATTERN
from app.rag.prompt import requests_source_code

READ_ONLY_TOOLS = {"search_code", "read_file", "list_files", "get_git_diff", "run_tests"}
ACTION_PATTERN = re.compile(
    r"(?<![\w./\\-])(?:(?:localis|identifi|recherch|cherch|trouv|inspect|v[eé]\u0301?rifi|expliqu|compar|analys)[^\W\d_]*"
    r"|lis|lire|ex[eé]\u0301?cut(?:e|er|ez)|locate|find|search|read|inspect|check|explain|compare)"
    r"(?!\w|[./\\-]\w)", re.IGNORECASE)
PLAN_SYSTEM = """Produis uniquement un objet JSON {"goal": demande exacte, "steps": [...]}.
Plan court de collecte de preuves, 2 à 6 étapes, sans synthèse (finale séparée).
Chaque étape : {"id":"1", "objective":"extrait EXACT de la demande",
"depends_on":[], "success":{"tool":"search_code", "kind":"contains", "value":"symbole"}}.
kind = contains (texte littéral de la demande), nonempty (données métier présentes)
ou observed (observation exécutée, y compris diff/recherche vide ou tests en échec).
observed ne prouve pas que l'objectif final est satisfait ; les erreurs et timeouts ne le valident pas.
Pour lire un fichier découvert, success.from_step désigne une recherche/lecture antérieure :
le chemin devra provenir de son observation validée. En cas de plusieurs chemins,
success.path peut sélectionner un chemin explicitement observé.
Les étapes indépendantes peuvent avoir depends_on=[]. Une dépendance désigne seulement une étape précédente.
Une lecture fondée sur une découverte doit dépendre de cette découverte. Outils disponibles seulement.
Chaque objectif couvre une seule action de collecte. Choisis un extrait non ambigu si un verbe se répète.
Pour inspecter un fichier explicitement nommé, utilise directement read_file ; list_files prouve seulement sa présence.
Une description de mécanisme n'est pas forcément un littéral du code : utilise nonempty plutôt que contains dans ce cas.
Couvre toutes les actions techniques.
Une instruction finale générique « explique le résultat » ou « explique le résultat avec ses sources »
ne crée pas d'étape : la finale utilise les preuves collectées. Une nouvelle cible à expliquer,
une comparaison ou une vérification garde sa propre collecte de preuves.
Copie les objectifs de la demande, en préservant exactement les chemins.
success contient TOUJOURS tool et kind. Pour une lecture, kind vaut nonempty ou contains.
Seules les clés tool, kind, value, from_step et path sont autorisées dans success.
Les paramètres d'appel, notamment la plage de lecture, seront fournis à l'exécution seulement.
La correction ou le réessai d'une même lecture n'est pas une étape de collecte autonome.
N'utilise aucun chemin fictif ou placeholder. Sans chemin déjà observé, omets seulement success.path.
Un chemin donné par l'utilisateur reste dans l'objectif et l'appel ; sauf lecture multifichiers,
success.path exige from_step et sélectionne seulement un chemin observé dans cette dépendance.
Une lecture explicite cible un seul fichier par étape.
Une lecture ou comparaison nommant plusieurs fichiers utilise plusieurs étapes avec le même objectif exact,
chacune avec success.path choisissant un fichier de cet objectif. Chaque lecture dépend de la précédente.
Tous les fichiers de l'action doivent être lus ; la finale compare les preuves collectées.
success.from_step remplace uniquement path ; conserve tool et kind dans la condition de lecture.
Objectifs et contraintes viennent exclusivement de la demande et de la mémoire utilisateur.
N'invente pas de travail supplémentaire. Ne planifie jamais édition, commandes ou Git write.
Lors d'un replanning, conserve exactement les étapes COMPLETED, remplace seulement le reste.
L'état fourni contient les preuves déjà validées et la raison réelle d'invalidation.
Si une décision utilisateur manque, réponds {"stop":"user_decision", "reason":"information manquante"}.
"""


def _collection_actions(question):
    # ponytail: only generic trailing restitution; qualified explanations remain collection actions.
    final_instruction = re.search(r"\bexplique\s+le\s+résultat(?:\s+avec\s+ses\s+sources)?[.!?]?\s*$", question, re.IGNORECASE)
    return [match for match in ACTION_PATTERN.finditer(question)
            if final_instruction is None or match.start() != final_instruction.start()]


def planning_decision(question, mode="auto"):
    if mode == "off":
        return "M6_DIRECT", "planning désactivé explicitement"
    # ponytail: lexical dependency cues, not a semantic classifier; benchmark false positives/misses.
    text = question.casefold()
    actions = _collection_actions(text)
    dependent = re.search(r"\b(?:puis|ensuite|avant de|à partir|en fonction|associés|associées|correspondants|correspondantes|then|based on|associated|corresponding)\b", text)
    conversation = re.search(r"\b(?:session|conversation|ai-je|avais-je|did i|have i)\b", text)
    inspection = re.search(r"\b(?:fichier|code|implémentation|composant|tests?|file|implementation|component)\b|\.py\b", text)
    if mode == "adaptive":
        if actions and (not conversation or inspection):
            return "M8_ADAPTIVE", "couverture explicite et décisions locales dans la boucle M6"
        return "M6_DIRECT", "aucune collecte projet explicite ; conversation M6 conservée"
    if len(actions) >= 2 and dependent and (not conversation or inspection):
        return "M8_PLANNED", "plusieurs actions avec dépendance explicite"
    return "M6_DIRECT", "aucune dépendance multi-étapes explicite détectée"


def adaptive_requirements(goal):
    """Collection obligations, not a tool sequence or a prediction of future paths.

    Same lexical scope as the M8 router, not general language understanding.
    Semantic deliverables still require grounding and independent review.
    """
    actions = _collection_actions(goal)
    requirements = []
    for index, match in enumerate(actions):
        objective = goal[match.start():actions[index + 1].start() if index + 1 < len(actions) else len(goal)].strip(" ,;.")
        action = match[0].casefold()
        locating = action.startswith(("localis", "identifi", "recherch", "cherch", "trouv", "locate", "find", "search"))
        paths = sorted(requested_paths(objective))
        success = {"tool": "search_code" if locating else "read_file", "kind": "nonempty"}
        if locating:
            target = re.match(r"\s+(?:(?:la définition de|la fonction|la classe|the function|the class)\s+)?(`?[A-Za-z_]\w*`?)", goal[match.end():])
            if target and target[1].strip("`").casefold() not in {"les", "le", "la", "des", "the", "how", "comment", "une", "un"}:
                success.update(kind="contains", value=target[1].strip("`"))
        for path in (paths if len(paths) > 1 and not locating else [None]):
            identifier = str(len(requirements) + 1)
            requirements.append(dict(id=identifier, objective=objective, depends_on=[],
                                     success=dict(success), required_path=path,
                                     status="PENDING", evidence=[]))
    return requirements


def requested_paths(question):
    return {path.replace("\\", "/").lstrip("./")
            for path in re.findall(r"[\w\u0300-\u036f./\\-]+\.(?:py|md|json|toml|txt|yaml|yml|ini|csv)\b", question)}


class ExecutionStopped(RuntimeError):
    pass


def validate_plan(data, goal, tools, max_steps, completed=()):
    if not isinstance(data, dict) or data.get("goal") != goal:
        raise ValueError("Plan goal must exactly match the user request.")
    steps = data.get("steps")
    if not isinstance(steps, list) or not 2 <= len(steps) <= max_steps:
        raise ValueError("Plan must contain 2 to max_plan_steps steps.")
    validated = []
    ids = set()
    covered_actions = set()
    split_reads = {}
    required_actions = {match.start() for match in _collection_actions(goal)}
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"id", "objective", "depends_on", "success"}:
            raise ValueError("Invalid step fields.")
        identifier, objective, dependencies, success = (step[key] for key in ("id", "objective", "depends_on", "success"))
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", identifier) or identifier in ids:
            raise ValueError("Invalid or duplicate step ID.")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError(f"Step {identifier}: Step objective must be an exact excerpt of the user request.")
        actions = list(ACTION_PATTERN.finditer(objective))
        if len(actions) != 1:
            raise ValueError(f"Step {identifier}: Each objective must cover one explicit action.")
        # Normalize only the action word. Regex spans stay in the original request;
        # file names, search literals and observations never lose accents.
        action = actions[0]
        verb = unicodedata.normalize("NFD", action[0]).casefold()
        verb_pattern = "".join(r"[eé]\u0301?" if char == "e" else re.escape(char)
                               for char in verb if not unicodedata.combining(char))
        excerpt_pattern = re.escape(objective[:action.start()]) + verb_pattern + re.escape(objective[action.end():])
        occurrences = list(re.finditer(excerpt_pattern, goal, re.IGNORECASE))
        if not occurrences:
            raise ValueError(f"Step {identifier}: Step objective must be an exact excerpt of the user request.")
        objective_actions = {position for occurrence in occurrences
                             for position in required_actions if occurrence.start() <= position < occurrence.end()}
        if len(objective_actions) != 1:
            raise ValueError(f"Step {identifier}: Each objective must cover one unambiguous collection action.")
        action_start = next(iter(objective_actions))
        action_end = min((position for position in required_actions if position > action_start), default=len(goal))
        objective_paths = requested_paths(objective)
        required_paths = requested_paths(goal[action_start:action_end])
        if not required_paths <= objective_paths:
            detail = ""
            if len(required_paths) > 1:
                excerpt = re.sub(r"[\s,;]*(?:puis|then|et|and)\s*$", "", goal[action_start:action_end], flags=re.IGNORECASE).strip()
                detail = (f" Chaque lecture doit utiliser objective={json.dumps(excerpt, ensure_ascii=False)}. "
                          f"Sélectionne un chemin différent par success.path parmi {sorted(required_paths)} ; "
                          "chaque lecture dépend de la précédente.")
            raise ValueError(f"Step {identifier}: Step objective omits an explicit path from its action." + detail)
        # Keep the user's spelling in the validated state, including verb accents.
        objective = goal[occurrences[0].start():occurrences[0].end()]
        covered_actions.update(objective_actions)
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) or dep not in ids for dep in dependencies):
            raise ValueError(f"Step {identifier}: Dependencies must refer to earlier steps; cycles are forbidden.")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"Step {identifier}: Dependencies must be unique.")
        if not isinstance(success, dict) or set(success) - {"tool", "kind", "value", "from_step", "path"}:
            raise ValueError(f"Step {identifier}: Invalid success condition.")
        if success.get("tool") not in set(tools) & READ_ONLY_TOOLS or success.get("kind") not in {"nonempty", "contains", "observed"}:
            raise ValueError(f"Step {identifier}: Unknown tool or success condition.")
        if success["kind"] == "contains":
            value = success.get("value")
            if not isinstance(value, str) or not value.strip() or value.casefold() not in goal.casefold():
                raise ValueError(f"Step {identifier}: Expected text must come from the user request.")
        origin = success.get("from_step")
        if origin is not None and (not isinstance(origin, str) or origin not in dependencies):
            raise ValueError(f"Step {identifier}: Evidence origin must be an explicit dependency.")
        split_read = success["tool"] == "read_file" and len(objective_paths) > 1
        if "path" in success and (not isinstance(success["path"], str) or (not origin and not split_read)):
            raise ValueError(f"Step {identifier}: Path selection requires an evidence origin.")
        if success["tool"] == "read_file":
            if split_read:
                selected = success.get("path")
                if selected not in objective_paths:
                    raise ValueError(f"Step {identifier}: A read step must have a single explicit path.")
                previous = split_reads.setdefault(action_start, {})
                if selected in previous or (previous and not set(previous.values()) <= set(dependencies)):
                    raise ValueError(f"Step {identifier}: Split reads require distinct paths and dependencies on earlier reads.")
                previous[selected] = identifier
            if not origin and not objective_paths:
                raise ValueError(f"Step {identifier}: A read path must come from a dependency or the current objective.")
        ids.add(identifier)
        validated.append({**step, "objective": objective, "status": "PENDING", "evidence": []})
    if covered_actions != required_actions:
        raise ValueError("Plan omits an explicit action from the user request.")
    for action_start, reads in split_reads.items():
        action_end = min((position for position in required_actions if position > action_start), default=len(goal))
        if set(reads) != requested_paths(goal[action_start:action_end]):
            raise ValueError("Plan omits an explicit path from its action.")
    for old in completed:
        match = next((step for step in validated if step["id"] == old["id"]), None)
        if match is None or any(match[key] != old[key] for key in ("objective", "depends_on", "success")):
            raise ValueError("Replanning cannot change completed steps.")
        match.update(status="COMPLETED", evidence=list(old["evidence"]))
    return validated


def result_data(content):
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return None


def correct_remaining_plan(data, corrections, goal, tools, max_steps, current):
    """Apply explicit step patches atomically, then use the ordinary plan validator."""
    proposal = json.loads(json.dumps(data))
    if not isinstance(proposal, dict) or not isinstance(proposal.get("steps"), list):
        raise ValueError("No proposed plan to correct.")
    if not isinstance(corrections, list) or not corrections:
        raise ValueError("Provide at least one step correction.")
    completed = [step for step in current if step["status"] == "COMPLETED"]
    for correction in corrections:
        if not isinstance(correction, dict) or set(correction) != {"step_id", "changes"}:
            raise ValueError("Use step_id and changes for each correction.")
        identifier, changes = correction["step_id"], correction["changes"]
        matches = [step for step in proposal["steps"] if isinstance(step, dict) and step.get("id") == identifier]
        if len(matches) != 1 or any(step["id"] == identifier for step in completed):
            raise ValueError("Select a unique unfinished step in the current plan.")
        if changes == {"remove": True}:
            proposal["steps"].remove(matches[0])
            continue
        if not isinstance(changes, dict) or not changes or set(changes) - {"objective", "depends_on", "success"}:
            raise ValueError("Only objective, depends_on and success can be corrected.")
        matches[0].update(changes)
    return validate_plan(proposal, goal, tools, max_steps, completed)


def evidence_paths(source):
    if source.get("name") == "search_code":
        return source.get("search_paths", [])
    if source.get("name") == "read_file":
        return [source["file"]]
    data = result_data(source["content"].partition("Result:\n")[2])
    return data.get("files", []) if isinstance(data, dict) else []


def business_text(name, data):
    """Literal tool output text, excluding paths, line numbers and other metadata."""
    if name == "search_code":
        return "\n".join(item.get("text", "") for item in data.get("matches", []) if isinstance(item, dict))
    if name == "list_files":
        return "\n".join(data.get("files", []))
    if name == "run_tests":
        return data.get("stdout", "")
    if name == "read_file" and "content" not in data:
        return "\n".join(item.get("text", "") for item in data.get("content_excerpts", []) if isinstance(item, dict))
    return data.get({"read_file": "content", "get_git_diff": "diff"}.get(name), "")


def observed_read_paths(success, objective, origins):
    paths = {path for source in origins for path in evidence_paths(source)}
    selected = success.get("path")
    explicit = requested_paths(objective)
    if selected or explicit:
        return paths & ({selected} if selected else explicit)
    if len(paths) > 1:
        if re.search(r"\btests?\b", objective, re.IGNORECASE):
            # Test intent needs observed test code; a test-looking filename alone is insufficient.
            return {match["file"] for source in origins
                    for match in (result_data(source["content"].partition("Result:\n")[2]) or {}).get("matches", [])
                    if re.search(r"\bassert\b|\bdef\s+test_", match.get("text", ""))} & paths
        if re.search(r"\b(?:implémentation|implementation|définition|definition|fonction|function|symbole|symbol)\b", objective, re.IGNORECASE):
            definitions = {path for source in origins for path in source.get("definition_paths", [])}
            return definitions & paths
    return paths


def verifies(success, name, content, objective="", origins=()):
    if name != success["tool"]:
        return False
    data = result_data(content)
    if not isinstance(data, dict) or data.get("error") or data.get("ok") is False:
        return False
    payload = {"search_code": "matches", "read_file": "content", "list_files": "files", "get_git_diff": "diff"}.get(name)
    if name == "run_tests":
        value = data.get("stdout", "")
        observed = (data.get("timed_out") is False and type(data.get("returncode")) is int
                    and data["returncode"] in {0, 1, 5}
                    and data.get("passed") is (data["returncode"] == 0))
        present = observed and data["passed"] is True
    else:
        value = data.get(payload)
        if name == "read_file" and value is None and isinstance(data.get("content_excerpts"), list):
            value = "\n".join(str(item.get("text", "")) for item in data["content_excerpts"] if isinstance(item, dict))
        if name == "read_file" and re.search(r"\b(?:implémentation|implementation)\b", objective, re.IGNORECASE) and isinstance(value, str):
            lines = [line for line in value.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if not lines:
                return False
            # ponytail: one-line declarations only; multiline/decorated signatures and inline comments need syntax analysis.
            if lines and all(re.fullmatch(r"\s*(?:(?:async\s+)?def|class)\s+.*:\s*", line) for line in lines):
                return False  # Declarations alone do not verify an implementation read.
        if name == "read_file" and isinstance(value, str):
            definitions = [match for origin in origins for match in origin.get("definition_matches", [])]
            test_target = re.search(r"\btests?\b", objective, re.IGNORECASE)
            targeted = not test_target and (
                re.search(r"\b(?:implémentation|implementation|fonction|function|symbole|symbol|méthode|method)\b", objective, re.IGNORECASE)
                or any(re.search(rf"\b{re.escape(match['name'])}\b", objective) for match in definitions))
            if targeted and any(origin.get("name") == "search_code" for origin in origins):
                if not definitions:
                    return False
                if data.get("truncated"):
                    return False  # Disjoint retained lines cannot establish a complete symbol.
                tree = _parse_fragment(value)
                if tree is None:
                    return False
                path = str(data.get("path", "")).replace("\\", "/").lstrip("./")
                start = data.get("start_line", 1)
                observed = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if any(symbol.get("name") == node.name and symbol.get("start_line") >= start
                               and symbol.get("start_line") <= start + node.lineno - 1
                               and symbol.get("end_line") == start + node.end_lineno - 1
                               for symbol in data.get("complete_symbols", []) if isinstance(symbol, dict)):
                            observed.add((node.name, start + node.lineno - 1))
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        observed.update((target.id, start + node.lineno - 1) for target in targets if isinstance(target, ast.Name))
                if not any(match["file"] == path and (match["name"], match["line"]) in observed for match in definitions):
                    return False
        present = bool(value)
        observed = isinstance(value, list if name in {"search_code", "list_files"} else str)
        observed = observed and not (not value and data.get("truncated"))
    if success["kind"] == "observed":
        return observed
    if success["kind"] == "nonempty":
        return present
    expected = success.get("value")
    return (present and success["kind"] == "contains" and isinstance(expected, str) and bool(expected)
            and expected.casefold() in business_text(name, data).casefold())


def answer_covers_actions(steps, sources, final_sources, answer):
    """Certify literal comparisons and pytest results; citations are checked separately."""
    # ponytail: literal coverage is necessary, not a semantic proof; retain the grounding verifier.
    for step in steps:
        action = ACTION_PATTERN.search(step["objective"])[0].casefold()
        if step["success"]["tool"] == "run_tests":
            for source in sources:
                if source.get("observation_id") not in step["evidence"]:
                    continue
                data = result_data(source["content"].partition("Result:\n")[2]) or {}
                summary = data.get("stdout", "").strip().splitlines()
                counts = {kind: int(count) for count, kind in
                          re.findall(r"\b(\d+)\s+(passed|failed)\b", summary[-1] if summary else "")}
                ids = {index for index, final in enumerate(final_sources, 1)
                       if final.get("observation_id") == source.get("observation_id")}
                for claim, cited in extract_claims(answer):
                    if not ids.intersection(cited):
                        continue
                    prose = claim.replace("`", "")
                    reported = re.findall(r"\b(\d+)\s+(?:tests?\s+)?(passed|failed|réussis?|échoués?)\b", prose, re.IGNORECASE)
                    for count, kind in reported:
                        kind = "passed" if kind.casefold().startswith(("pass", "réussi")) else "failed"
                        if not counts or int(count) != counts.get(kind, 0):
                            return False
                    codes = re.findall(r"\b(?:returncode|code de retour|code de sortie|exit code)\s*(?:[=:]|est|vaut|de)?\s*(-?\d+)\b", prose, re.IGNORECASE)
                    if any(int(code) != data.get("returncode") for code in codes):
                        return False
        if step["success"]["kind"] == "observed" and step["success"]["tool"] == "run_tests" and action.startswith(("identifi", "find")):
            # A process exit code does not identify failing tests. Require the actual
            # pytest node IDs, with a local citation, for this explicit deliverable.
            observations = [source for source in sources if source.get("observation_id") in step["evidence"]]
            for source in observations:
                data = result_data(source["content"].partition("Result:\n")[2]) or {}
                failures = re.findall(r"^FAILED\s+(\S+::\S+)", data.get("stdout", ""), re.MULTILINE)
                ids = {index for index, final in enumerate(final_sources, 1)
                       if final.get("observation_id") == source.get("observation_id")}
                if (not failures and data.get("passed") is True and data.get("returncode") == 0
                        and data.get("timed_out") is False and not data.get("truncated")):
                    if not any(ids.intersection(cited) for _, cited in extract_claims(answer)):
                        return False
                    continue  # A successful pytest run has no failing node IDs to enumerate.
                if not failures or not all(any(node in claim and ids.intersection(cited)
                                               for claim, cited in extract_claims(answer)) for node in failures):
                    return False
        if action.startswith(("expli", "analy")):
            return False  # These untyped semantic deliverables have no deterministic certificate.
        if not action.startswith("compar"):
            continue
        relevant = {previous["id"] for previous in steps if previous["objective"] == step["objective"]}
        for previous in reversed(steps):
            if previous["id"] in relevant:
                relevant.update(previous["depends_on"])
        evidence = {identifier for previous in steps if previous["id"] in relevant for identifier in previous["evidence"]}
        reads = {}
        for source in sources:
            if source.get("observation_id") not in evidence or source.get("name") != "read_file":
                continue
            data = result_data(source["content"].partition("Result:\n")[2]) or {}
            try:
                tree = ast.parse(data.get("content", ""))
            except (SyntaxError, TypeError):
                return False
            values = {}
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        values.pop(target.id, None)
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if not isinstance(value, (str, int, float, bool, type(None))):
                    continue
                values.update((target.id, value) for target in targets if isinstance(target, ast.Name))
            reads[source["file"]] = (source["observation_id"], values)
        if len(reads) < 2:
            return False
        common = set.intersection(*(set(values) for _, values in reads.values()))
        observed_names = set.union(*(set(values) for _, values in reads.values()))
        named = {name for name in observed_names if re.search(rf"\b{re.escape(name)}\b", " ".join(s["objective"] for s in steps))}
        if not named <= common:
            return False
        targets = named or common
        if not targets or (not named and len(targets) != 1):
            return False  # An ambiguous/nonliteral comparison is not certified by keywords.
        claims = extract_claims(answer)
        for identifier, values in reads.values():
            ids = {index for index, source in enumerate(final_sources, 1) if source.get("observation_id") == identifier}
            for name in targets:
                literal = re.escape(str(values[name]))
                # Recognize elementary value assertions, not unrelated line/count numbers.
                assertion = r"(?:\b(?:vaut|valent|equals?)\s+|(?<![=!<>])=(?!=)\s*)[`'\"]*"
                continuation = r"(?:\b(?:contre|et|versus|vs\.?|and)|,)\s+[`'\"]*"
                if not any(ids.intersection(source_ids)
                           and re.search(assertion, claim)
                           and re.search(rf"(?:{assertion}|{continuation}){literal}(?![\w.])", claim)
                           for claim, source_ids in claims):
                    return False
    return True


def observed_source_code(question, sources):
    """Return complete, unambiguous current source targets, never read from disk."""
    paths = requested_paths(question)
    target_text = question
    for path in paths:
        target_text = target_text.replace(path, "")
    whole_file = bool(re.search(r"\b(?:contenu|content)[^.!?\n]{0,40}\b(?:fichier|file)\b"
                                r"|\bfichier (?:complet|entier)\b|\b(?:full|whole|entire) file\b", question, re.IGNORECASE))
    file_code = any(re.search(rf"\b(?:code|contenu|content)\s+(?:exact\s+)?(?:de|du fichier|of)\s+`?{re.escape(path)}", question, re.IGNORECASE)
                    for path in paths)
    generic_code = bool(re.search(r"\b(?:montre|show|affiche|display)\s+(?:le\s+|the\s+)?code\s*[.!?]", question, re.IGNORECASE))
    targets = []
    for index, source in enumerate(sources, 1):
        if source.get("type") != "tool" or source.get("name") != "read_file" or (paths and source["file"] not in paths):
            continue
        data = result_data(source["content"].partition("Result:\n")[2]) or {}
        code = data.get("content")
        if data.get("truncated") or not isinstance(code, str) or data.get("path") != source["file"]:
            continue
        start, end = data.get("start_line"), data.get("end_line")
        lines = code.replace("\r\n", "\n").split("\n")
        if type(start) is not int or type(end) is not int or len(lines) != end - start + 1:
            continue
        symbols = [symbol for symbol in data.get("complete_symbols", []) if isinstance(symbol, dict)
                   and isinstance(symbol.get("name"), str)
                   and re.search(rf"\b{re.escape(symbol['name'])}\b", target_text)]
        if whole_file or file_code or (paths and generic_code and not symbols):
            if start == 1 and type(data.get("total_lines")) is int and end == data["total_lines"]:
                targets.append((index, code))
        else:
            for symbol in symbols:
                first, last = symbol.get("start_line"), symbol.get("end_line")
                if type(first) is int and type(last) is int and start <= first <= last <= end:
                    targets.append((index, "\n".join(lines[first - start:last - start + 1])))
    # Every explicitly named file must have a complete target. Ambiguity stays a refusal.
    if paths and not paths <= {sources[index - 1]["file"] for index, _ in targets}:
        return []
    return targets


def answer_code_observed(answer, sources, question=""):
    """Exact fenced excerpts must occur in a locally cited current read.

    With a source request, require the complete observed target as well.
    Does not certify prose or semantic correctness.
    Rejects rewritten code rather than treating a plausible template as evidence.
    """
    blocks = []
    for claim, ids in extract_claims(answer):
        for span in CODE_PATTERN.finditer(claim):
            if span["inline"]:
                continue
            code = span[0]
            if span["ticks"] or span["tildes"]:
                if not (span["ticks_end"] or span["tildes_end"]):
                    return False
                code = code.split("\n", 1)[1].rsplit("\n", 1)[0]
            code = code.replace("\r\n", "\n")
            blocks.append((code, ids))
            code = code.strip("\n")
            if not code or not any(
                source.get("name") == "read_file" and source.get("type") == "tool"
                and code in str((result_data(source["content"].partition("Result:\n")[2]) or {}).get("content", "")).replace("\r\n", "\n")
                for index, source in enumerate(sources, 1) if index in ids
            ):
                return False
    if requests_source_code(question):
        targets = observed_source_code(question, sources)
        return bool(targets) and all(any(target.replace("\r\n", "\n") in code and index in ids
                                        for code, ids in blocks) for index, target in targets)
    return True


class PlanningExecution:
    def __init__(self, goal, mode, reason, *, max_tool_calls, max_total_tokens, max_seconds,
                 max_iterations=5, max_replans=2, debug=False):
        self.started = perf_counter()
        self.debug = debug
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.max_seconds = max_seconds
        self.used_tokens = 0
        self.tool_calls = 0
        self.iterations = 0
        self.max_iterations = max_iterations
        self.max_replans = max_replans
        self.state = dict(goal=goal, mode=mode, reason=reason, plan=[], current_step=None,
                          completed=[], remaining=[], observations=[], errors=[], replans=0,
                          stop_reason=None, budgets={}, events=[])
        self.refresh()
        self.event("routing", mode=mode, reason=reason, request=goal)

    def event(self, event, **details):
        record = json.loads(json.dumps(dict(event=event, seconds=round(perf_counter() - self.started, 4), **details), ensure_ascii=False))
        self.state["events"].append(record)
        if self.debug:
            print("M8 " + json.dumps(record, ensure_ascii=False), flush=True)

    def refresh(self):
        self.state["completed"] = [s["id"] for s in self.state["plan"] if s["status"] == "COMPLETED"]
        self.state["remaining"] = [s["id"] for s in self.state["plan"] if s["status"] != "COMPLETED"]
        self.state["budgets"] = dict(tool_calls=max(0, self.max_tool_calls - self.tool_calls),
                                      tokens=max(0, self.max_total_tokens - self.used_tokens),
                                      iterations=max(0, self.max_iterations - self.iterations),
                                      replans=max(0, self.max_replans - self.state["replans"]),
                                      seconds=max(0, self.max_seconds - (perf_counter() - self.started)))

    @property
    def current(self):
        return next((s for s in self.state["plan"] if s["status"] != "COMPLETED"), None)

    def check(self, reserve=0):
        self.refresh()
        if self.state["budgets"]["seconds"] <= 0 or self.used_tokens > self.max_total_tokens or reserve > self.state["budgets"]["tokens"]:
            self.stop("global_budget")
            raise ExecutionStopped("global_budget")

    def stop(self, reason, detail=None):
        if self.state["stop_reason"]:
            return
        self.state["stop_reason"] = reason
        if self.current and self.current["status"] == "RUNNING":
            self.current["status"] = "FAILED"
            self.event("step_failed", step_id=self.current["id"], reason=reason)
        self.state["current_step"] = None
        if detail:
            self.state["errors"].append(detail)
        self.refresh()
        self.event("stop", reason=reason, completed=self.state["completed"], remaining=self.state["remaining"], detail=detail)

    def summary(self):
        self.refresh()
        completed = [f"{s['id']} ({s['objective']})" for s in self.state['plan'] if s['status'] == 'COMPLETED']
        remaining = [f"{s['id']} ({s['objective']})" for s in self.state['plan'] if s['status'] != 'COMPLETED']
        if "human" in self.state and not self.state["plan"]:
            remaining = ["objectif initial sans plan valide : " + self.state["goal"]]
        evidence = (f"\nObservations conservées : {', '.join(self.state['observations']) or 'aucune'}. "
                    f"Appels MCP exécutés : {self.tool_calls}." if "human" in self.state else "")
        return (f"Exécution arrêtée : {self.state['stop_reason']}.\n"
                f"Collecte validée : {', '.join(completed) or 'aucune'}.\n"
                f"Étapes non terminées : {', '.join(remaining) or 'aucune'}.\n"
                "Objectif final non validé.\n"
                f"Informations/diagnostic : {self.state['errors'][-1] if self.state['errors'] else 'preuves ou budget insuffisants' }." + evidence)

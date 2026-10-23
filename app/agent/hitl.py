"""Bounded terminal intervention; guidance never becomes tool evidence."""

import json
import re
import sys

MAX_INTERVENTIONS = 3
MAX_GUIDANCE = 2000


def display_text(value, limit=1200):
    text = str(value)
    text = re.sub(r"(?i)((?:api[_-]?key|token|password|secret|authorization)\s*[\"']?\s*[:=]\s*)[^\s,}\n]+", r"\1[REDACTED]", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(char for char in text if char.isprintable() or char in "\n\t")[:limit]


def terminal_intervention(snapshot):
    if not sys.stdin or not sys.stdin.isatty():
        return "unavailable", ""
    print("\nCODEAGENT — INTERVENTION HUMAINE REQUISE")
    print("Objectif :", display_text(snapshot["goal"]))
    print("Blocage :", display_text(snapshot["reason"]))
    print("Diagnostic :", display_text(snapshot["error"]))
    print("Étape courante :", display_text(snapshot["current_step"]))
    for label, identifiers in (("Étapes terminées", snapshot["completed"]), ("Étapes restantes", snapshot["remaining"])):
        print(label + " :", display_text(json.dumps(
            [dict(id=step["id"], objective=step["objective"]) for step in snapshot["plan"] if step["id"] in identifiers],
            ensure_ascii=False)))
    print("Dernier appel :", display_text(json.dumps(snapshot["last_tool"], ensure_ascii=False)))
    print("Preuves :", display_text(json.dumps(snapshot["evidence"], ensure_ascii=False)))
    print("Budget restant :", snapshot["budgets"])
    print("Échecs par tool :", snapshot.get("tool_failures", {}),
          "; quota par tool :", snapshot.get("max_tool_failures", "indisponible"))
    if snapshot.get("final"):
        print("FINALISATION BLOQUÉE :", snapshot.get("grounding"))
        print("Candidate conservée :", display_text(snapshot.get("candidate")))
        print("Réessai : mêmes sources ; timeout/exception du grounding = vérification seule.")
    print("Action ciblée :", snapshot["suggestion"])
    while True:
        print("1. Examiner plan et sources (extraits bornés)")
        print("2. Fournir une indication ; proposition et appels revalidés")
        print("3. Réessayer dans le budget restant")
        print("4. Arrêter ; retourner le bilan partiel non validé")
        if not snapshot.get("final"):
            print("5. Sélectionner un chemin observé")
            print("6. Fournir un chemin du projet (instruction à vérifier par MCP)")
            print("7. Exécuter un tool autorisé avec arguments JSON")
            print("8. Corriger les étapes restantes (patches JSON, validation du plan complet)")
        try:
            choice = input("Choix : ").strip()
            if choice == "1":
                for error in snapshot.get("errors", [])[-3:]:
                    print("Diagnostic conservé :", display_text(error))
                for step in snapshot["plan"]:
                    print(display_text(json.dumps(step, ensure_ascii=False)))
                for source in snapshot["sources"]:
                    print(display_text(json.dumps(source, ensure_ascii=False)))
                if snapshot.get("raw_plan"):
                    print("Plan brut :", display_text(snapshot["raw_plan"], 3000))
            elif choice == "2":
                return "guidance", input("Indication ciblée (instruction, pas preuve) : ").strip()
            elif choice == "3":
                return "retry", ""
            elif choice == "4":
                return "stop", ""
            elif choice in {"5", "6", "7", "8"} and not snapshot.get("final"):
                step_id = input("Étape restante (ID) : ").strip()
                if choice == "5":
                    paths = snapshot.get("observed_paths", [])
                    for index, path in enumerate(paths, 1):
                        print(f"{index}. {display_text(path)}")
                    index = int(input("Numéro du chemin : "))
                    if not 1 <= index <= len(paths):
                        raise ValueError("Numéro hors liste.")
                    return "path", dict(step_id=step_id, path=paths[index - 1])
                if choice == "6":
                    return "path", dict(step_id=step_id, path=input("Chemin du projet : ").strip())
                if choice == "7":
                    for tool in snapshot.get("tools", []):
                        print(display_text(json.dumps(tool["function"], ensure_ascii=False), 3000))
                    name = input("Nom du tool : ").strip()
                    arguments = json.loads(input("Arguments JSON : "))
                    return "tool", dict(step_id=step_id, name=name, arguments=arguments)
                changes = json.loads(input('Changements JSON (objective, depends_on, success ou {"remove":true}) : '))
                corrections = [dict(step_id=step_id, changes=changes)]
                while input("Autre étape à corriger ? [o/N] : ").strip().casefold() == "o":
                    identifier = input("Étape restante (ID) : ").strip()
                    corrections.append(dict(step_id=identifier, changes=json.loads(input("Changements JSON : "))))
                action = input('Action MCP JSON {"step_id":"…","name":"…","arguments":{…}} (vide = modèle) : ').strip()
                payload = dict(corrections=corrections)
                if action:
                    payload["tool_action"] = json.loads(action)
                return "correct_step", payload
            else:
                print("Choix invalide : sélectionner une action affichée.")
        except (ValueError, TypeError) as error:
            print("Entrée invalide :", display_text(error))
        except (EOFError, KeyboardInterrupt):
            print("\nIntervention annulée ; arrêt partiel.")
            return "stop", ""

# CodeAgent — Project Memory

[English](memory.md) | [Français](memory.fr.md) | [简体中文](memory.zh-CN.md)

[README du projet](../README.fr.md)


> Mémoire synthétique du projet : fonctionnalités, décisions, résultats et limites de M1 à M8.

## État actuel

**M1–M7 terminés. M8 PASS dans son périmètre initial de planification contrôlée ; deux scénarios autonomes réels validés avec `OLLAMA_MAX_LOADED_MODELS=1`. Qualification HITL actuelle partielle. M9 non commencé.**

CodeAgent analyse les projets Python, indexe leur code, produit des réponses sourcées avec vérification des citations et du grounding, maintient l’index et utilise des tools read-only via MCP.

M7 apporte la mémoire conversationnelle persistante et le budget de contexte partagé. M8 ajoute une planification conditionnelle et une exécution contrôlée avec replanning.

## M1 — Lecture et extraction

Le premier milestone a posé la base du projet :

- parcours récursif des fichiers Python ;
- filtrage des dossiers inutiles ;
- lecture du contenu ;
- gestion des chemins relatifs ;
- premiers tests unitaires.

Décision importante : garder le pipeline simple et indépendant du LLM afin que l’analyse du dépôt reste déterministe.

## M2 — Recherche sémantique

Le code a ensuite été découpé avec l’AST Python en chunks représentant principalement les fonctions, classes, variables et imports.

Chaque chunk conserve sa provenance :

- fichier ;
- type ;
- nom ;
- lignes de début et de fin.

Les chunks sont encodés avec `all-MiniLM-L6-v2` puis indexés dans FAISS.

La recherche a ensuite été améliorée avec un reranking lexical appliqué aux candidats retournés par FAISS.

Pipeline obtenu :

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

## M3 — LLM local et RAG

M3 a ajouté la génération de réponses sur le code.

Avant l’intégration, plusieurs LLM locaux ont été comparés avec un benchmark commun afin de sélectionner un modèle adapté au projet.

Modèle retenu :

- `Qwen2.5-Coder 14B`
- exécuté localement avec Ollama.

Le pipeline RAG final est :

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

Le contexte transmis au LLM conserve les métadonnées des chunks, ce qui permet de produire des réponses indiquant les fichiers et lignes utilisés.

Exemple :

```text
app/parser/file_loader.py:6-9
```

## M4 — Mémoire projet

M4 a ajouté une mémoire persistante et incrémentale au projet afin d’éviter de reconstruire entièrement l’index à chaque exécution.

Fonctionnalités ajoutées :

- sauvegarde persistante de l’index FAISS ;
- sauvegarde des chunks associés ;
- rechargement automatique de l’index existant ;
- historique des questions et réponses ;
- création d’un manifeste du projet ;
- détection des fichiers ajoutés, modifiés et supprimés ;
- mise à jour incrémentale des embeddings et de l’index ;
- conservation des chemins relatifs pour identifier les fichiers du projet.

Le manifeste associe chaque fichier à un hash SHA-256 de son contenu. Deux états successifs du projet peuvent ainsi être comparés pour déterminer précisément les modifications.

Pipeline de mise à jour :

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

Lorsqu’aucune modification n’est détectée, CodeAgent réutilise directement l’index déjà sauvegardé.

Les analyses sont également enregistrées dans un historique JSONL contenant la date, la question et la réponse générée.

## M5 — RAG Quality & Grounding

M5 a renforcé et évalué le pipeline RAG avec :

- un benchmark de retrieval avec Hit@K, Recall@K, Precision@K et MRR ;
- une recherche hybride dense + lexicale ;
- un reranking et une sélection de contexte améliorés ;
- un prompt renforcé contre les hallucinations ;
- des citations `[Sx]` vérifiées automatiquement ;
- une vérification claim-by-claim du support des réponses par les sources.

Sur le holdout final de 16 questions, M5 améliore `Hit@5` et `Recall@5` de `0.6667` à `0.8333`, ainsi que le comportement réponse/abstention de `0.6250` à `0.7500`. Le `MRR@5` passe de `0.5069` à `0.3917` et la vérification du grounding ajoute en moyenne `2.54 s` au pipeline.

## M6 — MCP & Tool Use

M6 ajoute l'utilisation contrôlée de tools via MCP :

- tools read-only `read_file`, `search_code`, `list_files`, `get_git_diff` et `run_tests` ;
- tool calling avec `Devstral 24B` ;
- boucle `LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM` ;
- timeout, limite d'étapes, retries limités, détection des appels dupliqués et erreurs structurées ;
- réutilisation du grounding M5 sur les réponses fondées sur des observations de tools ;
- suivi par question des tokens Devstral / grounding, des étapes, appels de tools, temps par étape et temps individuel des tools ;
- génération finale dédiée à partir des observations sourcées, avec validation des citations puis grounding ;
- réduction du contexte tool avec résultats de recherche bornés, lecture ciblée par plage de lignes et consolidation des sources redondantes.

Le benchmark end-to-end final de 20 tâches atteint `80 %` de succès, avec `100 %` de succès de routing, `100 %` de précision de sélection des tools, `100 %` de précision no-tool et `100 %` de précision des arguments. Les 20 tâches produisent 13 réponses groundées acceptées, 4 rejets `unsupported` et 0 `citation_error`. La latence moyenne reste élevée (`131.05 s`) et dépend principalement des LLM locaux, les tools eux-mêmes étant désormais rapides.

Sur le benchmark de grounding figé de 40 cas équilibrés, `Qwen2.5-Coder 14B` atteint `97.5 %` d'accuracy, `100 %` de recall sur les cas unsupported, 0 false accept et 1 false reject. `Devstral 24B` atteint `90 %` avec 0 false accept et 4 false rejects ; `Qwen2.5-Coder 14B` reste donc le grounder par défaut. Sur le benchmark de gating de 20 tâches, Devstral atteint `100 %` de précision. Laya et Verdict atteignent `75 %`, ModernBERT-NLI `65 %` et GLiNER2.5 `45 %`. Jev n'a pas pu être évalué car l'API externe est restée inaccessible malgré un appel conforme à la documentation. La cascade Verdict -> Devstral réduit la latence de `42.9 %` mais baisse la précision de `100 %` à `93.3 %` ; elle n'est pas retenue.

L'optimisation finale du contexte réduit le smoke test multi-tool d'environ `10.5k` à `6.6k` tokens sans perte de grounding. `search_code` a également été optimisé en évitant la traversée des dossiers ignorés avant filtrage : son temps observé passe d'environ `17.1 s` à `0.01-0.12 s` dans les smoke tests.

## M7 — Memory & Context Management

**M7 terminé** pour la mémoire et le contexte ; la fiabilité des finales autonomes reste distincte.

- `ConversationMemory` conserve le contexte récent et compresse automatiquement les anciens tours en extraits ordonnés et prioritaires, sans LLM de résumé.
- L’archive append-only `history.jsonl` conserve les échanges et observations originaux ; les checkpoints atomiques `sessions/<session_id>.json` permettent la reprise et la reconstruction depuis le JSONL en cas de corruption.
- Les checkpoints de version future ou en avance sur l’archive sont refusés sans écrasement. La mémoire conversationnelle reste séparée de l’index projet.
- `ContextBudget` partage le contexte entre mémoire, RAG, tools et LLM, avec réserve de génération et estimation conservatrice en octets UTF-8 ; un élément indispensable trop grand provoque un refus explicite.
- Le chemin résolu du projet et `--session` isolent les conversations, y compris entre projets homonymes.
- Les sources actuelles ont priorité sur l’historique ; sélection et compression conservent provenance, chemins, lignes et valeurs. Finale et grounding utilisent les mêmes sources retenues.
- Un échange ancien n’est pas une preuve du code actuel : une source modifiée doit être relue. Les erreurs ne prouvent que leur diagnostic.

Le [benchmark déterministe mémoire](../benchmarks/memory/results/m7.json) valide **9/9 rappels et 6/6 contrôles** : contexte borné, rappel, fidélité extractive, sources/grounding, récupération et absence de modèle de résumé.
Les essais réels historiques conservent **7/7 lectures `read_file` récupérées, mais 0/7 finales autonomes complètes**, notamment pour citations ou récupération omises ; ces chiffres ne sont pas un rejeu après les corrections M8.
La récupération d’anciens échanges reste lexicale et linéaire ; la recherche sémantique reste optionnelle. Un résumé borné peut évincer des extraits, dont les originaux restent archivés.
Une session suppose un seul processus écrivain, sans purge automatique de l’archive. La mémoire ne garantit ni un rappel fidèle du modèle ni une finale correctement sourcée.

<a id="m8--planning--controlled-autonomous-execution"></a>
## M8 — Planning & Controlled Autonomous Execution

**PASS dans le périmètre initial du milestone.** M8 apporte plans courts conditionnels, état minimal, étapes MCP réelles, reprise bornée du travail restant, conservation des acquis, arrêts explicites et comparaison avec M6 direct. Les objectifs initiaux figurent dans `README.md` au Git `0bf0437`. HITL optionnel possède son périmètre distinct ci-dessous ; aucune fiabilité autonome générale n'est revendiquée.

<a id="m8-contract"></a>
### Architecture et contrats

- `app/agent/planning.py` porte routage lexical, validation du plan, état et certificats de preuves. `ToolLoop` réutilise l'unique boucle MCP M6, mémoire M7 et citations/grounding M5/M6. Les demandes simples restent directes ; `--planning auto` est le défaut, `off` désactive la planification, `adaptive` reste expérimental.
- Le planner retourne le JSON `{goal, steps}` ou `{"stop":"user_decision","reason":"..."}`. Un plan contient 2–6 étapes avec exactement `id`, `objective`, `depends_on`, `success` ; IDs uniques, dépendances seulement antérieures, `[]` permis pour les étapes indépendantes.
- Les objectifs gardent extraits non ambigus et chemins de la demande, avec normalisation bornée de casse/accents du verbe. Toutes les actions de collecte sont couvertes. Une lecture coordonnée multifichiers garde le même objectif complet, une étape par fichier et des sélections `success.path` explicites.
- `success` autorise seulement `tool`, `kind`, `value`, `from_step`, `path` ; outils : `read_file`, `search_code`, `list_files`, `get_git_diff`, `run_tests` ; conditions : `contains`, `nonempty`, `observed`. Un chemin découvert provient d'une dépendance terminée désignée par `from_step` ; les chemins inventés sont refusés.
- `observed` constate un résultat exécuté, y compris recherche/diff vide ou tests échoués, sans certifier l'objectif final. Erreurs et timeouts ne terminent pas une étape. Une implémentation demandée exige son corps complet, pas sa signature ou un template.
- Le replanning remplace seulement le travail restant et garde étapes terminées/preuves. Une réparation de validation par génération partage le quota de replanning. Un groupe invalide reçoit un seul rappel borné et n'est jamais exécuté avant validation. Retries, déduplication et budgets restent partagés.
- Les sources gardent IDs d'observation originaux, fichiers, lignes et arguments. La sélection peut renuméroter `[Sx]` ; `source_mapping` relie citations finales et observations. Les preuves actuelles précèdent l'historique ; les instructions humaines ne deviennent jamais des preuves. Finale et grounding utilisent les mêmes sources retenues.
- `goal_achieved` exige collecte terminée, citations valides, grounding, format demandé et couverture, sans abstention injustifiée. Nombres pytest explicites/codes de retour sont comparés aux résultats cités. Collecte, `COMPLETED` ou `SUPPORTED` seuls ne prouvent pas la réussite finale.

<a id="m8-hitl"></a>
### HITL et reprise

`--hitl off` est le défaut. `on-failure` suspend après un rejet observable du plan, d'une étape ou de la finale. La même tâche conserve connexion MCP, sources, étapes terminées, erreurs, déduplication et compteurs ; un seul tour M7 est archivé à la fin.

| Menu | Action |
|---|---|
| 1–4 | Inspecter l'état borné/masqué ; fournir une indication ; réessayer dans les quotas ; arrêter avec bilan partiel non validé. |
| 5–7 | Sélectionner un chemin observé ; fournir un chemin projet en gardant les autres arguments ; exécuter un tool read-only existant avec arguments JSON validés. |
| 8 | Patcher les étapes restantes identifiables, supprimer éventuellement l'une avec `{"remove":true}`, puis valider atomiquement le plan entier avant action. |

Les étapes terminées ne sont pas réécrites. Permissions, JSON Schema, limites du projet, dépendances, provenance, doublons, citations et grounding restent contrôlés. Les actions directes consomment les quotas existants d'appels/itérations. Limites : trois interventions par tâche, 2 000 caractères par indication, aucune remise à zéro. L'attente humaine compte dans le délai ; la saisie synchrone vérifie l'échéance à son retour. Sans terminal, retourner `intervention_required`.

Une finale refusée autorise une génération supplémentaire avec les mêmes sources et aucune nouvelle collecte. Une exception de grounding autorise un seul réessai de vérification de la même candidate/sources, distinct de `UNSUPPORTED`. `resumed` prouve seulement l'autorisation : vérifier appels ultérieurs et résultat final. Restaurer la conversation après redémarrage ne reprend pas une exécution interrompue.

<a id="m8-corrections"></a>
### Causes et corrections conservées

| Cause démontrée | Correction conservée |
|---|---|
| Casse verbale/accents NFC/NFD, fichiers comptés comme actions, dépendances inutiles | Borner la reconnaissance sans changer positions/chemins ; permettre étapes indépendantes ; diagnostics par étape et extraits multifichiers exacts pour réparation. |
| Mauvais fichier de tests, statut processus incohérent, collecte assimilée à complétion | Valider chemins pytest avant exécution/certification, exiger statut cohérent, conserver couverture/refus des faux succès. |
| Recherche corrigée supprimée ; signature/template substitué au corps | Garder observations de récupération, séparer diagnostic/contenu ; utiliser `total_lines`, complétude AST et livraison de code fidèle cité localement. |
| Prompt final demandant un reste fictif ou un récit des tools | Transmettre plan/preuves validés, demander des faits cités localement, refuser l'abstention injustifiée avant grounding. |
| H : `/api/chat` vide malgré un appel généré | Le rejeu brut révèle un `read_file` hors étape perdu au parsing Ollama 0.40.8. Placer l'étape courante après les observations ; conserver filtrage/validation. |
| Métriques des sorties vides perdues ; timeout de chargement Qwen avec Devstral résident | Enregistrer statistiques avant refus du contenu, conserver `done_reason` ; `OLLAMA_MAX_LOADED_MODELS=1` natif sur la machine mesurée, sans gestionnaire applicatif ni hausse du timeout. |

<a id="m8-validation"></a>
### Validation et périmètre

**Audit final : 799 tests réussis**, dont **509 tests concernés** ; mémoire **6/6**, planning **6/6** sur **16 scénarios / 32 exécutions** ; syntaxe, imports, options CLI et `git diff --check` réussis. Aucun changement fonctionnel applicatif/script dans cette consolidation : les preuves réelles historiques restent applicables au runtime ; les tests sont réorganisés. Les tests déterministes utilisent LLM/grounding scriptés, MCP mémoire/stdio réel et véritable pytest. Ils vérifient les invariants, pas l'autonomie Ollama.

La campagne réelle finale du **23 octobre 2026** utilise Ollama **0.40.8**, `devstral:24b`, `qwen2.5-coder:14b`, température **0**, seed **42**, contexte **16384**, génération limitée à **512**, timeout HTTP **300 s**, `keep_alive=30m`, `OLLAMA_MAX_LOADED_MODELS=1`. Budgets partagés : **12 itérations, 6 étapes, 2 replannings, 16 MCP, 32768 tokens, 900 s**, timeout outil **60 s**. MCP est un vrai sous-processus stdio ; aucune réponse ni verdict de grounding scripté.

| Scénario réel | Résultat vérifié |
|---|---|
| Principal, `main-final`, HITL `on-failure` | **PASS AUTONOMOUS**, **3 MCP**, **248.454 s**, zéro intervention/doublon/réessai final ; une réparation autonome du plan. Deux fichiers lus intégralement, véritable pytest, couverture/citations/format/grounding valides ; revue indépendante **13/13**. |
| Historique H inchangé, `H-final`, HITL `off` | **PASS AUTONOMOUS**, **4 MCP**, **274.461 s**, zéro intervention/replanning/doublon. Implémentation et tests lus ; `marker` retourne 42 et `assert marker() == 42` correctement cité ; revue indépendante **13/13**. |

Demande principale exacte :

> Lis app/agent/planning.py et tests/test_planning.py, puis exécute les tests de tests/test_planning.py. Fournis un bilan factuel en citant les fichiers consultés et le résultat réel des tests.

À cette révision, lectures des lignes **1–596** et **1–505**. MCP : `131 passed, 1 warning in 0.74s`, code **0**, `passed=true`, `timed_out=false`, stderr vide. Pytest indépendant : `131 passed, 1 warning in 0.69s`. L'avertissement existant `cache_dir` vient de la désactivation du plugin cache. Ces nombres sont historiques, avant réorganisation des tests. Les citations livrées S1/S2 identifient les fichiers, S3 le résultat réel de pytest.

**HITL : qualification PARTIAL sur la révision actuelle.** Les tests déterministes interruption/reprise passent. Les essais historiques du **18 octobre** avec Ollama + MCP mémoire incluent **PASS ASSISTED F** (3 MCP, 2 interventions, **258.028 s**) et **PASS ASSISTED E2** (3 MCP, 3 actions préautorisées, **552.317 s**). Les décisions de chat enregistrées fournissent actions/indications après snapshots, jamais des réponses LLM/plans/finales de remplacement. Ces succès bornés à d'anciennes révisions sont partiellement reproductibles : des F ultérieurs et la finale H assistée échouent. La collecte/refinalisation assistée du 23 octobre conserve les preuves mais échoue en finale. Les derniers succès avec zéro intervention ne valident **pas** un nouveau cycle interruption/reprise complet sur le code actuel.

**Décisions expérimentales :** pilotes adaptatif/global à **1/4** strict chacun (14541 / 28720 tokens) : **KEEP GLOBAL**. Finalisation guidée à **3/10** livrées dans les deux variantes pour **10.44%** de tokens supplémentaires : **KEEP CURRENT**, prototypes de guidage/révision retirés. JSON Schema natif : meilleur JSON brut, aucun gain strict (**1/4** chacun ; 14836 / 16012 tokens, **+7.93%**) : **KEEP CURRENT**, prototype/15 tests retirés, aucun `--planning-format` en production. Le schéma ne certifiait pas provenance/sémantique ; `format` avec `tools` natifs n'était pas utilisé. Les holdouts non exécutés ne sont ni succès ni échecs. Détails : [revue autonomie](../benchmarks/planning/results/autonomy-20261017-review.json), [finalisation](../benchmarks/planning/results/finalization-20261017-review.json), [structured](../benchmarks/planning/results/structured-20261017-review.json), [protocole structured](../benchmarks/planning/structured-protocol.json).

<a id="m8-reproduction"></a>
### Reproduction et preuves

Depuis la racine, utiliser dépendances/modèles installés et nouvelles sorties hors dépôt :

```powershell
$py = '.venv/Scripts/python.exe'
$out = Join-Path $env:TEMP ('codeagent-m8-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
& $py -m pytest -q
& $py -m scripts.benchmark_memory --output (Join-Path $out 'memory.json')
& $py -m scripts.benchmark_planning --output (Join-Path $out 'planning.json')
& $py -m app.main --project . --planning auto --hitl on-failure --debug --session ('m8-' + [guid]::NewGuid())
```

Choisir **2**, soumettre la demande exacte ci-dessus. Pour H et les scénarios optionnels d'échec assisté, voir les [instructions planning](../benchmarks/llm/README.fr.md#memory-and-planning-protocols). Le benchmark réel exige `--real` ; `app.main` ci-dessus utilise Ollama directement, sans ce drapeau. Les replays scriptés restent distincts. Sur la machine mesurée, démarrer Ollama installé avec `$env:OLLAMA_MAX_LOADED_MODELS='1'` limité au processus. Préserver services existants, restaurer état initial puis décharger uniquement modèles/processus créés pour l'essai. Le serveur temporaire historique est arrêté, zéro processus Ollama restant ; aucun réglage persistant modifié.

Preuves finales locales hors dépôt :

```text
E:/AI/Codex/visualizations/2026/10/23/01a16a21-572d-7520-b967-5111912d7a82/m8-final
```

`main-final/`, `H-final/` : protocoles, requêtes/réponses HTTP, traces MCP, sources, durées, résultats, revues indépendantes. `tested-source/`, empreintes, `session-changes.patch` : révision validée. `final-suite.txt`, `final-memory.json`, `final-planning.json`, `final-check-exits.json` : contrôles historiques. `diagnostic.json`, `H-raw-exact.json`, `H-corrected/` : échecs parsing/chargement. Les scripts archivés `run_campaign.py`, `review_run.py`, `review_history.py` reproduisent/revoient avec des dossiers neufs. Les artefacts locaux ne font pas partie d'un clone.

Les preuves antérieures restent sous `data/memory/m8-autonomy-20261017`, `m8-finalization-20261017`, `m8-structured-20261017` (dont `prototype.zip`, `prototype-files.json`, `manifest.json`), `m8-hitl-20261017`, `m8-hitl-executable-20261018/final-delivery`, `m8-hitl-validation-20261018-853fc1c4/E-02` ; les deux derniers gardent succès assistés, décisions et revues. Les audits échoués du 23 octobre restent dans les dossiers de visualisations `01a169cb-76e8-7e91-aadc-6510601947a4/m8-hitl-audit` et `01a16a07-8d01-7f31-9519-9b7b30be1a38/m8-final`. Aucun échec historique n'est réécrit comme succès.

<a id="m8-limitations"></a>
### Limites restantes

Les deux succès valident ces scénarios/configuration, sans autonomie générale ni généralisation aux tâches inédites. Routage lexical, certificats de preuves et format sont bornés ; le grounding peut accepter/refuser à tort des affirmations sémantiques. Un premier plan exige encore la réparation autonome. La reprise actuelle est vérifiée déterministement ; un cycle Ollama assisté complet actuel n'est pas revalidé. Le chargement dépend du matériel ; un timeout réel sans limite de résidence reste conservé. L'attente peut épuiser le budget mural ; saisie synchrone et redémarrage restent limités ci-dessus. Aucune seconde boucle MCP, outil d'écriture, framework ou implémentation M9 ajouté.

## Décisions techniques principales

- fonctionnement local ;
- Python AST pour structurer le code ;
- `all-MiniLM-L6-v2` pour les embeddings ;
- FAISS comme index vectoriel ;
- reranking lexical après la recherche vectorielle ;
- Ollama pour l’exécution des LLM locaux ;
- `Qwen2.5-Coder 14B` retenu après benchmark ;
- séparation claire entre retrieval et génération ;
- conservation systématique de la provenance fichier/lignes ;
- persistance locale de l’index FAISS et des chunks ;
- manifeste SHA-256 pour détecter les changements du projet ;
- mise à jour incrémentale afin de recalculer uniquement les fichiers modifiés ;
- historique des analyses stocké en JSONL ;
- recherche hybride dense + lexicale ;
- validation automatique des citations ;
- vérification du grounding avec le LLM local ;
- MCP pour exposer et exécuter les tools read-only ;
- `Devstral 24B` retenu pour le tool calling ;
- `Qwen2.5-Coder 14B` retenu pour le grounding M6 après benchmark dédié ;
- préchargement Ollama et `keep_alive` pour limiter les cold starts lorsque les modèles peuvent rester résidents ;
- routeurs spécialisés évalués mais non intégrés, Devstral restant plus précis.

## État du projet et suite

M1–M7 terminés. M8 PASS dans son périmètre initial, avec les limites de validation ci-dessus. M9 — Advanced RAG & Reliability reste futur ; toute évolution exige un gain mesuré.

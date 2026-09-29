# CodeAgent — Project Memory

> Mémoire synthétique du projet jusqu’à la fin du Milestone 6.

## État actuel

Les Milestones 1, 2, 3, 4, 5 et 6 sont terminés.

Le socle RAG est terminé à M5 et le tool use contrôlé via MCP est terminé à M6.

CodeAgent est actuellement capable d’analyser un projet Python, d’indexer son code, de retrouver les portions pertinentes pour une question, de générer une réponse sourcée, de vérifier les citations et le grounding, de maintenir sa mémoire projet et d'utiliser des tools read-only via MCP dans une boucle contrôlée.

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

Le socle RAG est terminé à M5 et le tool use contrôlé est terminé à M6.

La suite prévue est M7 — Memory & Context Management, puis M8 — Planning & Controlled Autonomous Execution. Le grounding pourra encore être amélioré avec un modèle spécialisé si un futur benchmark montre un gain mesurable.

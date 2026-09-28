# CodeAgent — Project Memory

> Mémoire synthétique du projet jusqu’à la fin du Milestone 5.

## État actuel

Les Milestones 1, 2, 3, 4 et 5 sont terminés.

Le périmètre principal de CodeAgent est considéré comme terminé à M5.

CodeAgent est actuellement capable d’analyser un projet Python, d’indexer son code, de retrouver les portions pertinentes pour une question, de générer une réponse avec un LLM local en citant les sources utilisées, de vérifier automatiquement les citations et le grounding, puis de conserver et mettre à jour automatiquement la mémoire du projet entre deux exécutions.

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
- vérification du grounding avec le LLM local.

## État du projet et suite

Le périmètre principal de CodeAgent est terminé à M5.

Une évolution future peut ajouter des capacités agentiques avec MCP et des tools. Le grounding pourra également être amélioré avec un modèle NLI ou un second LLM spécialisé.

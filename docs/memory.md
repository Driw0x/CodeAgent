# CodeAgent — Project Memory

> Mémoire synthétique du projet jusqu’à la fin du Milestone 3.

## État actuel

Les Milestones 1, 2 et 3 sont terminés.

CodeAgent est actuellement capable d’analyser un projet Python, d’indexer son code, de retrouver les portions pertinentes pour une question et de générer une réponse avec un LLM local en citant les fichiers et lignes concernés.

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

## Décisions techniques principales

- fonctionnement local ;
- Python AST pour structurer le code ;
- `all-MiniLM-L6-v2` pour les embeddings ;
- FAISS comme index vectoriel ;
- reranking lexical après la recherche vectorielle ;
- Ollama pour l’exécution des LLM locaux ;
- `Qwen2.5-Coder 14B` retenu après benchmark ;
- séparation claire entre retrieval et génération ;
- conservation systématique de la provenance fichier/lignes.

## Prochaine étape

Le prochain milestone est M4 — mémoire projet :

- persistance des index ;
- historique des analyses ;
- mise à jour incrémentale ;
- suivi des modifications du dépôt.

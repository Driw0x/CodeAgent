# M5 — RAG Quality & Grounding

## Objectif

Améliorer la qualité du retrieval et la cohérence entre les sources
récupérées et les réponses générées par CodeAgent.

Le périmètre principal de CodeAgent est déjà considéré comme terminé à M4.
M5 constitue une phase d'amélioration et d'évaluation du système RAG.

## Benchmark

Le benchmark contient 25 questions portant sur les différents composants
du projet.

Chaque question définit les chunks attendus à partir de :

- fichier
- type du chunk
- nom de la fonction, classe ou variable

## Baseline M4

### Dataset

- Questions : 25
- Chunks : 167
- Embedding model : `sentence-transformers/all-MiniLM-L6-v2`

### Raw FAISS

| Metric | Score |
|---|---:|
| Hit@1 | 0.1200 |
| Hit@3 | 0.4000 |
| Hit@5 | 0.6000 |
| Hit@10 | 0.6800 |
| Hit@25 | 0.8000 |
| Recall@25 | 0.7667 |
| MRR@5 | 0.2947 |
| MRR@25 | 0.3139 |

### FAISS + lexical reranking

| Metric | Score |
|---|---:|
| Hit@1 | 0.3200 |
| Hit@3 | 0.6400 |
| Hit@5 | 0.7200 |
| Recall@5 | 0.6667 |
| MRR@5 | 0.4713 |

## Impact du reranking

| Metric | Improvement |
|---|---:|
| Hit@1 | +0.2000 |
| Hit@3 | +0.2400 |
| Hit@5 | +0.1200 |
| MRR@5 | +0.1767 |

Le reranking lexical améliore nettement le classement des chunks
récupérés par FAISS.

## Analyse des erreurs

Sur 25 questions :

- 18 : source pertinente présente dans le top 5 final
- 5 : source absente des 25 candidats FAISS
- 2 : source présente dans les candidats FAISS mais perdue après reranking

Le principal axe d'amélioration identifié est donc le candidate retrieval.

## Expériences prévues

### 1. Analyse des erreurs

Identifier les questions en échec et examiner les chunks retournés.

### 2. Retrieval hybride

Comparer :

- FAISS dense
- recherche lexicale
- dense + lexical

### 3. Reranking

Évaluer différentes stratégies de classement des candidats.

### 4. Grounding

Vérifier que les affirmations générées sont effectivement supportées
par les chunks cités.

### 5. Évaluation finale

Comparer le pipeline M5 à la baseline M4 avec le même benchmark.

## Future improvements

Les améliorations non retenues dans M5 peuvent être conservées ici
pour de futurs travaux.
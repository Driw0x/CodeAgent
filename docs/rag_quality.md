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

## Analyse des erreurs — Baseline M4

Sur 25 questions :

- 18 : source pertinente présente dans le top 5 final
- 5 : source absente des 25 candidats FAISS
- 2 : source présente dans les candidats FAISS mais perdue après reranking

Le principal axe d'amélioration identifié est donc le candidate retrieval.

## Hybrid retrieval — Baseline M5

Une recherche hybride dense + lexicale a été ajoutée afin d'améliorer la couverture des candidats avant reranking.

### Dataset

- Questions : 25
- Chunks : 111
- Embedding model : `sentence-transformers/all-MiniLM-L6-v2`
- Corpus : fixture figée de CodeAgent

Les résultats montrent une amélioration du candidate retrieval par rapport au retrieval dense seul :

| Metric | Dense | Hybrid |
|---|---:|---:|
| Hit@1 | 0.0800 | 0.1600 |
| Hit@3 | 0.4400 | 0.4400 |
| Hit@5 | 0.6400 | 0.6400 |
| Hit@10 | 0.7200 | 0.8000 |
| Hit@25 | 0.8000 | 0.8800 |
| Recall@25 | 0.8000 | 0.8800 |
| MRR@25 | 0.3042 | 0.3574 |

La recherche hybride améliore principalement la couverture des candidats à moyen et grand `k`. Le nombre de questions pour lesquelles aucune source pertinente n'est présente dans les 25 candidats reste limité à 3.

Après fusion avec le reranking lexical via RRF 1:1, le pipeline atteint :

| Metric | M5 hybrid + reranking |
|---|---:|
| Hit@1 | 0.3600 |
| Hit@3 | 0.7200 |
| Hit@5 | 0.8400 |
| Recall@5 | 0.8000 |
| MRR@5 | 0.5413 |

L'analyse des erreurs donne :

- 21 questions correctement résolues dans le top 5 final
- 3 échecs de candidate retrieval
- 1 échec de reranking

La recherche hybride améliore donc la couverture des candidats, tandis que le reranking RRF permet de repositionner efficacement les chunks pertinents dans le top 5 final.

## Grounding

M5 ajoute des citations `[Sx]`, leur validation automatique et une vérification claim-by-claim du support des affirmations par les sources citées.

Le même LLM local est utilisé pour la génération et la vérification. Ce choix conserve une architecture locale simple, mais peut produire des erreurs corrélées.

## Évaluation finale M4 vs M5

Le holdout final contient 16 questions, dont 12 avec sources attendues.

| Metric | M4 | M5 |
|---|---:|---:|
| Hit@1 | 0.4167 | 0.1667 |
| Hit@3 | 0.5833 | 0.5833 |
| Hit@5 | 0.6667 | 0.8333 |
| Recall@5 | 0.6667 | 0.8333 |
| MRR@5 | 0.5069 | 0.3917 |
| Response behavior match | 0.6250 | 0.7500 |

M5 améliore la couverture top-5 et le comportement réponse/abstention, mais dégrade le classement précoce. La latence moyenne end-to-end passe de `4.36 s` pour M4 à `6.90 s` pour M5, soit `+2.54 s`.

Sur M5, 8 réponses ont été vérifiées par le grounding : 6 ont été supportées et 2 rejetées. Aucune citation invalide n'a été observée.

Le holdout final est figé et ne doit plus être utilisé pour retuner M5.

## Future improvements

- modèle NLI ou second LLM spécialisé pour le grounding ;
- amélioration du ranking top-1 ;
- réduction de la latence du grounding ;
- MCP + tools.

# CodeAgent
Projet personnel de création d’un Agent IA local pour l’analyse de code Python et la mémoire de projet.

## Présentation

CodeAgent est un assistant IA local conçu pour analyser des projets Python et répondre à des questions sur leur code.

L’objectif du projet est de créer un agent capable de :

- lire automatiquement les fichiers Python d’un projet,
- indexer le code avec des embeddings,
- comprendre la structure générale du projet,
- répondre à des questions sur le code,
- conserver une mémoire simple du projet,
- utiliser des outils de manière contrôlée lorsque nécessaire,
- planifier les tâches multi-étapes lorsque cela apporte un bénéfice réel.

Le projet fonctionne entièrement en local grâce à un LLM local et une base vectorielle.

### Statut du projet

**Core RAG + Tool Use completed — M6 · M7/M8 planned**

Le socle principal de CodeAgent est terminé et fonctionnel : analyse de code, recherche hybride, pipeline RAG, réponses sourcées, persistance de l'index, mise à jour incrémentale, grounding et utilisation contrôlée de tools via MCP.

Les Milestones 1 à 6 sont terminés. Le prochain milestone ajoute la gestion de la mémoire conversationnelle et du contexte, avant la planification multi-étapes contrôlée.

---

## Fonctionnalités actuelles

### Analyse du code Python

* Lecture récursive des fichiers Python.
* Filtrage des dossiers inutiles (`.venv`, `__pycache__`, etc.).
* Extraction du contenu des fichiers.
* Affichage des chemins relatifs du projet.

### Parsing AST

Extraction automatique des principaux éléments du code :

* Fonctions (`FunctionDef`)
* Classes (`ClassDef`)
* Variables (`Assign`, `AnnAssign`)
* Imports (`Import`)
* Imports spécifiques (`ImportFrom`)

Chaque élément est converti en chunk contenant :

```python
{
    "file": ...,
    "type": ...,
    "name": ...,
    "content": ...,
    "start_line": ...,
    "end_line": ...
}
```

### Recherche sémantique

* Découpage du projet en chunks.
* Génération d'embeddings avec `all-MiniLM-L6-v2`.
* Stockage des vecteurs dans un index FAISS.
* Génération d'embeddings pour les requêtes utilisateur.
* Recherche des k chunks les plus pertinents.
* Recherche sémantique des candidats avec FAISS.
* Recherche hybride dense + lexicale.
* Reranking lexical des chunks candidats.
* Sélection des chunks les plus pertinents pour le contexte RAG.

Exemples de requêtes :

* "function that reads file content"
* "function that reads directory"

---

## Architecture

```text
app
├── agent
├── llm
├── memory
├── parser
├── rag
├── retrieval
├── tools
├── main.py
└── mcp_server.py
```

---

## Technologies utilisées

### Parsing

* Python AST

### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### Base vectorielle

* FAISS

### LLM local

* Ollama
* Qwen2.5-Coder 14B
* Devstral 24B pour le tool calling

### RAG

* FAISS
* Recherche hybride dense + lexicale
* Reranking lexical
* Injection de contexte avec provenance fichier/lignes
* Citations `[Sx]` et vérification du grounding

### Tests

* Pytest

---

## Installation

### 1. Cloner le projet

```bash
git clone <repo>
cd CodeAgent
```

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 3. Installer Ollama

Installer Ollama sur la machine, puis vérifier que l'installation fonctionne :

```bash
ollama --version
```

### 4. Télécharger le LLM local

CodeAgent utilise `qwen2.5-coder:14b` pour le RAG / grounding et `devstral:24b` pour le tool calling.

Télécharger le modèle avec :

```bash
ollama pull qwen2.5-coder:14b
ollama pull devstral:24b
```

Vérifier que le modèle est disponible :

```bash
ollama list
```

---

## Exécution du projet

Depuis la racine du projet, lancer CodeAgent avec :

```bash
python -m app.main
```

Le programme demande ensuite une question sur le projet :

```text
Question : Comment les embeddings sont-ils ajoutés dans FAISS ?
```

CodeAgent propose un mode RAG et un mode Agent. Le mode Agent utilise la boucle MCP M6, précharge les modèles locaux et affiche par question les tokens Devstral / grounding, les temps par étape, les appels de tools, leur durée, le statut du grounding et le temps de réponse.

---

## Documentation

- [Project memory](docs/memory.md)
- [M5 — RAG Quality & Grounding](docs/rag_quality.md)

# Roadmap

## Milestone 1 — Lecture et extraction du code

### Objectifs

* [x] Initialisation du projet
* [x] Lecture récursive des fichiers Python
* [x] Gestion des chemins
* [x] Ignorer les fichiers inutiles
* [x] Extraction du contenu
* [x] Mise en place des tests unitaires

### Résultat

Le projet est capable de parcourir automatiquement un dépôt Python et d'en extraire les informations nécessaires à l'analyse.

---

## Milestone 2 — Embeddings et recherche sémantique

### Objectifs

* [x] Découpage du code en chunks
* [x] Parsing AST
* [x] Génération des embeddings
* [x] Indexation vectorielle
* [x] Recherche des voisins les plus proches
* [x] Embedding des requêtes utilisateur
* [x] Recherche sémantique fonctionnelle

### Résultat

CodeAgent est capable de retrouver automatiquement les portions de code les plus pertinentes à partir d'une question en langage naturel.

---

## Milestone 3 — RAG sur le code

### Benchmark LLM

* [x] Définir un benchmark commun pour comparer les LLM locaux
* [x] Comparer plusieurs LLM locaux
* [x] Sélectionner et intégrer un LLM local

### Pipeline RAG

* [x] Construire le pipeline RAG
* [x] Injecter les chunks retrouvés dans le contexte
* [x] Répondre à des questions sur le projet
* [x] Générer des explications de code
* [x] Référencer les fichiers et lignes concernées

### Exemple

Question :

> Où est définie la fonction qui lit les fichiers ?

Réponse :

> La fonction `read()` est définie dans `app/parser/file_loader.py`, lignes 6-9.

Sources:
- `app/parser/file_loader.py:6-9`

---

## Milestone 4 — Mémoire projet

### Objectifs

* [x] Sauvegarde persistante des index
* [x] Historique des analyses
* [x] Mise à jour incrémentale des embeddings
* [x] Suivi des modifications du projet

---

## Milestone 5 — RAG Quality & Grounding

### Objectifs

- [x] Construire un benchmark RAG
- [x] Ajouter Recall@K, Precision@K et MRR
- [x] Évaluer le retrieval actuel comme baseline
- [x] Ajouter une recherche hybride dense + lexicale
- [x] Évaluer et améliorer le reranking
- [x] Améliorer le contexte envoyé au LLM
- [x] Renforcer le prompt contre les hallucinations
- [x] Ajouter les citations fichier + lignes
- [x] Vérifier automatiquement les citations
- [x] Ajouter une vérification source <=> réponse
- [x] Comparer le pipeline final à la baseline M4

### Résultat

Sur le holdout final de 16 questions, M5 améliore `Hit@5` et `Recall@5` de `0.6667` à `0.8333`, ainsi que le comportement réponse/abstention de `0.6250` à `0.7500`. Le compromis observé est un `MRR@5` plus faible (`0.5069 → 0.3917`) et un surcoût moyen de `+2.54 s` dû au grounding.

---

## Milestone 6 — MCP & Tool Use

### Objectifs

- [x] Définir une interface minimale pour les tools
- [x] Décrire chaque tool avec un nom, une description et un `JSON Schema`
- [x] Valider les paramètres avant exécution
- [x] Implémenter un premier ensemble de tools read-only : lecture de fichiers, recherche de code, listing de fichiers, inspection du diff Git et exécution des tests
- [x] Laisser le LLM décider si un tool est nécessaire et lequel utiliser
- [x] Intégrer les tools via MCP
- [x] Mettre en place la boucle `LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM`
- [x] Ajouter timeout, nombre maximal d'étapes, retries limités et détection des appels identiques `tool + arguments`
- [x] Valider les résultats des tools et retourner des erreurs structurées au LLM
- [x] Réutiliser le grounding M5 pour la réponse finale lorsque des sources projet sont utilisées
- [x] Évaluer le tool use sur un benchmark dédié
- [x] Suivre par question les tokens Devstral / grounding, les étapes, appels de tools, temps par étape et temps individuel des tools
- [x] Optimiser le contexte tool avec recherche bornée, lecture ciblée et consolidation des sources redondantes
- [x] Évaluer séparément les modèles de grounding sur un holdout dédié

### Implémentation

Le premier périmètre reste read-only. Les tools modifiant l'état du projet ne seront ajoutés qu'après stabilisation de la boucle d'exécution et devront demander une confirmation explicite pour les opérations à risque.

Le benchmark suit le taux de réussite des tâches, les appels invalides, retries, appels dupliqués, erreurs de tools, nombre d'étapes, grounding, tokens et latence.

### Résultat

CodeAgent détermine lorsqu'un outil est nécessaire, sélectionne et exécute le bon tool via MCP, récupère d'erreurs simples sans boucle incontrôlée et génère une réponse finale sourcée avant validation du grounding.

Sur le benchmark end-to-end final de 20 tâches, le routing, la sélection du tool, le no-tool et les arguments atteignent `100 %`, tandis que le succès end-to-end atteint `80 %` (`13` réponses groundées acceptées, `4` rejets `unsupported`, `0` `citation_error`). La moyenne est de `2.10` étapes et `1.15` appels de tools par tâche ; la latence moyenne reste élevée à `131.05 s`, principalement à cause des LLM locaux.

Sur un holdout grounding de 40 cas équilibrés, `Qwen2.5-Coder 14B` atteint `97.5 %` d'accuracy, `100 %` de recall sur les cas unsupported, 0 false accept et 1 false reject. `Devstral 24B` atteint `90 %` avec 0 false accept et 4 false rejects ; Qwen14 est donc retenu pour le grounding M6. L'optimisation du contexte réduit le smoke test multi-tool d'environ `10.5k` à `6.6k` tokens, et l'optimisation de `search_code` réduit son temps observé d'environ `17.1 s` à `0.01-0.12 s`.

Sur le benchmark de gating de 20 tâches, Devstral atteint `100 %` de précision ; Laya et Verdict `75 %`, ModernBERT-NLI `65 %` et GLiNER2.5 `45 %`. Jev n'a pas pu être évalué car l'API externe est restée inaccessible malgré un appel conforme à la documentation. Une cascade Verdict -> Devstral testée sur un holdout de 30 tâches réduit la latence moyenne de `42.9 %`, mais atteint `93.3 %` de précision contre `100 %` pour Devstral seul ; elle n'est donc pas retenue.

### Future improvement

- Réévaluer Jev lorsque son API sera accessible et de futurs routeurs / grounders spécialisés uniquement s'ils améliorent le compromis précision / latence sans augmenter les false accepts.

---

## Milestone 7 — Memory & Context Management

### Objectifs

- [ ] Conserver une mémoire court terme des derniers tours de conversation
- [ ] Résumer et compresser les anciens tours
- [ ] Définir un budget de tokens pour le contexte
- [ ] Déclencher automatiquement la compression lorsque le seuil est dépassé
- [ ] Persister le résumé entre deux exécutions
- [ ] Optionnel : récupérer d'anciens échanges pertinents par recherche sémantique

### Résultat attendu

CodeAgent conserve le contexte récent, compresse automatiquement l'historique ancien lorsque le budget est dépassé et peut reprendre une conversation avec un résumé persistant.

---

## Milestone 8 — Planning & Controlled Autonomous Execution

### Objectifs

- [ ] Détecter les tâches nécessitant réellement plusieurs étapes dépendantes
- [ ] Générer un plan court uniquement pour ces tâches
- [ ] Maintenir un état minimal : objectif, étapes terminées, observations, échecs et budget restant
- [ ] Exécuter chaque étape avec la boucle de tools M6
- [ ] Replanifier uniquement les étapes restantes lorsqu'une observation ou une erreur invalide le plan
- [ ] Préserver le travail déjà validé lors d'un replanning
- [ ] Arrêter l'exécution lorsque l'objectif est atteint, que le budget est épuisé ou qu'une décision utilisateur est nécessaire
- [ ] Évaluer les tâches multi-étapes en comparant la boucle directe M6 et la planification M8

### Implémentation prévue

La boucle cible reste simple : `Goal -> Plan optionnel -> Step -> Tool -> Observation -> Next Step / Replan -> Final Answer`.

La planification n'est pas utilisée pour les requêtes simples : elles continuent d'utiliser directement la boucle M6.

### Résultat attendu

CodeAgent est capable d'exécuter de manière contrôlée des tâches de développement multi-étapes, de suivre leur progression et de récupérer d'échecs simples sans recommencer inutilement le travail déjà accompli.

---

## État actuel du projet

CodeAgent dispose désormais d'un pipeline complet permettant d'analyser un projet Python, d'indexer son code, de retrouver les portions pertinentes, de générer des réponses contextualisées avec un LLM local et de maintenir l'index à jour lorsque le projet évolue.

Le socle RAG est terminé à M5 et le tool use contrôlé via MCP est terminé à M6. M7 ajoute la gestion de la mémoire conversationnelle et du contexte ; M8 ajoutera ensuite la planification multi-étapes contrôlée.

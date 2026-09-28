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

**Core RAG completed — M5 · M6/M7 planned**

Le socle principal de CodeAgent est terminé et fonctionnel : analyse de code, recherche hybride, pipeline RAG, réponses sourcées, persistance de l'index, mise à jour incrémentale, validation des citations et vérification du grounding.

Le Milestone 5 — RAG Quality & Grounding est terminé. Les prochains milestones étendent ce socle vers l'utilisation contrôlée d'outils puis la planification multi-étapes.

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
├── llm
├── memory
├── parser
├── rag
├── retrieval
└── main.py
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

CodeAgent utilise par défaut `qwen2.5-coder:14b`.

Télécharger le modèle avec :

```bash
ollama pull qwen2.5-coder:14b
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

CodeAgent recherche les portions de code pertinentes, les injecte dans le contexte du LLM local et génère une réponse accompagnée des fichiers et lignes concernés.

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

- [ ] Définir une interface minimale pour les tools
- [ ] Décrire chaque tool avec un nom, une description et un `JSON Schema`
- [ ] Valider les paramètres avant exécution
- [ ] Implémenter un premier ensemble de tools read-only : lecture de fichiers, recherche de code, listing de fichiers, inspection du diff Git et exécution des tests
- [ ] Laisser le LLM décider si un tool est nécessaire et lequel utiliser
- [ ] Intégrer les tools via MCP
- [ ] Mettre en place la boucle `LLM -> Tool Call -> Validation -> Exécution -> Observation -> LLM`
- [ ] Ajouter timeout, nombre maximal d'étapes, retries limités et détection des appels identiques `tool + arguments`
- [ ] Valider les résultats des tools et retourner des erreurs structurées au LLM
- [ ] Réutiliser le grounding M5 pour la réponse finale lorsque des sources projet sont utilisées
- [ ] Évaluer le tool use sur un benchmark dédié

### Implémentation prévue

Le premier périmètre reste read-only. Les tools modifiant l'état du projet ne seront ajoutés qu'après stabilisation de la boucle d'exécution et devront demander une confirmation explicite pour les opérations à risque.

Le benchmark suivra au minimum le taux de réussite des tâches, les appels invalides, les retries, les appels dupliqués, les erreurs de tools et le nombre d'étapes.

### Résultat attendu

CodeAgent est capable de déterminer lorsqu'un outil est nécessaire, de sélectionner et exécuter le bon tool via MCP, de récupérer d'erreurs simples sans boucle incontrôlée et de produire une réponse finale fondée sur les observations obtenues.

### Future improvement

- Remplacer ou compléter la validation basée sur Qwen par un modèle NLI ou un second LLM spécialisé si les évaluations montrent un gain mesurable.

---

## Milestone 7 — Planning & Controlled Autonomous Execution

### Objectifs

- [ ] Détecter les tâches nécessitant réellement plusieurs étapes dépendantes
- [ ] Générer un plan court uniquement pour ces tâches
- [ ] Maintenir un état minimal : objectif, étapes terminées, observations, échecs et budget restant
- [ ] Exécuter chaque étape avec la boucle de tools M6
- [ ] Replanifier uniquement les étapes restantes lorsqu'une observation ou une erreur invalide le plan
- [ ] Préserver le travail déjà validé lors d'un replanning
- [ ] Arrêter l'exécution lorsque l'objectif est atteint, que le budget est épuisé ou qu'une décision utilisateur est nécessaire
- [ ] Évaluer les tâches multi-étapes en comparant la boucle directe M6 et la planification M7

### Implémentation prévue

La boucle cible reste simple : `Goal -> Plan optionnel -> Step -> Tool -> Observation -> Next Step / Replan -> Final Answer`.

La planification n'est pas utilisée pour les requêtes simples : elles continuent d'utiliser directement la boucle M6.

### Résultat attendu

CodeAgent est capable d'exécuter de manière contrôlée des tâches de développement multi-étapes, de suivre leur progression et de récupérer d'échecs simples sans recommencer inutilement le travail déjà accompli.

---

## État actuel du projet

CodeAgent dispose désormais d'un pipeline complet permettant d'analyser un projet Python, d'indexer son code, de retrouver les portions pertinentes, de générer des réponses contextualisées avec un LLM local et de maintenir l'index à jour lorsque le projet évolue.

Le socle RAG est considéré comme **terminé à M5**. Le pipeline inclut désormais l'évaluation du retrieval, la recherche hybride, les citations vérifiées et le contrôle du grounding. Les M6 et M7 sont planifiés pour ajouter respectivement le tool use via MCP puis la planification multi-étapes contrôlée.

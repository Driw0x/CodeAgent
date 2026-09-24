# CodeAgent
Projet personnel de création d’un Agent IA local pour l’analyse de code Python et la mémoire de projet.

## Présentation

CodeAgent est un assistant IA local conçu pour analyser des projets Python et répondre à des questions sur leur code.

L’objectif du projet est de créer un agent capable de :

- lire automatiquement les fichiers Python d’un projet,
- indexer le code avec des embeddings,
- comprendre la structure générale du projet,
- répondre à des questions sur le code,
- conserver une mémoire simple du projet.

Le projet fonctionne entièrement en local grâce à un LLM local et une base vectorielle.

### Statut du projet

**Core project completed — M4**

Le périmètre principal de CodeAgent est terminé et fonctionnel : analyse de code, recherche sémantique, pipeline RAG, réponses sourcées, persistance de l'index et mise à jour incrémentale.

Le Milestone 5 correspond à des améliorations futures visant à mesurer et renforcer la qualité du retrieval, du grounding et de la cohérence entre les sources et les réponses.

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
* Reranking lexical
* Injection de contexte avec provenance fichier/lignes

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

## Milestone 5 — RAG Quality & Grounding *(amélioration future)*

> Ce milestone est optionnel et n'est pas nécessaire pour considérer le périmètre principal de CodeAgent comme terminé.

### Objectifs

- [x] Construire un benchmark RAG
- [x] Ajouter Recall@K, Precision@K et MRR
- [x] Évaluer le retrieval actuel comme baseline
- [ ] Ajouter une recherche hybride dense + lexicale
- [ ] Évaluer et améliorer le reranking
- [ ] Améliorer le contexte envoyé au LLM
- [ ] Renforcer le prompt contre les hallucinations
- [ ] Ajouter les citations fichier + lignes
- [ ] Vérifier automatiquement les citations
- [ ] Ajouter une vérification source <=> réponse
- [ ] Comparer le pipeline final à la baseline M4

---

## État final du projet

CodeAgent dispose désormais d'un pipeline complet permettant d'analyser un projet Python, d'indexer son code, de retrouver les portions pertinentes, de générer des réponses contextualisées avec un LLM local et de maintenir l'index à jour lorsque le projet évolue.

Le développement principal est considéré comme **terminé à M4**. Le M5 reste une piste d'amélioration facultative consacrée à l'évaluation et au renforcement de la qualité du RAG et du grounding.

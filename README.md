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
* Retour des résultats triés par similarité.

Exemples de requêtes :

* "function that reads file content"
* "function that reads directory"

---

## Architecture

```text
CodeAgent
│
├── app
│   ├── parser
│   │   ├── file_loader.py
│   │   ├── chunker.py
│   │   └── variable.py
│   │
│   ├── memory
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   ├── utils
│   │   └── paths.py
│   │
│   └── main.py
│
├── tests
│
├── data
│
└── README.md
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

### Tests

* Pytest

---

## Installation

```bash
git clone <repo>
cd CodeAgent

pip install -r requirements.txt
```

---

## Exécution du projet

Depuis la racine du projet, lancer le programme principal avec :

```bash
python -m app.main
```

---

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

### Objectifs

* [ ] Ajouter un LLM local
* [ ] Construire le pipeline RAG
* [ ] Injecter les chunks retrouvés dans le contexte
* [ ] Répondre à des questions sur le projet
* [ ] Générer des explications de code
* [ ] Référencer les fichiers et lignes concernées

### Exemple attendu

Question :

> Où est définie la fonction qui lit les fichiers ?

Réponse :

> La fonction `read_file()` est définie dans `app/parser/file_loader.py`.
> Elle est utilisée pour charger le contenu des fichiers Python avant leur découpage en chunks.

---

## Milestone 4 — Mémoire projet

### Objectifs

* [ ] Sauvegarde persistante des index
* [ ] Historique des analyses
* [ ] Mise à jour incrémentale des embeddings
* [ ] Suivi des modifications du projet

---

## Objectif final

Construire un véritable assistant IA local capable de comprendre un projet logiciel, de conserver sa mémoire et d'assister efficacement le développeur dans ses tâches quotidiennes.

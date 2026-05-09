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
- et plus tard effectuer des analyses statiques et de sécurité.

Le projet fonctionne entièrement en local grâce à un LLM local et une base vectorielle.

---

## Roadmap

### Milestone 1 — Lecture du projet Python 
Objectif actuel :
- [x] Initialisation du projet
- [ ] Lecture récursive des fichiers `.py`
- [ ] Ignorer les dossiers inutiles (`.venv`, `__pycache__`, etc.)
- [ ] Extraction du contenu des fichiers
- [ ] Affichage des fichiers détectés

But :
Créer une première base capable de parcourir automatiquement un projet Python et récupérer son code.

---

### Milestone 2 — Embeddings et indexation
Objectifs :
- [ ] Découpage du code en chunks
- [ ] Génération des embeddings
- [ ] Stockage
- [ ] Recherche sémantique

---

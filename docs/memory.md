# CodeAgent — Project Memory

[Français](memory.md) | [简体中文](memory.zh-CN.md)

[README du projet](../README.md)

> Mémoire synthétique du projet : M1–M6 terminés, M7 terminé dans son périmètre mémoire et contexte ; limites autonomes documentées.

## État actuel

Les Milestones 1, 2, 3, 4, 5 et 6 sont terminés.

Le socle RAG est terminé à M5 et le tool use contrôlé via MCP est terminé à M6.

CodeAgent est actuellement capable d’analyser un projet Python, d’indexer son code, de retrouver les portions pertinentes pour une question, de générer une réponse sourcée, de vérifier les citations et le grounding, de maintenir sa mémoire projet et d'utiliser des tools read-only via MCP dans une boucle contrôlée.

M7 ajoute les sessions conversationnelles persistantes, leur compaction extractive et un budget commun au RAG,
aux observations MCP et aux requêtes Ollama. Les résultats, limites et protocoles M7 sont consolidés ci-dessous.

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

## M7 — Memory & Context Management

**M7 est terminé dans son périmètre fonctionnel.** Mémoire, budgets, compaction, reprise, isolation et
conservation des sources sont vérifiés. Cette clôture ne garantit pas la fiabilité de toutes les finales autonomes.
M8 et M9 restent des étapes futures.

### Architecture et persistance

La mémoire de projet (`index.faiss`, `chunks.json`, `manifest.json`) reste séparée de la conversation.
L'archive existante `history.jsonl` reçoit des champs optionnels `session_id` et `observations`, tout en conservant
`timestamp`, `question` et `answer`. Les anciennes analyses sans session restent lisibles et ne sont pas importées
automatiquement dans une nouvelle conversation.

`ConversationMemory` restaure uniquement la session demandée. Un checkpoint atomique
`sessions/<session_id>.json` conserve résumé, compteur de compaction et frontière des tours déjà traités.
Les checkpoints v1 sont reconstruits depuis le JSONL pour restaurer les décisions réaffirmées ;
les nouveaux checkpoints sont v2. Un état corrompu est reconstruit ; une version future ou un checkpoint
en avance sur l’archive est refusé sans écrasement. Les échanges récents sont reconstruits depuis l'archive, sans copie persistante supplémentaire du texte brut.

Le résumé contient des extraits avec rôle, ordre et priorité. Décisions, contraintes, chemins, erreurs et hypothèses
restent identifiables. Aucun modèle de résumé n'est appelé. Les informations anciennes pertinentes peuvent être
retrouvées par recherche lexicale en streaming dans l'archive. Les citations historiques sont explicitement
distinguées des sources du tour courant ; le grounding continue à vérifier uniquement les sources actuelles.

`ContextBudget` compte messages, schémas des tools et observations avec une estimation conservatrice :
un token par octet UTF-8, 32 unités par message et 256 unités réservées au framing. La génération a sa propre réserve
`num_predict`. `--context-tokens` (ou `CODEAGENT_CONTEXT_TOKENS`, défaut 16 384) configure à la fois budget et `num_ctx`
Ollama. Un élément indispensable trop volumineux produit une erreur explicite avant envoi, sans troncature arbitraire.

Les gros résultats de tools sont réduits en objets ou lignes entières, avec chemins, lignes et valeurs attachées.
Le résultat brut reste dans l'archive. La réponse finale et le grounding utilisent la même sélection de sources.
`--project` sépare également les index/sessions de projets différents et transmet le projet au serveur MCP stdio.

Politique de rétention : quatre tours bruts récents par défaut dans le contexte agent, sélection des tours plus anciens,
résumé borné à la moitié du budget mémoire ; éviction des extraits non prioritaires puis des plus anciens.
L'archive append-only n'est jamais purgée automatiquement. La récupération lexicale reste linéaire dans l'archive.
Une session persistante suppose un seul processus écrivain ; les écritures concurrentes et la rétention disque avancée
restent hors du périmètre local actuel.

### Réalisations M7.1 à M7.6

| Étape | Réalisation | Fichiers principaux | Vérification |
|---|---|---|---|
| M7.1 | Tours ordonnés, contexte récent, décisions et contraintes, sessions/projets séparés | `app/memory/conversation.py` | Rôles, corrections, tours anciens non compactés et absence de contamination |
| M7.2 | Budget commun système/mémoire/RAG/tools/schémas, réserve réponse et `num_ctx` explicite | `app/memory/context.py`, `app/llm/local_llm.py` | UTF-8, frontière exacte, refus avant réseau, plusieurs observations |
| M7.3 | Compaction automatique extractive, priorités, déduplication et éviction | `app/memory/conversation.py` | Extraits exacts, chronologie ORION/VEGA/ORION, hypothèses non promues en faits |
| M7.4 | Archive JSONL réutilisée, checkpoint atomique, reprise et reconstruction | `app/memory/history.py`, `app/memory/conversation.py` | Redémarrages, archive partielle, corruption, versions futures |
| M7.5 | Recherche lexicale ciblée et sélection structurée avant réduction | `app/memory/context.py`, `app/agent/tool_loop.py`, `app/rag/pipeline.py` | JSON valide, lignes/objets complets, valeurs exactes et sources communes au grounding |
| M7.6 | Intégration RAG/Agent, CLI et benchmark M6/M7 déterministe | `app/main.py`, `scripts/benchmark_memory.py` | MCP en mémoire/stdio, suivi RAG, budget, reprise, erreur récupérée |

`--session` choisit la conversation ; `--project` choisit le projet. Son chemin résolu détermine le dossier
mémoire hashé : deux projets de même nom restent séparés. Le projet est transmis explicitement au MCP stdio.
Les questions RAG explicites restent intactes ; un suivi pronominal simple peut reprendre la question précédente.
La reformulation sémantique complète et la recherche sémantique d'anciens échanges restent reportées.

### Sources et corrections vérifiées

La chaîne est `observation MCP originale → source → sélection M7 → prompt final → citations → grounding`.
`build_tool_source` garde chemin, bornes effectives, arguments et résultat. Les réductions travaillent sur des copies.
`consolidate_sources` retire les sources redondantes ; `fit_sources` renumérote les sources retenues de façon contiguë.
Le grounding reçoit exactement cette liste finale. Une erreur ne prouve que son diagnostic, jamais le contenu demandé.
Les données historiques ne deviennent pas des preuves du code actuel ; une source modifiée doit être relue.

Trois causes M7 démontrées ont été corrigées lors de la dernière campagne :

- Une réaffirmation ORION/VEGA/ORION perdait son dernier événement. La dernière occurrence conserve désormais son tour ; les checkpoints v1 sont reconstruits.
- Cinq observations pouvaient dépasser le budget : 9 133 estimés pour 6 444 disponibles. Le partage réserve maintenant système, appels, schémas et échappement JSON ; cinq IDs, valeurs et lignes restent présents.
- L'historique facultatif pouvait évincer la lecture récupérée après 60 échanges. Les sources actuelles passent désormais avant l'historique ; `source_valid` et `tool_error_recovered` repassent.

Les corrections associées déjà livrées sont conservées, sans nouveau changement de la boucle dans cette finalisation :

- Diagnostics `ValueError` attendus transmis comme `ToolError` pour les cinq tools. Un départ après EOF est refusé, y compris sur fichier vide ; une fin trop grande est bornée.
- Erreurs, exceptions et timeouts réellement exécutés conservés dans les sources finales de récupération.
- Feuilles de `TaskGroup` affichées et persistées avec type/message ; le groupe original est relancé. L'exception historique de `analyse-1` reste inconnue, non reproduite.
- Les faits d'une abstention partielle restent contrôlés. Sans lecture réussie, une annonce fictive de récupération devient une abstention.
- Une citation manquante à source unique est ajoutée seulement après vérification de toutes les claims. Aucune attribution automatique entre plusieurs sources ; les références invalides restent rejetées.
- Les demandes explicites sur le code du projet sans inspection actuelle produisent une abstention. Ce garde-fou reste heuristique ; les références indirectes dépendent du routeur.
- L'URL Ollama par défaut utilise `127.0.0.1`. Les URLs explicites restent respectées.

`--debug` ou `CODEAGENT_DEBUG=1` trace contextes, réponses de routage, candidate brute, prompt final,
sources avant/après sélection, mapping des IDs, claims et métriques par requête.
`missing_references`, `invalid_references` et `verifier_rejected_claim` distinguent les rejets ;
`supported=null` signale une exception, pas un verdict négatif. Les diagnostics optionnels sont enregistrés
localement dans le JSONL, jamais réinjectés dans la conversation. Le traceback complet reste dans le terminal.

Nettoyage associé préexistant : suppression de l'alias/commentaires inactifs d'embeddings et du tronquage
`compact_text`, réutilisation du reranking existant, fixture `anyio_backend` centralisée.
Aucun test historique ou fixture utile supprimé. L'audit n'a identifié aucun module entier prouvé inutilisé.
Le cache pytest et `tmp_path` vont dans les temporaires système ; `--basetemp` dans le dépôt est refusé.
`data/memory/`, les index et sessions utilisateur restent locaux, ignorés par Git et conservés.

### Tests et benchmarks

Campagne de référence du 14 octobre 2026 : **220 tests réussis**. Progression historique :
133 cas initiaux, 134 après ajout du test FAISS incrémental, puis 172, 188, 198 et 220.
Les 22 derniers cas nets vérifient notamment chronologie, budgets, sélection des sources et citations.
Aucun compteur artificiellement conservé. Aucun lint, type-check ou build supplémentaire configuré.

La finalisation du dépôt reproduit **220 passed en 6,58 s**, puis **220 passed en 5,46 s** après nettoyage,
sans échec ni test ignoré : variation de compteur **0**.
Le benchmark mémoire reproduit les six critères, **9/9 rappels M7**, **0/9** avec remise à zéro par tour,
maximum **6 271**, limite **6 444**, replay complet **48 957**, zéro appel de résumé LLM.
Le fichier `benchmarks/memory/results/m7.json` contient cette mesure et ses latences de fixtures.
L'import inutilisé `estimate_tokens` de son script a été supprimé pendant cette finalisation.

#### Protocole déterministe et résultats historiques

`deterministic-memory-v1` utilise trois conversations de 63 échanges prédéfinis (dont 60 échanges secondaires),
trois rappels chacune, une reprise au milieu, puis un fichier de 600 lignes lu après une erreur de fichier absent.
Le client MCP en mémoire et les tools sont réels ; les LLM de réponse/grounding sont des fixtures.
Les six critères exigent tous les rappels, extraits exacts, budget respecté, sources actuelles,
récupération des erreurs et aucun modèle de résumé. Ils ne mesurent pas la compréhension d'Ollama.

Première campagne conservée sous forme de mesures exactes :

| Mesure | M6 : remise à zéro par tour | M7 |
|---|---:|---:|
| Rappels littéraux réussis | 0/9 | 9/9 |
| Contexte maximum estimé | 5 354 | 5 945 |
| Limite d'entrée | 6 444 | 6 444 |
| Replay intégral, comparaison de taille séparée | jusqu'à 48 957 | jusqu'à 48 957 |
| Compressions par conversation | 0 | 62 |
| Tours compactés en fin de conversation | 0 | 65 |
| Fidélité des extraits anciens sélectionnés | pas de rappel | exacte sur les fixtures |
| Réponses finales sourcées / grounding de fixture | 3/3 | 3/3 |
| Erreurs de tool récupérées | 3/3 | 3/3 |
| Appels LLM de fixture par conversation | 8 | 8 |
| Appels de résumé au LLM | 0 | 0 |
| Appels de tools par conversation | 2 | 2 |
| Moyenne de latence par cas | 5,77–8,43 ms | 29,44–42,18 ms |
| P50 par cas | 1,07–3,16 ms | 19,27–22,16 ms |
| P95 par cas (4 questions mesurées) | 20,10–26,71 ms | 63,55–104,18 ms |
| Tokens réellement mesurés par modèle | non mesurés | non mesurés |

La dernière campagne corrige les allocations et atteint 6 271/6 444 ; la campagne intermédiaire
`m7_robustness.json` mesurait 6 418/6 444, toujours 9/9 rappels et six critères vrais.
La réduction d'environ 88 % compare M7 au replay intégral, pas au petit contexte sans mémoire.
Le comparateur M6 simule une remise à zéro avec les tools/budgets courants, pas l'exécution du commit M6.
Les latences de fixtures ont quatre questions mesurées par conversation ; elles ne prouvent aucun gain de latence modèle.
L'archive reste parcourue linéairement ; aucune indexation supplémentaire n'est justifiée sans mesure de volume.

#### Régression RAG et grounding historique

Sur le corpus figé de **25 questions / 133 chunks**, les résultats avant/après M7 sont :

| Mesure | M4 baseline | M5 reranking/contexte |
|---|---:|---:|
| Hit@5 | 0,7200 | 0,8400 |
| Recall@5 | 0,7200 | 0,8200 |
| MRR@5 | 0,5233 | 0,5440 |
| Contexte moyen, caractères | Non mesuré | 2 389,36 |
| Sources pertinentes perdues | Non mesuré | 0 |

Les trois agrégats de `benchmarks/rag/results/audit_baseline/` et
`benchmarks/memory/results/m7_rag_regression/` sont identiques.
La finalisation reproduit ces résultats sur les mêmes 25 questions et 133 chunks, MiniLM en mode offline :
Hit@5 0,84, Recall@5 0,82, MRR@5 0,544 et aucune source pertinente perdue.
Ce protocole reste distinct des résultats M5 historiques sur 111/167 chunks et du holdout de 16 questions.
Le benchmark RAG utilise MiniLM local ; il ne mesure ni la génération ni le grounding Ollama.

Le holdout Qwen réel `m7_grounding_holdout.json` conserve **39/40**, précision **0,975**,
zéro faux accepté, un faux rejet : `grounding_output_supported_01`.
La source impose explicitement `SUPPORTED`/`UNSUPPORTED` et supporte la claim rejetée.
Mesures de cette relance : préchargement mural **173,207 s**, moyenne chaude **2,853 s**,
médiane **2,560 s**, P95 **4,961 s**, total des 40 contrôles **114,136 s**, zéro erreur réseau.
La première tentative échoue au préchargement après 300 s, zéro cas évalué ; sa cause exacte reste inconnue.
Les durées absentes retournées comme zéro par `preload` ne sont pas des mesures de chargement nul.
Le holdout complet n'est pas relancé dans la dernière campagne sources, ni modifié pour obtenir 40/40.

#### Campagnes Ollama et autonomie

Les critères S1–S8 et fenêtres du protocole `m7-final-v2` sont dans
`benchmarks/memory/final_cases.json`, exécutés par `scripts/benchmark_m7_final.py`.
Les conversations réelles, le redémarrage dans un nouveau processus, l'historique de compaction prédéfini,
les rejeux de sources MCP fixées et les parcours autonomes sont des mesures distinctes.
La matrice suivante conserve les exécutions terminées et leurs échecs ; les essais interrompus sont exclus.

| Scénario réel | Exécutions terminées | Réussites | Échecs | Causes | Statut |
|---|---:|---:|---:|---|---|
| S1 Mémoire après trois questions intermédiaires | 3 | 3 | 0 | Deux faits exacts, zéro outil, `not_used` | Deux campagnes précédentes et code final vérifiés |
| S2 Compression et correction | 3 | 2 revues factuelles | 1 rejet de finale | Premier rappel invente des citations ; les deux suivants préservent ORION, valeur exacte et contrainte négative | Deux rappels fidèles après clarification ; premier échec conservé |
| S3 Persistance dans un nouveau processus Python | 3 | 3 | 0 | Même archive, même session ; deux faits exacts, zéro outil | Vérifié, y compris code final |
| S4 Isolation d'une autre session | 3 | 3 revues factuelles | 0 | Aucun ASTER-8617 ; absence de code et contrainte explicitement indiquée | Vérifié, y compris code final |
| S5 Finale FAISS à sources MCP fixées, v1 | 2 | 0 | 2 | Génération citée, puis timeout Qwen à 300 s | Ancienne campagne conservée |
| S5 Finale FAISS à sources MCP fixées, v2 | 3 | 3 | 0 | Faits exacts, quatre claims vérifiées par finale, références valides | Vérifié sur ces sources ; aucune inspection autonome déduite |
| S5 Parcours autonome avant garde-fou | 3 + 3 | 0 | 6 | Aucun tool malgré la consigne d'inspection ; usages du projet non prouvés | Limite de routage Devstral ; branche héritée sécurisée ensuite |
| S5 Parcours autonome après garde-fou | 1 | 0 réponse utile | 1 abstention | Candidate sans preuve remplacée par une abstention ; zéro tool | Protection vérifiée ; recherche autonome toujours non obtenue |
| S6 Finale de récupération à observations MCP fixées, v1 | 2 | 0 | 2 | Citation de récupération absente ; ou récupération omise puis timeout Qwen | Ancienne campagne conservée |
| S6 Finale de récupération à observations MCP fixées, v2 | 3 | 0 | 3 | Un diagnostic supporté mais récupération omise ; deux paragraphes de correction sans citation | Sources présentes ; génération non fiable |
| S6 Parcours autonome, première campagne v2 | 3 | 0 finale complète | 3 | Trois vraies lectures 200–226 ; une finale sans citation et deux finales omettant la récupération | Transport et récupération 3/3 ; génération 0/3 |
| S6 Parcours autonome après clarification des consignes | 3 | 0 finale complète | 3 | Trois vraies lectures 220–226 ; bloc de code et citation isolée, rejet avant Qwen | Transport et récupération 3/3 ; génération 0/3 |
| S6 Parcours autonome après garde-fou | 1 | 0 finale complète | 1 | Lecture 220–226 réussie ; même citation isolée et rejet | Aucune correction de citations revendiquée |
| S7 Rappel exclusivement conversationnel | 3 | 3 | 0 | Zéro outil, faits corrects, `not_used` | Vérifié ; routage supplémentaire 3/3 dans chacune des deux campagnes |
| S8 Claim non supportée, Qwen réel | 1 | 1 | 0 | Affirmation Kubernetes rejetée `UNSUPPORTED` en 201,359 s | Vérifié une fois ; aucun retuning du vérificateur |

Les sept S6 autonomes récupèrent réellement `read_file`, mais **0/7 finales complètes** passent.
Les finales FAISS à sources fixées v2 passent **3/3** ; elles ne prouvent pas une inspection autonome.
Les fenêtres v1 (retriever 1–90) et v2 (70–115, incluant `index.search`) diffèrent : aucun gain avant/après déduit.
La compaction S2 utilise un historique prédéfini puis une réponse Ollama réelle ; les autres rappels sont conversationnels.
Deux faux échecs de matching littéral sont conservés : `0,545` et « aucune dépendance ne doit être ajoutée »
en S2, « ni code ni contrainte » en S4. La revue factuelle les distingue du vrai premier rejet de citations S2.
Le code final S2 retourne ORION, 0.545 et la contrainte correcte, avec 12 compressions et zéro tool ;
`success: false` reste inchangé pour sa paraphrase dans `real_memory_final.json`.

Routage premier appel, dataset `routing_cases.json` : historique **1/3**, puis **2/3**, puis **3/3**,
conservés dans `m7_routing_before.json`, `m7_routing_after.json`, `m7_routing_final.json`.
Les rappels connus/absents restent sans tool ; la demande mixte sélectionne `get_git_diff`.
Une revalidation du code final réussit aussi **3/3**, latences **8,408/9,648/6,816 s**.
Cela ne mesure pas l'exécution des tools ni la qualité de leur finale ; aucun gain de latence annoncé.
La finalisation reproduit aussi **3/3**, zéro erreur, avec les latences **8,1779/8,9203/6,5370 s**
(moyenne **7,8784 s**). Le protocole reste celui du premier appel ; S5/S6 et le holdout Qwen ne sont pas relancés.
Le cas de session vide retourne pourtant « code de test `test_1` » et « priorité `P1` », absents de tout
historique transmis. Le routage sans tool est correct, le contenu est inventé. Le rappel connu retrouve
ORCHIDEE-9274 et turquoise. Cette nouvelle limite de génération n'est ni une contamination de session
ni une preuve de rappel fidèle ; le score 3/3 reste strictement un score de routage.

Le sous-ensemble MCP historique `no_tool_01`, `read_file_03`, `recovery_01`, `multi_step_01`
atteint **4/4** en routage/arguments et **2/4** end-to-end avant les dernières corrections de citations.
Il ne remplace pas les 20 tâches M6. Le rejet `multi_step_01` cite la recherche S1 pour une définition
présente dans la lecture S2. L'introduction « Voici un résumé des lignes 1 à 40 : » est ensuite rejetée
comme claim : limite d'extraction M5, sans règle ad hoc pour la contourner.
Un ancien rejeu de récupération a été accepté (six claims, **86,347 s**, dont **71,725 s** de finale et
**14,619 s** de grounding), mais accepte une formulation imprécise sur des scores bruts au lieu des rangs RRF.
Il ne valide ni les décisions autonomes ni l'exactitude universelle du vérificateur.

Comparaison M6/M7 à mêmes sources MCP et candidates : contenus, chemins et références intacts ;
candidate correctement citée acceptée dans les deux, candidate sans citation à plusieurs sources rejetée.
Le grounder de cette comparaison est une fixture. Les omissions de Devstral ne sont pas démontrées comme
un effet de compression. Le contrôle d'abstention partielle corrige, lui, un défaut hérité d'acceptation de S9.
La réparation générative expérimentale a donné **0/3 réparations complètes** (**17,699/17,517/17,446 s**),
après un premier essai général de **22,376 s** sans correction. Elle est retirée du code livré.

#### Performances Ollama mesurées

Paramètres observés : `keep_alive=30m`, `num_ctx=16384`, `num_predict=512`, température 0, seed 42,
timeout HTTP 300 s ; Agent : cinq étapes, un retry par tool, timeout tool 60 s.
Qwen est préchargé avant Devstral ; un warm-up échoué laisse continuer la session. Aucun déchargement
applicatif forcé ni changement de modèle, grounding, timeout ou réglage GPU dans cette finalisation.

| Rejeu, sans temps d'outils dans cette phase | Génération finale | Grounding | Temps mural | Résultat |
|---|---:|---:|---:|---|
| FAISS 1 | 36,042 s | 300,034 s | 336,078 s | Timeout |
| FAISS 2 | 15,107 s | 300,002 s | 315,111 s | Timeout |
| read_file 1 | 47,454 s | Aucun appel | 47,456 s | Citation manquante |
| read_file 2 | 20,048 s | 300,012 s | 320,061 s | Timeout ; récupération omise |
| FAISS v2, 1 | 157,167 s | 10,680 s | 167,849 s | Accepté, quatre claims supportées |
| FAISS v2, 2 | 33,564 s | 6,587 s | 40,152 s | Accepté, mêmes faits et sources |
| FAISS v2, 3 | 33,468 s | 4,798 s | 38,267 s | Accepté, mêmes faits et sources |
| read_file v2, 1 | 20,270 s | 1,302 s | 21,574 s | Diagnostic accepté, récupération omise |
| read_file v2, 2 | 61,830 s | Aucun appel | 61,832 s | Citation de récupération manquante |
| read_file v2, 3 | 61,191 s | Aucun appel | 61,193 s | Citation de récupération manquante |

Pour FAISS v2, première évaluation de prompt **121 140,123 ms**, chargement **471,416 ms**,
génération **34 663,701 ms** ; répétitions 2/3 : prompt **283,549/285,094 ms**,
génération **33 132,572/33 020,726 ms**. Le coût initial est surtout celui du prompt, pas 157 s de chargement.
S8 : chargement Qwen **193 261,843 ms**, prompt **7 617,799 ms**, génération **269,208 ms**,
temps mural **201,359 s**. Cela n'explique pas rétrospectivement les anciens timeouts.

S6 autonome après clarification : **98,767/97,551/98,094 s**, tools **0,103/0,013/0,013 s**,
setup MCP **0,740/0,726/0,703 s**, aucun grounding après rejet de citations.
Après garde-fou : **99,700 s**, agent **60,473 s**, finale **38,244 s**, tools **0,030 s**,
setup **0,757 s**, résidu environ **0,196 s** incluant fermeture du transport et traitement local.
Les chemins et sorties diffèrent entre campagnes ; aucun speedup end-to-end revendiqué.
Les premiers agrégats `tokens=0` des rejeux étaient un défaut de rapport : compteurs exacts dans
`diagnostics.requests`, script corrigé par `_refresh_token_stats` pour les mesures suivantes.

Seul gain réseau isolé : trois GET `/api/ps` via `localhost` en **2,055/2,034/2,032 s**,
contre **0,0161/0,000935/0,000627 s** via IPv4. Le serveur historique Ollama 0.40.3 écoute en IPv4 ;
`localhost` résout IPv6 d'abord sur cette machine.
Devstral est déclaré 10 % CPU / 90 % GPU, Qwen 100 % GPU ; tailles VRAM déclarées
**15 705 047 040** et **12 031 780 453 octets**, environ 25,8 Gio pour 16 Go physiques (RX 7800 XT).
Cela ne prouve pas une résidence simultanée. La RAM libre atteint **588 860 Kio** lors d'une campagne,
**708 092 Kio** (692 Mio) lors de la suivante ; le remplacement du runner Qwen est observé.
Pression mémoire établie, causalité de pagination/timeouts non mesurée. Les logs accessibles à 17:07
ne correspondent pas aux requêtes en cause. Le warm-up bloqué dans `socket.readinto` reste de cause inconnue ;
une campagne ultérieure avec modèle chargé termine routage et rappels. Les interruptions ne sont pas comptées.

### Limites connues

Aucun bug M7 démontré ne reste ouvert dans l'état livré ; les tests ne garantissent pas l'absence de tout défaut.
La mémoire extractive conserve les données, mais peut évincer un extrait du résumé ; l'archive originale reste disponible.
Les phrases indivisibles trop grandes restent dans l'archive et exigent une demande ciblée.
Le rappel lexical n'est pas un rappel sémantique universel. Une session suppose un seul processus écrivain ;
concurrence et rétention disque avancée restent reportées. L'index FAISS/manifest n'est pas transactionnel,
un renommage reste suppression/ajout et `ast.unparse` normalise le texte malgré les lignes originales.

Les limites autonomes sont conservées : FAISS sans inspection peut conduire à une abstention,
récupérations `read_file` réelles **7/7**, finales complètes **0/7** ; citations omises ou mal associées,
récupération omise, décisions de plage/chemin variables. Les références indirectes peuvent dépasser le garde-fou.
Un rappel sans historique peut également inventer une réponse malgré un routage sans tool correct,
comme `test_1`/`P1` dans la revalidation finale. L'absence de contamination mémoire n'empêche pas cette hallucination.
La mémoire ne remplace ni des sources actuelles ni leur vérification.

Limites du grounding indépendantes de M7 : introductions/blocs de code traités comme claims,
citations d'un paragraphe appliquées à plusieurs phrases, faux rejet `grounding_output_supported_01`,
qualificatif « efficaces » et combinaison de scores bruts acceptés sans preuve exacte.
`m9_regressions` dans `final_cases.json` conserve `weighted_raw_scores`, `exact_rrf_ranks` et le faux rejet historique.
Ces cas ne sont pas exécutés avec Qwen dans la campagne de clôture ; **39/40** reste inchangé.
Leur présence documente des limites reportées en M9, sans commencer M8 ni M9, ni assouplir le grounding.
Les modèles locaux, la résidence simultanée et les ressources matérielles limitent les performances ;
aucun résultat rejeté sans grounding ne se compare à une réponse acceptée avec grounding.

### Commandes de reproduction

Depuis la racine, activer l'environnement contenant `requirements.txt` :
Windows `.\.venv\Scripts\Activate.ps1` ; Linux/WSL `source .venv/bin/activate`.
Exécuter séquentiellement ; les sorties de revalidation restent dans les temporaires système.
Les tests et le benchmark mémoire ne nécessitent ni réseau ni Ollama.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest -q
python -m scripts.benchmark_memory
$env:HF_HUB_OFFLINE = '1'
python -m scripts.benchmark_rag --output-dir (Join-Path $env:TEMP 'codeagent-m7-rag-final')
python -m app.main --help
python -m scripts.benchmark_memory --help
python -m scripts.benchmark_m7_final --help
python -m compileall -q app scripts tests conftest.py
git diff --check
```

MiniLM doit déjà être disponible pour le benchmark RAG. Ne pas confondre son mode offline avec une fixture d'embeddings.
Contrôles ciblés (leur compteur dépend de l'état courant de la suite) :

```powershell
python -m pytest -q tests/test_conversation.py -k "recent_roles or sessions_projects or compaction or reaffirmed or checkpoint"
python -m pytest -q tests/test_conversation.py tests/test_memory_integration.py -k "budget or bounded or large_sources or observation or source or citation"
python -m pytest -q tests/test_memory_integration.py
python -m pytest --cache-show
```

La suite inclut aussi les tests historiques parser, embeddings, index incrémental, manifeste,
retrieval, citations, grounding, boucle, timing et optimisation des tools. Le test de temporaires
vérifie cache et `tmp_path` hors dépôt ainsi que l'absence des anciens dossiers pytest.
Pour éviter le bytecode local lors d'une compilation, définir `PYTHONPYCACHEPREFIX` vers un chemin temporaire système.

Pour Ollama, conserver les modèles déjà installés `qwen2.5-coder:14b` et `devstral:24b`.
Vérifier `ollama list`, démarrer `ollama serve` dans un autre terminal si nécessaire, puis lancer chaque mesure
après la précédente. Aucune deuxième génération Ollama pendant ces essais.
Le holdout complet décharge/précharge Qwen intentionnellement pour son protocole à froid.

```powershell
ollama ps
Invoke-RestMethod http://127.0.0.1:11434/api/ps | ConvertTo-Json -Depth 8
python -m scripts.benchmark_tool_routing --dataset benchmarks/memory/routing_cases.json --output (Join-Path $env:TEMP 'codeagent-m7-routing-final.json')
python -m scripts.benchmark_m7_final --mode memory --output (Join-Path $env:TEMP 'codeagent-m7-real-memory-final.json')
python -m scripts.benchmark_m7_final --mode finals --repetitions 3 --output (Join-Path $env:TEMP 'codeagent-m7-fixed-finals.json')
python -m scripts.benchmark_m7_final --mode autonomous --repetitions 3 --output (Join-Path $env:TEMP 'codeagent-m7-autonomous-final.json')
python -m scripts.benchmark_m7_final --mode unsupported --output (Join-Path $env:TEMP 'codeagent-m7-unsupported-final.json')
python -m scripts.benchmark_grounding_models --models qwen2.5-coder:14b --output (Join-Path $env:TEMP 'codeagent-m7-grounding-holdout.json')
```

Les drapeaux littéraux du script restent à compléter par une revue de la candidate et de chaque claim.
Un timeout est une exception, jamais un verdict `UNSUPPORTED`. Conserver état `/api/ps`, métriques Ollama,
RAM/VRAM, timings muraux et journal `--debug` avant tout diagnostic matériel.

### Validation manuelle

Créer une session dédiée ; les journaux vont hors dépôt, la mémoire applicative reste dans `data/memory/` :

```powershell
python -m app.main --project . --session m7-personal-final --debug 2>&1 |
    Tee-Object -FilePath (Join-Path $env:TEMP 'codeagent-m7-personal-final.log')
```

Choisir **2** (Agent) et envoyer chaque question sur une ligne :

1. `Dans cette session, retiens le code ASTER-8617 et la contrainte aucune dépendance ajoutée. Réponds brièvement.`
2. Poser trois questions intermédiaires, puis `Quel code et quelle contrainte ai-je mentionnés dans cette session ?`
3. Quitter par entrée vide, arrêter le processus, relancer la même commande et répéter le rappel. Changer ensuite `--session` : les faits doivent être absents, zéro tool.
4. `Explique-moi brièvement le rôle de FAISS dans CodeAgent.` Vérifier les tools réellement exécutés, les sources, citations, grounding et faits ; une abstention ne valide pas l'inspection.
5. `Lis app/retrieval/faiss_retriever.py des lignes 227 à 277. Si la plage est invalide, corrige-la.` Vérifier l'erreur, la lecture réellement récupérée et la finale citant diagnostic et récupération ; aucune action fictive ni question supplémentaire pour masquer son échec.

Les rappels attendent ASTER-8617 et la contrainte exacte, zéro tool, `Grounding: not_used`.
Une finale technique attend les faits supportés et `Grounding: passed`. La longueur actuelle du retriever
est 226 lignes : départ 227 refusé, fin trop grande bornée ; revoir le protocole si ce fichier change.
Les modes `memory`, `finals`, `autonomous`, `unsupported` ci-dessus reproduisent respectivement
continuité/compaction/reprise/isolation, finales fixées, choix autonomes et rejet d'une claim inventée.

Pour deux projets de même nom, créer leurs dossiers hors dépôt :

```powershell
$validationRoot = python -c "import tempfile; print(tempfile.mkdtemp(prefix='codeagent-m7-projects-'))"
$projectA = Join-Path $validationRoot 'a/same'
$projectB = Join-Path $validationRoot 'b/same'
New-Item -ItemType Directory -Force -Path $projectA,$projectB | Out-Null
python -m app.main --project $projectA --session m7-project-isolation --debug
python -m app.main --project $projectB --session m7-project-isolation --debug
python -m app.main --project $projectA --session m7-project-isolation --debug
```

Choisir 2. Dans A, enregistrer `PROJET-A-6842` et « aucune dépendance ajoutée », puis quitter.
Dans B, demander le rappel : absent. Dans le nouveau processus A : deux faits présents.
Les checkpoints doivent différer malgré les mêmes noms de projet/session.
Pour tester le suivi RAG, choisir **0** et poser `Où est définie read_dir ?`, puis `Que fait cette fonction ?` ;
sources actuelles et grounding restent requis. Les tests stdio de `tests/test_memory_integration.py`
reproduisent indépendamment du modèle une plage invalide puis une lecture valide.

### Conservation des preuves

Les fixtures, datasets et JSON du dépôt restent conservés, y compris les routages 1/3 et 2/3,
le holdout 39/40 et les deux corpus RAG avant/après. Les seuls résultats régénérés par la commande
mémoire par défaut sont dans `benchmarks/memory/results/m7.json` ; les chiffres de sa première campagne
sont conservés ci-dessus. Les sorties de revalidation supplémentaires restent hors dépôt.

Les transcriptions détaillées de la dernière campagne sources sont conservées, sans suppression, dans
`E:\AI\Codex\visualizations\2026\10\14\01a13ba8-ed1f-7023-b685-925cddea4da1` :
`memory.json`, `m6_comparison.json`, `real_memory.json`, `real_memory_after.json`, `real_memory_final.json`,
`finals.json`, `finals_after.json`, `autonomous.json`, `autonomous_after.json`, `autonomous_guard.json`,
`unsupported.json`, `citation_repair_real.json`, `citation_repair_real_after.json`, `routing.json`,
`routing_final.json`, `resources.json`, `resume_validation.json` et `rag/`.
La campagne antérieure reste dans
`E:\AI\Codex\visualizations\2026\10\14\01a13b6e-3ccd-7ae1-8a01-2139da70a6c3`.
Ces fichiers locaux ne feront pas partie du push ; le présent document conserve les mesures utiles,
les échecs, les limites et leurs commandes. Aucun succès historique n'est présenté comme nouvellement reproduit.
Les nouvelles sorties RAG et routage de cette finalisation sont dans
`E:\AI\Codex\visualizations\2026\10\14\01a13c4d-cb46-7350-b684-666f26f69d24` (`rag/`, `routing.json`).

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

M7 est terminé dans son périmètre mémoire et contexte, avec les limites autonomes et matérielles documentées ci-dessus. La suite prévue est M8 — Planning & Controlled Autonomous Execution,
puis M9 — Advanced RAG & Reliability. Le grounding spécialisé reste conditionnel à un gain mesuré.

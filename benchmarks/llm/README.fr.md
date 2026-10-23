[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

[README du projet](../../README.fr.md)

## Résultats du benchmark

Sept modèles locaux ont été évalués avec le même benchmark, le même runtime Ollama et les mêmes paramètres de génération.

| Modèle                  | Latence moyenne | Débit moyen    |
| ----------------------- | --------------: | -------------: |
| `qwen2.5-coder:7b`      |          3.22 s | 86.71 tokens/s |
| `qwen2.5-coder:14b`     |          4.13 s | 45.21 tokens/s |
| `deepseek-coder-v2:16b` |          5.85 s | 36.26 tokens/s |
| `gemma3:12b`            |          5.64 s | 18.42 tokens/s |
| `devstral:24b`          |         10.12 s | 10.54 tokens/s |
| `qwen3-coder:30b`       |         14.13 s | 12.23 tokens/s |
| `llama3.1:8b`           |          3.64 s | 70.95 tokens/s |

## Modèle sélectionné

Le modèle retenu pour CodeAgent est :

```text
qwen2.5-coder:14b
```

La sélection repose sur les performances mesurées par le benchmark ainsi que sur la validation manuelle des cas de test nécessaires au fonctionnement du pipeline RAG.

`qwen2.5-coder:7b` reste une alternative légère intéressante lorsque la vitesse et la consommation de ressources sont prioritaires.

Les modèles plus lourds testés n'ont pas apporté de gain suffisamment important pour justifier leur coût d'exécution dans le cadre actuel de CodeAgent.

<a id="memory-and-planning-protocols"></a>
## Protocoles mémoire et planning

Exécuter depuis la racine du dépôt ; les rapports générés vont dans un répertoire temporaire :

```powershell
$benchmarkOutput = Join-Path $env:TEMP ('codeagent-bench-' + [guid]::NewGuid().ToString())
python -m scripts.benchmark_memory --output (Join-Path $benchmarkOutput 'memory.json')
python -m scripts.benchmark_planning --execution both --output (Join-Path $benchmarkOutput 'planning.json')
python -m scripts.benchmark_planning --real --execution auto --cases-file benchmarks/planning/autonomy-pilot.json --case H-tests --debug --output (Join-Path $benchmarkOutput 'planning-real.json')
```

Sans `--real`, les décisions du modèle et le grounding sont scriptés ; les outils MCP travaillent sur les fichiers d'un projet temporaire. Ces contrôles vérifient les mécanismes d'exécution, les preuves et les budgets, pas la qualité du modèle. Modes planning : `off` utilise M6 directement, `auto` utilise le planning M8, `both` compare ces modes ; `adaptive` révise localement les étapes restantes, et `compare` compare `auto` à `adaptive`.

`--real` nécessite le service Ollama existant et les modèles installés ; il ne démarre pas le service. Les décisions Ollama et les outils MCP en processus sont réels, sur des fixtures temporaires, sans transport stdio. `--hitl on-failure` active une pause interactive après un blocage, sans forcer d'intervention. Les contrôles automatiques de succès nécessitent une revue factuelle distincte des actions, lectures complètes, réponse, sources et grounding.

Les JSON de `benchmarks/planning/` définissent des cas exécutables (`autonomy-pilot`, `autonomy-holdout`, `hitl-pilot`) ou des protocoles expérimentaux figés (`finalization-protocol`, `structured-protocol`). Les fichiers sous `results/` sont des rapports générés ou des revues historiques indépendantes, pas des configurations d'exécution. Le prototype natif JSON Schema a été retiré ; `--planning-format` n'existe pas dans le code livré. Voir les [contrats et preuves M8](../../docs/memory.fr.md#m8--planning--controlled-autonomous-execution) pour les décisions expérimentales, la validation stdio réelle et les conditions de reproduction.

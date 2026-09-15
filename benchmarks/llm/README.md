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
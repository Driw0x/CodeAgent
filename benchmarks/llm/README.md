## Résultats du benchmark

Sept modèles locaux ont été évalués avec le même benchmark, le même runtime Ollama et les mêmes paramètres de génération.

| Modèle                  | Latence moyenne |    Débit moyen | Observation                                                                      |
| ----------------------- | --------------: | -------------: | -------------------------------------------------------------------------------- |
| `qwen2.5-coder:7b`      |          3.22 s | 86.71 tokens/s | Très rapide et globalement précis, mais légèrement plus sujet aux extrapolations |
| `qwen2.5-coder:14b`     |          4.13 s | 45.21 tokens/s | Très bon équilibre entre qualité, respect du contexte et performances            |
| `deepseek-coder-v2:16b` |          5.85 s | 36.26 tokens/s | Correct, mais moins fiable sur certaines instructions et références              |
| `gemma3:12b`            |          5.64 s | 18.42 tokens/s | Réponses précises et concises, mais génération plus lente                        |
| `devstral:24b`          |         10.12 s | 10.54 tokens/s | Très bonne qualité mais coût d'exécution élevé                                   |
| `qwen3-coder:30b`       |         14.13 s | 12.23 tokens/s | Très bonne qualité, mais trop lent pour le gain observé                          |
| `llama3.1:8b`           |          3.64 s | 70.95 tokens/s | Rapide et correct, mais moins fiable sur la provenance exacte                    |

Les performances sont conservées séparément de l'évaluation qualitative afin de ne pas réduire la sélection à un score unique.

Plusieurs modèles produisent des réponses qualitativement satisfaisantes sur les cas simples. Les différences les plus importantes apparaissent sur le raisonnement multi-chunks, le respect strict du contexte et la conservation exacte des références de fichiers et de lignes.

## Modèle sélectionné

Le modèle retenu pour CodeAgent est :

```text
qwen2.5-coder:14b
```

Ce choix repose sur le meilleur compromis observé entre :

* compréhension et explication de code ;
* raisonnement sur plusieurs chunks ;
* respect du contexte fourni ;
* résistance aux hallucinations ;
* respect des instructions ;
* conservation exacte des références de fichiers et de lignes ;
* vitesse de génération suffisante pour une utilisation interactive.

`qwen2.5-coder:7b` reste une alternative légère intéressante lorsque la vitesse et la consommation de ressources sont prioritaires.

Les modèles plus lourds testés n'ont pas apporté de gain suffisamment important pour justifier leur coût d'exécution dans le cadre actuel de CodeAgent.
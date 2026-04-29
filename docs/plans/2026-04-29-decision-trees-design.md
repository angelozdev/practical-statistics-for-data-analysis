# Diseño — Notebook 24: Árboles de Decisión

## Objetivo

Enseñar cómo funciona un árbol de decisión (splits, impureza Gini, profundidad)
usando un dataset sintético de estudiantes que aprueba o reprueba un examen.

## Dataset

- ~120 estudiantes sintéticos
- Features: `horas_estudio` (0–10) y `horas_sueno` (3–10)
- Target: `aprueba` (1 = aprueba, 0 = reprueba)
- Frontera real: `aprueba` si `horas_estudio >= 4 AND horas_sueno >= 6` + ruido gaussiano
- Split 80/20 train/test con `random_state=42`

## Estructura del notebook

| Sección | Título | Contenido |
|---|---|---|
| 1 | Dataset + EDA | Generar dataset, scatter plot coloreado por `aprueba` |
| 2 | Anatomía del primer split | Calcular Gini a mano para un corte candidato, mostrar por qué el árbol elige ese umbral |
| 3 | Tres profundidades | `export_text` + `plot_tree` lado a lado para `max_depth` 1, 2, 3 |
| 4 | Train vs Test | Tabla accuracy train y test para los 3 modelos; identificar underfitting y overfitting |
| 5 | Nuevo estudiante | Predecir con 5h estudio / 7h sueño; trazar el camino de decisión en cada árbol |
| 6 | Reflexiones contraintuitivas | Los 4 puntos del enunciado explicados con ejemplos del notebook |

## Reflexiones a cubrir

1. Pruning acepta clasificar mal algunos datos de training a propósito
2. Igualdad de tamaños de grupo no garantiza pureza
3. El árbol descubre interacciones (estudio > 4 AND sueño > 6) sin que nadie las programe
4. Conteos absolutos engañan — el porcentaje es lo que importa

## Dependencias

- `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `pandas` (ya en pyproject.toml)
- No requiere dependencias nuevas

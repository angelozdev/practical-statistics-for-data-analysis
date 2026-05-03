# Diseño — Notebook 25: Random Forest

**Fecha:** 2026-05-03
**Notebook:** `notebooks/25-random-forest.ipynb`
**Prerrequisito:** Notebook 24 (Árboles de Decisión)

---

## Objetivo

Llevar al lector de cero a producción con Random Forest, partiendo del problema concreto que motiva el algoritmo (inestabilidad de un árbol solo) hasta un pipeline listo para usar en producción.

## Audiencia

Primera exposición al algoritmo. Nivel matemático moderado: una fórmula clave por concepto, explicada en lenguaje natural.

## Enfoque pedagógico

**Problema primero.** Cada concepto (Bagging, mtry, B, Variable Importance) se introduce como solución a un problema demostrado en código. El lector ve el problema antes de aprender la solución.

---

## Dataset

**400 estudiantes sintéticos** — extensión del dataset del notebook 24.

| Variable | Descripción | Rango |
|---|---|---|
| `horas_estudio` | Horas de estudio el día anterior | 0–10 h |
| `horas_sueno` | Horas de sueño la noche anterior | 3–10 h |
| `asistencia` | % de clases asistidas | 50–100 % |
| `tareas_entregadas` | Proporción de tareas entregadas | 0–1 |
| `nota_parcial` | Nota del examen parcial | 0–10 |
| `aprueba` | Target: ¿aprobó el examen final? | 0 / 1 |

Regla generadora: combinación ponderada de las 5 variables con ruido 15%.

---

## Secciones

### 0 · Dataset — 400 estudiantes extendido
- Generar dataset sintético con 5 features
- Mostrar distribución del target
- Scatter matrix o pairplot rápido

### 1 · El problema — un árbol solo es inestable
- Entrenar 3 `DecisionTreeClassifier` con distintas semillas
- Mostrar que dan predicciones diferentes para el mismo estudiante
- Mensaje: fragilidad del árbol solo

### 2 · Bagging — "preguntarle a muchos"
- Explicar bootstrap con analogía visual
- Entrenar B=10 árboles sobre muestras bootstrap distintas
- Votar por mayoría — mostrar estabilización
- Fórmula: `ŷ = moda(árbol₁(x), ..., árbolB(x))`

### 3 · mtry — "no dejes que un árbol lo vea todo"
- Explicar `max_features` (mtry): m variables aleatorias por split
- Sin mtry → árboles correlacionados → votan igual → no sirve el ensemble
- Graficar accuracy vs mtry = 1, 2, 3, 4, 5

### 4 · B — ¿cuántos árboles necesito?
- Graficar accuracy en test vs B = 1 a 200
- Mostrar estabilización del error
- Mensaje: más árboles no hace overfitting, pero hay retorno decreciente

### 5 · Variable Importance — ¿qué variable importó más?
- Extraer `feature_importances_` del modelo entrenado
- Gráfico de barras horizontal ordenado
- Reflexión: ¿es `nota_parcial` la más importante?

### 6 · RF vs Árbol solo — comparación directa
- `RandomForestClassifier` vs `DecisionTreeClassifier` mismo train/test split
- Tabla: accuracy train, accuracy test, gap
- Mensaje: el bosque promedia el ruido del árbol individual

### 7 · Producción — Pipeline listo
- `Pipeline(StandardScaler + RandomForestClassifier)`
- `GridSearchCV` sobre `n_estimators` y `max_features`
- `StratifiedKFold` con scoring F1
- Mostrar mejores parámetros y resultado final

### 8 · Lo que sorprende del Random Forest
- 4 insights contraintuitivos:
  1. Más árboles nunca produce overfitting (a diferencia de max_depth)
  2. mtry pequeño puede superar a mtry grande
  3. Variable Importance puede mentir con features correlacionadas
  4. El error OOB es gratis — no necesitas separar un test set adicional

---

## Dependencias

```
scikit-learn>=1.3
numpy
pandas
matplotlib
seaborn
```

Sin dependencias nuevas (mismo entorno que notebooks anteriores).

---

## Criterios de éxito

- El lector puede explicar Bagging, mtry y B sin ver el código
- La comparación RF vs árbol único muestra mejora clara en accuracy test
- El pipeline de producción es copy-paste ready

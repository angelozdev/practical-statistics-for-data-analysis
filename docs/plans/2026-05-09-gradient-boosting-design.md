# Diseño — Notebook 26: Gradient Boosting

**Fecha:** 2026-05-09
**Notebook:** `notebooks/26-gradient-boosting.ipynb`
**Prerrequisito:** Notebook 25 (Random Forest)

---

## Objetivo

Demostrar prácticamente por qué Gradient Boosting es más potente que Random Forest cuando hay interacciones sutiles y ruido moderado, y enseñar las tres lecciones centrales del algoritmo: bias reduction secuencial, captura de interacciones, y el trade-off learning rate ↔ n_estimators.

## Audiencia

Lector con base de Random Forest (notebook 25). Nivel intermedio-avanzado. Quiere entender el "por qué" mecánico, no solo el "cómo".

## Enfoque pedagógico

**Problema-primero**, consistente con notebooks 23–25. Cada concepto se introduce como respuesta a un fallo demostrado en código del modelo anterior. RF actúa como baseline; GB aparece cuando RF se queda corto.

---

## Dataset

**800 solicitantes sintéticos** — escenario "riesgo crediticio simulado" diseñado específicamente para que GB destaque.

| Variable | Tipo | Rango |
|---|---|---|
| `ingreso_mensual` | continuo | 800 – 8000 |
| `deuda_actual` | continuo | 0 – 12000 |
| `score_historial` | continuo | 0 – 100 |
| `antiguedad_laboral_meses` | continuo | 0 – 240 |
| `num_consultas_recientes` | entero | 0 – 15 |
| `tipo_empleo` | categórico | `fijo` / `temporal` / `independiente` |
| `default` | target binario | 0 / 1 |

**Regla generadora (3 capas):**

1. **Lineal débil:** `score_historial` bajo + ratio `deuda/ingreso` alto → riesgo sube.
2. **Interacción 1 (oculta):** `num_consultas_recientes` alto **solo importa si** `deuda/ingreso > 0.5`. Pánico de crédito.
3. **Interacción 2 (oculta):** `antiguedad` protege **solo si** `tipo_empleo == "fijo"`. Si es temporal o independiente, antigüedad no salva.
4. **Ruido:** flip 15% de etiquetas.

**Por qué GB ganará:**

- RF capta la capa 1 fácil. Las capas 2-3 las captura parcialmente porque cada árbol bootstrap puede no ver la interacción.
- GB las captura porque cada árbol nuevo se entrena sobre los **residuales** de los anteriores → ataca exactamente lo que el ensemble previo falló.
- Esperable: RF accuracy ~0.78–0.82, GB ~0.84–0.88. Gap visible y explicable.

---

## Secciones

### 0 · Dataset — riesgo crediticio sintético
- Generar 800 muestras con las 3 capas
- Distribución del target, scatter rápido
- One-hot de `tipo_empleo`

### 1 · El baseline — Random Forest
- `RandomForestClassifier` defaults sobre el dataset
- Reportar accuracy train/test, matriz de confusión
- Mensaje: "no está mal, pero ¿podemos hacerlo mejor?"

### 2 · El problema — RF promedia, no afina
- Mostrar errores donde RF falla: casos con `num_consultas` alto + ratio deuda/ingreso > 0.5
- Histograma de probabilidades RF: muchas en zona gris (0.4–0.6)
- Mensaje: RF promedia árboles independientes → no corrige sesgo sistemático

### 3 · La idea de Gradient Boosting — corrección secuencial
- Analogía: "cada árbol nuevo aprende los errores del anterior"
- Fórmula clave: `Fₘ(x) = Fₘ₋₁(x) + ν · hₘ(x)` donde `hₘ` se ajusta al residual
- Mini-demo (3 cells): apilar 3 `DecisionTreeRegressor` shallow a mano sobre residuales → mostrar curva de error bajando

### 4 · GradientBoostingClassifier — defaults
- `GradientBoostingClassifier()` sin tunear
- Accuracy train/test, matriz de confusión
- **Comparación directa con RF** (tabla): accuracy, gap, errores en casos difíciles
- Cumple el bullet del usuario: "Comparar accuracy de los dos. ¿Cuál gana? ¿Por cuánto?"

### 5 · Feature importances — ¿qué dominó?
- `feature_importances_` de GB y RF lado a lado
- Gráfico de barras horizontales comparativo
- Reflexión: ¿coinciden? ¿GB le da más peso a las features de interacción?
- Cumple el bullet: "Extraer feature_importances_ del GB. ¿Cuál variable dominó?"

### 6 · Lección 1 visual — bias reduction secuencial
- Plot: `staged_predict` → accuracy test vs número de árboles, m=1..200
- Misma curva para RF (constante a partir de B chico)
- Mensaje visual: GB sigue mejorando, RF estancado

### 7 · Lección 2 — interacciones que RF pierde
- Partial dependence 2D para `num_consultas × ratio_deuda` en RF y GB
- Mostrar que GB captura la región "pánico" más nítida
- Cubre la lección "interacciones sutiles"

### 8 · Lección 3 — tuning learning_rate y n_estimators
- Heatmap accuracy test sobre grid: `lr ∈ {0.01, 0.05, 0.1, 0.3}` × `n_estimators ∈ {50, 100, 200, 500}`
- **Comparación pedida:** lr=0.01 + 500 árboles vs lr=0.1 + 100. Tabla con accuracy, tiempo, overfitting gap
- Mensaje: lr bajo + más árboles = mejor pero más lento. Trade-off explícito
- Cumple el bullet: "Tunear learning_rate y n_estimators del GB. ¿Mejora o empeora con lr=0.01 y 500 árboles vs lr=0.1 y 100?"

### 9 · XGBoost — el estándar de la industria
- `XGBClassifier` defaults sobre mismo dataset
- Comparación final 3-way: RF / GB sklearn / XGBoost
- Mencionar regularización L1/L2, early stopping, paralelización
- Mensaje: GB sklearn es el concepto, XGBoost la versión productiva

### 10 · Lo que sorprende del Gradient Boosting
- 4 insights contraintuitivos:
  1. Más árboles **sí** puede causar overfitting (a diferencia de RF)
  2. `learning_rate` y `n_estimators` son inversos: bajar uno = subir el otro
  3. GB sin tuning a veces pierde con RF — el poder está en el tuning
  4. Interpretabilidad: `staged_predict` permite ver cómo el modelo aprende paso a paso

---

## Dependencias

```
scikit-learn>=1.8        (ya instalado)
numpy, pandas            (ya instalado)
matplotlib, seaborn      (ya instalado)
xgboost                  (NUEVO — uv add xgboost)
```

---

## Estilo

- Markdown intercalado: cada sección abre con párrafo "qué/por qué", cierra con mensaje clave
- Plots con `seaborn` + `matplotlib`, paleta consistente
- Semilla fija `RANDOM_STATE = 42` para reproducibilidad
- Tablas comparativas con `pd.DataFrame` + `.style.background_gradient` cuando aplique

---

## Cobertura del checklist del usuario

| Bullet pedido | Sección |
|---|---|
| Dataset (sintético, diseñado para GB) | 0 |
| GBC con defaults | 4 |
| RF sobre mismos datos | 1 |
| Comparar accuracy, quién gana y por cuánto | 4 (tabla directa) |
| `feature_importances_` del GB, qué variable dominó | 5 |
| Tunear `learning_rate` y `n_estimators`, lr=0.01/500 vs lr=0.1/100 | 8 |

---

## Criterios de éxito

- El lector puede explicar por qué GB > RF en este dataset sin ver el código
- Las tres lecciones (bias reduction, interacciones, tuning) tienen cada una un plot que las ancla visualmente
- El notebook corre end-to-end en menos de 2 minutos
- La diferencia GB vs RF en accuracy es ≥ 4 puntos (si no, el dataset no demuestra nada)

---

## Riesgos a verificar al ejecutar

- Que el dataset realmente produzca gap GB > RF. Si no, ajustar fuerza de las interacciones o reducir ruido.
- Que `staged_predict` muestre la curva esperada (GB sigue bajando, RF plana).
- Que partial dependence 2D no sea visualmente confuso — si lo es, fallback a heatmap manual.

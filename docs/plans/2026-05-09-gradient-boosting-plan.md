# Notebook 26 — Gradient Boosting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **IMPORTANT — User preference (memory: feedback_no_commits_or_branches):** No git commits and no branch changes during execution. Skip every commit step in this plan unless the user explicitly asks. Save and run cells; user handles git.

**Goal:** Construir un notebook didáctico end-to-end que demuestre por qué Gradient Boosting supera a Random Forest en datasets con interacciones sutiles, cubriendo bias reduction secuencial, captura de interacciones, y trade-off learning_rate ↔ n_estimators.

**Architecture:** Notebook Jupyter con 11 secciones (0–10) siguiendo patrón "problema-primero" de notebooks 23–25. Dataset sintético de 800 muestras diseñado con 3 capas (lineal + 2 interacciones ocultas + ruido 15%) para que GB gane a RF por ≥4 puntos de accuracy. Una task por sección, cada task = ejecutar celda(s) y verificar output esperado.

**Tech Stack:** scikit-learn (GradientBoostingClassifier, RandomForestClassifier, partial_dependence), xgboost (NUEVA dep), numpy, pandas, matplotlib, seaborn, Jupyter.

**Reference:** `docs/plans/2026-05-09-gradient-boosting-design.md`

---

## Task 0: Setup — dependencias y notebook vacío

**Files:**
- Modify: `pyproject.toml` (vía `uv add xgboost`, no editar a mano)
- Create: `notebooks/26-gradient-boosting.ipynb` (vacío con kernel Python 3)

**Step 1: Instalar xgboost**

```bash
uv add xgboost
```
Expected: `pyproject.toml` actualizado con `"xgboost>=X.X"`, `uv.lock` regenerado.

**Step 2: Crear notebook vacío**

Crear archivo `notebooks/26-gradient-boosting.ipynb` con un solo cell markdown de título:

```markdown
# Notebook 26 — Gradient Boosting

Por qué GB supera a Random Forest cuando hay interacciones sutiles.
```

**Step 3: Smoke test del kernel**

Agregar cell de código al final con:
```python
import sklearn, xgboost, numpy, pandas, matplotlib, seaborn
print("OK")
```
Ejecutar. Expected: `OK` sin error.

**Step 4: Imports + setup global**

Reemplazar el smoke cell por el setup canónico:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
```
Ejecutar. Expected: sin error.

---

## Task 1: Sección 0 — Dataset sintético

**Files:**
- Modify: `notebooks/26-gradient-boosting.ipynb` (agregar sección 0)

**Step 1: Markdown introductorio**

```markdown
## 0 · Dataset — riesgo crediticio sintético

Construimos 800 solicitantes con 3 capas: una señal lineal débil y dos interacciones ocultas. Las interacciones son la trampa que RF subajusta y GB capta.
```

**Step 2: Función generadora**

```python
def generar_dataset(n=800, ruido=0.15, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    ingreso = rng.uniform(800, 8000, n)
    deuda = rng.uniform(0, 12000, n)
    score = rng.uniform(0, 100, n)
    antiguedad = rng.uniform(0, 240, n)
    consultas = rng.integers(0, 16, n)
    empleo = rng.choice(["fijo", "temporal", "independiente"], n, p=[0.5, 0.3, 0.2])

    ratio = deuda / ingreso

    # Capa 1 — lineal débil
    riesgo = (100 - score) / 100 + ratio * 0.5

    # Capa 2 — interacción consultas × ratio
    riesgo += np.where(ratio > 0.5, consultas * 0.08, 0)

    # Capa 3 — antiguedad protege solo si fijo
    proteccion = np.where(empleo == "fijo", antiguedad / 240, 0)
    riesgo -= proteccion * 0.6

    # Probabilidad de default
    prob = 1 / (1 + np.exp(-(riesgo - riesgo.mean())))
    y = (rng.uniform(0, 1, n) < prob).astype(int)

    # Ruido — flip 15%
    flip = rng.uniform(0, 1, n) < ruido
    y = np.where(flip, 1 - y, y)

    df = pd.DataFrame({
        "ingreso_mensual": ingreso,
        "deuda_actual": deuda,
        "score_historial": score,
        "antiguedad_laboral_meses": antiguedad,
        "num_consultas_recientes": consultas,
        "tipo_empleo": empleo,
        "default": y,
    })
    return df

df = generar_dataset()
df.head()
```
Ejecutar. Expected: tabla con 5 filas, 7 columnas.

**Step 3: Verificar balance del target**

```python
df["default"].value_counts(normalize=True)
```
Expected: ratio entre 0.30 y 0.55 para clase 1 (no extremo).

**Step 4: One-hot del categórico + split train/test**

```python
X = pd.get_dummies(df.drop(columns=["default"]), columns=["tipo_empleo"], drop_first=True)
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(X_train.shape, X_test.shape)
```
Expected: `(600, 8) (200, 8)`.

---

## Task 2: Sección 1 — Random Forest baseline

**Step 1: Markdown**

```markdown
## 1 · El baseline — Random Forest

Empezamos con RF defaults. Es nuestro punto de comparación. Si GB no le gana, no vale la pena.
```

**Step 2: Entrenar RF**

```python
rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

acc_train_rf = accuracy_score(y_train, rf.predict(X_train))
acc_test_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"RF train: {acc_train_rf:.3f}  test: {acc_test_rf:.3f}")
```
Expected: train ~0.99, test entre 0.75 y 0.85.

**Step 3: Matriz de confusión**

```python
cm_rf = confusion_matrix(y_test, rf.predict(X_test))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no default", "default"],
            yticklabels=["no default", "default"])
plt.title("Random Forest — matriz de confusión (test)")
plt.show()
```
Expected: heatmap 2×2 con números legibles.

**Step 4: Cierre con mensaje**

```markdown
RF tiene un gap notable train-test. ¿Es lo mejor que podemos hacer?
```

---

## Task 3: Sección 2 — El problema de RF

**Step 1: Markdown**

```markdown
## 2 · El problema — RF promedia, no afina

RF promedia árboles independientes. Eso reduce varianza pero no corrige errores sistemáticos. Veamos en qué casos falla.
```

**Step 2: Aislar errores en zona de interacción**

```python
proba_rf = rf.predict_proba(X_test)[:, 1]
errores = (rf.predict(X_test) != y_test)
ratio_test = X_test["deuda_actual"] / X_test["ingreso_mensual"]
zona_riesgo = (ratio_test > 0.5) & (X_test["num_consultas_recientes"] > 7)

print(f"Error rate global:        {errores.mean():.3f}")
print(f"Error rate en zona panico: {errores[zona_riesgo].mean():.3f}")
```
Expected: error en zona pánico **mayor** que el global (>5 puntos de diferencia).

**Step 3: Histograma de probabilidades**

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(proba_rf, bins=30, edgecolor="black")
ax.axvspan(0.4, 0.6, color="orange", alpha=0.2, label="zona gris")
ax.set_xlabel("P(default) según RF")
ax.set_ylabel("frecuencia")
ax.set_title("RF deja muchas predicciones en zona gris")
ax.legend()
plt.show()
```
Expected: barra alta en zona 0.4–0.6.

---

## Task 4: Sección 3 — Idea de GB + demo manual 3 árboles

**Step 1: Markdown teórico**

```markdown
## 3 · La idea — corrección secuencial

Cada árbol nuevo aprende los errores (residuales) del ensemble previo:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

donde `h_m` se ajusta al residual `y - F_{m-1}(x)` y `ν` (learning_rate) controla cuánto contribuye cada árbol.

Vamos a hacerlo a mano con 3 `DecisionTreeRegressor` shallow.
```

**Step 2: Demo manual — 3 árboles apilados sobre residuales**

```python
y_train_arr = y_train.to_numpy(dtype=float)
F = np.full_like(y_train_arr, y_train_arr.mean())  # F_0 = media
lr_demo = 0.3
errores_paso = [np.mean((y_train_arr - F) ** 2)]

for paso in range(3):
    residual = y_train_arr - F
    arbol = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    arbol.fit(X_train, residual)
    F = F + lr_demo * arbol.predict(X_train)
    errores_paso.append(np.mean((y_train_arr - F) ** 2))

print("MSE por paso:", [round(e, 4) for e in errores_paso])
```
Expected: lista de 4 MSE, cada uno **menor** que el anterior.

**Step 3: Plot del descenso**

```python
plt.plot(range(len(errores_paso)), errores_paso, "o-")
plt.xlabel("árbol m")
plt.ylabel("MSE entrenamiento")
plt.title("Cada árbol nuevo reduce el error sobre los residuales")
plt.show()
```
Expected: curva descendente.

---

## Task 5: Sección 4 — GBC defaults vs RF (comparación directa)

**Step 1: Markdown**

```markdown
## 4 · GradientBoostingClassifier — defaults

Sin tunear. Comparamos contra RF en el mismo split.
```

**Step 2: Entrenar GBC defaults**

```python
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb.fit(X_train, y_train)

acc_train_gb = accuracy_score(y_train, gb.predict(X_train))
acc_test_gb = accuracy_score(y_test, gb.predict(X_test))
print(f"GB train: {acc_train_gb:.3f}  test: {acc_test_gb:.3f}")
```
Expected: test ≥ 0.82.

**Step 3: Tabla comparativa**

```python
tabla = pd.DataFrame({
    "modelo": ["Random Forest", "Gradient Boosting"],
    "accuracy_train": [acc_train_rf, acc_train_gb],
    "accuracy_test": [acc_test_rf, acc_test_gb],
    "gap": [acc_train_rf - acc_test_rf, acc_train_gb - acc_test_gb],
})
tabla
```
Expected: GB con `accuracy_test` ≥ RF + 0.04. **Si no se cumple, abrir checkpoint con el usuario antes de seguir** (ajustar dataset: subir n a 1200, o reducir ruido a 0.10, o subir interacción coef de 0.08 a 0.12).

**Step 4: Errores en zona pánico — GB vs RF**

```python
errores_gb = (gb.predict(X_test) != y_test)
print(f"RF en zona panico: {errores[zona_riesgo].mean():.3f}")
print(f"GB en zona panico: {errores_gb[zona_riesgo].mean():.3f}")
```
Expected: GB con error menor que RF en zona pánico.

---

## Task 6: Sección 5 — Feature importances comparadas

**Step 1: Markdown**

```markdown
## 5 · Feature importances — ¿qué dominó?

Ambos modelos exponen `feature_importances_`. ¿Coinciden? ¿GB pondera distinto las features que entran en interacción?
```

**Step 2: Tabla comparativa**

```python
imp = pd.DataFrame({
    "feature": X_train.columns,
    "RF": rf.feature_importances_,
    "GB": gb.feature_importances_,
}).sort_values("GB", ascending=True)
imp
```
Expected: tabla con dos columnas numéricas.

**Step 3: Plot horizontal comparativo**

```python
fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(imp))
ax.barh(y_pos - 0.2, imp["RF"], height=0.4, label="RF")
ax.barh(y_pos + 0.2, imp["GB"], height=0.4, label="GB")
ax.set_yticks(y_pos)
ax.set_yticklabels(imp["feature"])
ax.set_xlabel("importancia")
ax.set_title("Feature importances — RF vs GB")
ax.legend()
plt.tight_layout()
plt.show()
```
Expected: gráfico con dos barras por feature.

**Step 4: Reflexión markdown**

```markdown
Si GB le da más peso a `num_consultas_recientes` o `ratio_deuda_ingreso`, está reconociendo la interacción que RF reparte entre features.
```

---

## Task 7: Sección 6 — Bias reduction visual con `staged_predict`

**Step 1: Markdown**

```markdown
## 6 · Lección 1 — bias reduction secuencial

`staged_predict` deja ver el accuracy del ensemble en cada paso m=1..n_estimators. RF no lo permite porque sus árboles son independientes; lo simulamos refitting con distintos B.
```

**Step 2: Curva GB con `staged_predict`**

```python
acc_gb_por_m = [accuracy_score(y_test, pred) for pred in gb.staged_predict(X_test)]
```
Expected: lista de 100 floats.

**Step 3: Curva RF con n_estimators creciente**

```python
acc_rf_por_b = []
for b in [1, 5, 10, 25, 50, 75, 100]:
    rf_tmp = RandomForestClassifier(n_estimators=b, random_state=RANDOM_STATE)
    rf_tmp.fit(X_train, y_train)
    acc_rf_por_b.append((b, accuracy_score(y_test, rf_tmp.predict(X_test))))
acc_rf_por_b
```
Expected: lista de tuplas `(b, acc)`.

**Step 4: Plot conjunto**

```python
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(acc_gb_por_m) + 1), acc_gb_por_m, label="GB (staged_predict)")
b_rf, acc_rf_vals = zip(*acc_rf_por_b)
ax.plot(b_rf, acc_rf_vals, "o-", label="RF (refit con B creciente)")
ax.set_xlabel("número de árboles")
ax.set_ylabel("accuracy test")
ax.set_title("GB sigue afinando, RF se estanca")
ax.legend()
plt.show()
```
Expected: línea GB ascendente sobre línea RF plana.

---

## Task 8: Sección 7 — Interacciones que RF pierde (partial dependence)

**Step 1: Markdown**

```markdown
## 7 · Lección 2 — interacciones que RF pierde

Partial dependence 2D para `num_consultas_recientes × deuda_actual`. La región "pánico" (alta consulta + alta deuda) debería verse más nítida en GB.
```

**Step 2: PDP 2D para ambos modelos**

```python
features_pd = [("num_consultas_recientes", "deuda_actual")]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
PartialDependenceDisplay.from_estimator(
    rf, X_train, features=features_pd, ax=axes[0],
)
axes[0].set_title("Random Forest")
PartialDependenceDisplay.from_estimator(
    gb, X_train, features=features_pd, ax=axes[1],
)
axes[1].set_title("Gradient Boosting")
plt.tight_layout()
plt.show()
```
Expected: dos contour plots; GB con gradiente más marcado en la esquina derecha-arriba.

**Step 3: Si visual confuso → fallback heatmap manual**

Solo si el PDP de sklearn no se ve claro, agregar este cell extra:

```python
def heatmap_pd(model, n_grid=20):
    cons_grid = np.linspace(0, 15, n_grid)
    deuda_grid = np.linspace(0, 12000, n_grid)
    Z = np.zeros((n_grid, n_grid))
    base = X_train.median(numeric_only=True)
    base_full = X_train.iloc[[0]].copy()
    for col in base_full.columns:
        if col in base:
            base_full[col] = base[col]
    for i, c in enumerate(cons_grid):
        for j, d in enumerate(deuda_grid):
            row = base_full.copy()
            row["num_consultas_recientes"] = c
            row["deuda_actual"] = d
            Z[i, j] = model.predict_proba(row)[0, 1]
    return cons_grid, deuda_grid, Z
```
(Solo plotear si el PDP nativo falla.)

---

## Task 9: Sección 8 — Tuning lr × n_estimators

**Step 1: Markdown**

```markdown
## 8 · Lección 3 — tuning learning_rate y n_estimators

Trade-off central de GB. lr bajo necesita más árboles; lr alto puede overfittear. Comparemos lr=0.01/500 vs lr=0.1/100.
```

**Step 2: Comparación directa pedida por el usuario**

```python
import time

configs = [
    {"learning_rate": 0.01, "n_estimators": 500},
    {"learning_rate": 0.1, "n_estimators": 100},
]

filas = []
for cfg in configs:
    t0 = time.perf_counter()
    m = GradientBoostingClassifier(**cfg, random_state=RANDOM_STATE)
    m.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    filas.append({
        "config": f"lr={cfg['learning_rate']}, n={cfg['n_estimators']}",
        "acc_train": accuracy_score(y_train, m.predict(X_train)),
        "acc_test": accuracy_score(y_test, m.predict(X_test)),
        "tiempo_seg": round(elapsed, 2),
    })
pd.DataFrame(filas)
```
Expected: tabla 2 filas con `acc_test` y tiempo. Mensaje en markdown sobre cuál ganó y por qué.

**Step 3: Heatmap del grid completo**

```python
lrs = [0.01, 0.05, 0.1, 0.3]
ns = [50, 100, 200, 500]
grid = np.zeros((len(lrs), len(ns)))

for i, lr in enumerate(lrs):
    for j, n in enumerate(ns):
        m = GradientBoostingClassifier(
            learning_rate=lr, n_estimators=n, random_state=RANDOM_STATE
        )
        m.fit(X_train, y_train)
        grid[i, j] = accuracy_score(y_test, m.predict(X_test))

sns.heatmap(grid, annot=True, fmt=".3f",
            xticklabels=ns, yticklabels=lrs, cmap="viridis")
plt.xlabel("n_estimators")
plt.ylabel("learning_rate")
plt.title("Accuracy test — grid lr × n_estimators")
plt.show()
```
Expected: heatmap 4×4 con colores graduados.

**Step 4: Markdown de cierre**

```markdown
Patrón esperable: la diagonal lr×n ≈ constante mantiene accuracy similar. lr=0.3 con n alto puede empezar a overfittear.
```

---

## Task 10: Sección 9 — XGBoost

**Step 1: Markdown**

```markdown
## 9 · XGBoost — el estándar de la industria

GB de sklearn es excelente para entender el concepto. XGBoost añade regularización L1/L2, early stopping, paralelización por features, y trees `gbtree`/`gblinear`. Es lo que verás en producción y en competencias.
```

**Step 2: Entrenar XGBoost defaults**

```python
xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1,
)
xgb.fit(X_train, y_train)

acc_test_xgb = accuracy_score(y_test, xgb.predict(X_test))
print(f"XGBoost test: {acc_test_xgb:.3f}")
```
Expected: ≥ acc_test_gb (suele empatar o superar levemente).

**Step 3: Tabla 3-way final**

```python
final = pd.DataFrame({
    "modelo": ["RF", "GB sklearn", "XGBoost"],
    "accuracy_test": [acc_test_rf, acc_test_gb, acc_test_xgb],
})
final
```
Expected: XGBoost ≥ GB ≥ RF (idealmente).

**Step 4: Markdown sobre ventajas**

```markdown
XGBoost añade:
- **Regularización L1/L2** sobre los pesos hoja → menos overfit que GB sklearn.
- **Early stopping** con validación interna.
- **Paralelización** por features (no por árboles, dado que es secuencial).
- **Manejo nativo de NaN**.
```

---

## Task 11: Sección 10 — Insights finales

**Step 1: Markdown**

```markdown
## 10 · Lo que sorprende del Gradient Boosting

1. **Más árboles SÍ puede causar overfitting** — al revés que RF. Con `learning_rate` alto, agregar árboles eventualmente memoriza ruido.
2. **`learning_rate` y `n_estimators` son inversos** — bajar uno requiere subir el otro para mantener capacidad. Es un solo hiperparámetro disfrazado de dos.
3. **GB sin tuning a veces pierde con RF** — el poder de GB está en el tuning. RF es más perdonador con defaults.
4. **`staged_predict` es interpretabilidad gratis** — puedes ver cómo el modelo aprende paso a paso, qué errores corrige primero. RF no permite esto (sus árboles son independientes).
```

---

## Task 12: Verificación final end-to-end

**Step 1: Restart kernel & run all**

En Jupyter: Kernel → Restart & Run All. Cronometrar.

**Step 2: Verificar criterios de éxito**

| Criterio | Verificación |
|---|---|
| GB > RF en accuracy test ≥ 4 puntos | Tabla sección 4 |
| Notebook corre end-to-end < 2 minutos | Cronómetro |
| Las 3 lecciones tienen plot anclado | Secciones 6, 7, 8 |
| Sin errores ni warnings críticos | Output limpio |

**Step 3: Si algún criterio falla**

- Gap GB vs RF < 4 puntos: subir `n=1200` en Task 1 o aumentar coef interacción de 0.08 a 0.12.
- Tiempo > 2 min: reducir grid en Task 9 a 3×3 (`lrs=[0.01,0.1,0.3]`, `ns=[50,200,500]`).
- Plot PDP confuso: usar el fallback heatmap manual de Task 8 Step 3.

**Step 4: NO hacer commit**

Avisar al usuario que el notebook está listo. Mostrar `git status`. Dejar a su criterio commitear.

---

## Resumen de tasks

| # | Sección | Output clave |
|---|---|---|
| 0 | Setup | xgboost instalado, notebook vacío |
| 1 | 0 — Dataset | df 800×7, split 600/200 |
| 2 | 1 — RF baseline | acc_test_rf ~0.78–0.82 |
| 3 | 2 — Problema RF | error mayor en zona pánico |
| 4 | 3 — Idea GB + demo | MSE descendente con 3 árboles |
| 5 | 4 — GBC vs RF | tabla con gap ≥ 4 pts |
| 6 | 5 — Feature importances | barras comparativas |
| 7 | 6 — Bias reduction | GB asciende, RF se estanca |
| 8 | 7 — Interacciones | PDP 2D más nítido en GB |
| 9 | 8 — Tuning | tabla lr=0.01/500 vs lr=0.1/100 + heatmap |
| 10 | 9 — XGBoost | tabla 3-way final |
| 11 | 10 — Insights | 4 sorpresas |
| 12 | Verify | restart & run all, cumplir criterios |

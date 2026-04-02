# Design: Notebook 18 — Naive Bayes

## Context

Notebook 18 introduces classification (chapter 5 of "Practical Statistics for Data Science"). First topic: Naive Bayes. The user wants a practical, intuitive approach using a fun scenario: predicting whether a friend will come out or cancel based on categorical predictors.

## Dataset: `data/friends.csv`

~40 rows of historical invitations. All columns are binary (Sí/No):

- `con_novia` — is the friend with his girlfriend?
- `llueve` — is it raining?
- `viernes` — is it Friday?
- `sale` (target) — did the friend come out?

Biases: con_novia strongly reduces probability of coming out, viernes increases it, lluvia reduces it slightly.

## Approach

Manual calculation first (Bayes theorem step by step), then verify with sklearn's `CategoricalNB`.

## Notebook Structure

1. **El problema** — intro: "Mi amigo nos cancela mucho, puedo predecir cuando?"
2. **Datos + exploración** — load CSV, frequency tables per predictor vs `sale`, bar charts
3. **Bayes Exacto** — filter rows matching exact scenario, count outcomes. Show limitation: some combinations have zero matches
4. **Teorema de Bayes** — formula P(Y|X) = P(X|Y) * P(Y) / P(X), explain each term with the friend example
5. **Lo "Naive"** — independence assumption: P(X1,X2,X3|Y) = P(X1|Y) * P(X2|Y) * P(X3|Y). Full manual calculation for one scenario (con_novia=Sí, llueve=No, viernes=Sí)
6. **Predicción manual** — compute P(sale=Sí|X) vs P(sale=No|X), normalize, compare
7. **Verificación con sklearn** — CategoricalNB, confirm matches manual calculation
8. **Todas las combinaciones** — table of all 8 possible scenarios with predicted probabilities
9. **Resumen** — key concepts table

## Conventions

- Sections: `## N · Título`
- Colors: steelblue (normal), coral (highlight)
- seaborn for visualizations
- Spanish with English technical terms in parentheses

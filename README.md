---
title: Sistema Predictivo de Riesgo Academico
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app/app.py
pinned: false
---

# Prototipo funcional de alerta temprana academica

Aplicacion en Streamlit para identificar estudiantes en riesgo de bajo rendimiento en programacion a partir del desempeno en matematicas.

## Estructura del proyecto

```text
proyecto_ml/
|
|- data/
|  |- datos.csv
|- models/
|  |- .gitkeep
|- app/
|  |- app.py
|- src/
|  |- preprocessing.py
|  |- training.py
|  |- evaluation.py
|- requirements.txt
|- README.md
```

## Requisitos

- Python 3.10+

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run app/app.py
```

Tambien puedes usar:

```bash
streamlit run app.py
```

## Formato de datos requerido

Columnas obligatorias:

- `id_estudiante`
- `fundamentos_matematicas`
- `algebra_lineal`
- `calculo_diferencial`
- `calculo_integral`
- `taller_programacion`
- `desarrollo_software_1`
- `desarrollo_software_2`

## Definicion de la variable objetivo

```text
promedio_programacion =
  (taller_programacion + desarrollo_software_1 + desarrollo_software_2) / 3

riesgo_programacion =
  1 si promedio_programacion < 3.0
  0 si promedio_programacion >= 3.0
```

`promedio_programacion` solo se usa para construir la etiqueta objetivo y no entra como feature de entrenamiento.

## Funcionalidades implementadas

- Limpieza de datos (nulos, conversion numerica, validacion de rango 0-5)
- EDA completo (descriptiva, histogramas, boxplots, heatmap, distribucion de riesgo)
- Analisis de Pearson con interpretacion cualitativa
- Entrenamiento Random Forest + Train/Test + K-Fold (5)
- Ajuste de umbral para priorizar Recall
- Metricas: Accuracy, Precision, Recall, F1, Confusion Matrix
- Persistencia del modelo con Joblib (`models/modelo_riesgo.pkl`)
- Modo inferencia con formulario de notas matematicas
- Salida de clase y probabilidad de riesgo

## Nota metodologica

La metrica principal del sistema es **Recall**, ya que el objetivo institucional es detectar la mayor cantidad posible de estudiantes en riesgo, incluso aceptando algunos falsos positivos.

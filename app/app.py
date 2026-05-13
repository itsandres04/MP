from __future__ import annotations

import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.evaluation import classification_metrics, confusion_matrix_df, report_df
from src.preprocessing import (
    AVG_PROGRAMMING_COL,
    FEATURE_COLS,
    ID_COL,
    PROGRAMMING_COLS,
    TARGET_COL,
    add_target_from_programming_average,
    build_demo_dataset,
    clean_dataset,
    pearson_interpretation,
    validate_required_columns,
)
from src.training import (
    build_pipeline,
    cross_validated_predictions,
    find_best_threshold_for_recall,
    load_model_bundle,
    save_model_bundle,
    split_train_test,
)


st.set_page_config(page_title="Sistema Predictivo de Riesgo", page_icon="🎓", layout="wide")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def plot_histograms(df: pd.DataFrame, cols: list[str]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#1f77b4")
        axes[i].set_title(f"Histograma: {col}")
    fig.tight_layout()
    st.pyplot(fig)


def plot_boxplots(df: pd.DataFrame, cols: list[str]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#ffbf66")
        axes[i].set_title(f"Boxplot: {col}")
    fig.tight_layout()
    st.pyplot(fig)


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str]):
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    ax.set_title("Heatmap de correlacion")
    fig.tight_layout()
    st.pyplot(fig)


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def single_prediction_ui(bundle):
    st.subheader("Prediccion individual")
    st.caption("Entrada de notas matematicas y salida de clase + probabilidad de riesgo.")

    c1, c2 = st.columns(2)
    with c1:
        fundamentos = st.number_input("Fundamentos Matematicas", min_value=0.0, max_value=5.0, value=2.8, step=0.1)
        algebra = st.number_input("Algebra Lineal", min_value=0.0, max_value=5.0, value=3.1, step=0.1)
    with c2:
        diferencial = st.number_input("Calculo Diferencial", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        integral = st.number_input("Calculo Integral", min_value=0.0, max_value=5.0, value=2.7, step=0.1)

    if st.button("Predecir riesgo", type="primary"):
        sample = pd.DataFrame(
            [
                {
                    "fundamentos_matematicas": fundamentos,
                    "algebra_lineal": algebra,
                    "calculo_diferencial": diferencial,
                    "calculo_integral": integral,
                }
            ]
        )
        proba_risk = float(bundle["model"].predict_proba(sample)[0, 1])
        threshold = float(bundle["threshold"])
        pred = int(proba_risk >= threshold)
        if pred == 1:
            st.error(f"⚠ Riesgo academico detectado | Probabilidad: {proba_risk * 100:.1f}%")
        else:
            st.success(f"✅ No riesgo academico | Probabilidad de riesgo: {proba_risk * 100:.1f}%")
        st.write(f"Umbral de decision usado: **{threshold:.2f}**")


ensure_dirs()

st.title("Sistema Predictivo de Riesgo Academico")
st.write("Herramienta para identificar estudiantes en riesgo de bajo rendimiento en programacion.")

tab_manual_user, tab_manual_func, tab_app = st.tabs(["📖 Manual de Usuario", "🔧 Manual de Funcionalidades", "🚀 Aplicacion"])

with tab_manual_user:
    st.header("📖 Manual de Usuario")
    st.caption("Para la administracion de la Escuela de Sistemas")
    
    st.markdown("---")
    
    st.subheader("1. ¿Que es este sistema?")
    st.write("""
    Este sistema ayuda a identificar **estudiantes en riesgo academico** en cursos de programacion, 
    basandose en su desempeno en materias de matematicas.
    
    **Objetivo principal:** Detectar estudiantes que necesitan apoyo antes de que reprueben.
    """)
    
    st.markdown("---")
    
    st.subheader("2. Primeros pasos")
    
    with st.expander("Paso 1: Preparar tus datos", expanded=True):
        st.write("""
        Necesitas un archivo CSV con la informacion de los estudiantes. El archivo debe tener estas columnas:
        
        | Columna | Descripcion |
        |---------|-------------|
        | `id_estudiante` | Identificador unico del estudiante |
        | `fundamentos_matematicas` | Nota (0.0 - 5.0) |
        | `algebra_lineal` | Nota (0.0 - 5.0) |
        | `calculo_diferencial` | Nota (0.0 - 5.0) |
        | `calculo_integral` | Nota (0.0 - 5.0) |
        | `taller_programacion` | Nota (0.0 - 5.0) |
        | `desarrollo_software_1` | Nota (0.0 - 5.0) |
        | `desarrollo_software_2` | Nota (0.0 - 5.0) |
        
        **Nota:** Si no tienes datos, puedes usar los **datos de ejemplo** incluidos para probar el sistema.
        """)
    
    with st.expander("Paso 2: Cargar los datos"):
        st.write("""
        1. Ve a la pestaña **"Aplicacion"**
        2. Tienes dos opciones:
           - **Usar datos de ejemplo:** Marca la casilla "Usar datos de ejemplo" (ideal para probar)
           - **Subir tu propio CSV:** Haz clic en "Sube CSV historico" y selecciona tu archivo
        
        3. Veras una vista previa de tus datos en pantalla.
        """)
    
    with st.expander("Paso 3: Explorar los datos (EDA)"):
        st.write("""
        Despues de cargar los datos, puedes explorarlos visualmente:
        
        - **Estadistica descriptiva:** Promedios, medianas, minimos, maximos
        - **Distribucion de riesgo:** Grafica de cuantos estudiantes estan en riesgo
        - **Histogramas:** Distribucion de las notas en cada materia
        - **Boxplots:** Identificar valores atipicos
        - **Heatmap:** Correlacion entre las materias
        
        **Consejo:** Presta especial atencion al **Analisis de Pearson** - te muestra que materias de matematicas tienen mayor relacion con el desempeno en programacion.
        """)
    
    with st.expander("Paso 4: Entrenar el modelo"):
        st.write("""
        1. Haz clic en el boton **"Entrenar modelo"**
        2. El sistema:
           - Divide los datos en entrenamiento (80%) y prueba (20%)
           - Entrena un modelo **Random Forest**
           - Ajusta el umbral para maximizar la deteccion de estudiantes en riesgo
           - Ejecuta validacion cruzada para resultados mas confiables
        
        3. Veras las metricas del modelo:
           - **Recall:** Es la mas importante. Indica que porcentaje de estudiantes en riesgo realmente detecta el sistema.
           - **Precision:** De los que el sistema dice que estan en riesgo, que porcentaje realmente lo esta.
           - **Accuracy:** Precision general.
           - **F1:** Balance entre Recall y Precision.
        
        4. El modelo se guarda automaticamente en `models/modelo_riesgo.pkl`
        """)
    
    with st.expander("Paso 5: Ver resultados por estudiante"):
        st.write("""
        Despues de entrenar:
        1. Veras una tabla con **todos los estudiantes**
        2. Cada fila muestra:
           - Datos del estudiante
           - `probabilidad_riesgo`: Numero entre 0 y 1 (ej: 0.75 = 75% de probabilidad)
           - `prediccion_riesgo_programacion`: 1 = En riesgo, 0 = No en riesgo
        
        3. Haz clic en **"Descargar resultados"** para guardar el CSV completo.
        """)
    
    with st.expander("Paso 6: Prediccion individual"):
        st.write("""
        Una vez que tienes un modelo entrenado:
        
        1. Ve a la seccion **"Modo inferencia"**
        2. Ingresa las notas de matematicas de un estudiante:
           - Fundamentos Matematicas
           - Algebra Lineal
           - Calculo Diferencial
           - Calculo Integral
        
        3. Haz clic en **"Predecir riesgo"**
        4. El sistema te dira:
           - Si el estudiante esta **en riesgo** o **no en riesgo**
           - La probabilidad de riesgo
           - El umbral de decision usado
        """)
    
    st.markdown("---")
    
    st.subheader("3. Interpretacion de resultados")
    
    with st.expander("¿Como interpretar 'Riesgo'?"):
        st.write("""
        **Definicion de riesgo:**
        
        - Se calcula el **promedio de las 3 materias de programacion**
        - Si el promedio es **menor a 3.0** → El estudiante esta **EN RIESGO**
        - Si el promedio es **3.0 o mayor** → El estudiante **NO esta en riesgo**
        
        **Importante:** El sistema usa las notas de **matematicas** para **predecir** este riesgo.
        Las notas de programacion solo se usan para entrenar el modelo, no para predecir.
        """)
    
    with st.expander("¿Que es el umbral de decision?"):
        st.write("""
        El modelo da una **probabilidad** (ej: 65% de riesgo).
        
        El **umbral** decide a partir de que probabilidad se considera "en riesgo":
        
        - Umbral **bajo** (ej: 0.30): Detecta MAS estudiantes en riesgo, pero puede tener mas falsos positivos
        - Umbral **alto** (ej: 0.70): Detecta MENOS estudiantes, pero con mayor certeza
        
        **Este sistema:** Ajusta automaticamente el umbral para **maximizar el Recall**, es decir,
        prefiere detectar a todos los estudiantes en riesgo incluso si algunos no lo estan realmente.
        Esto es una decision institucional: es mejor dar apoyo a alguien que no lo necesita
        que dejar sin apoyo a alguien que si lo necesita.
        """)
    
    st.markdown("---")
    
    st.subheader("4. Tips y recomendaciones")
    st.info("""
    💡 **Tips:**
    
    1. **Primero prueba con datos de ejemplo** para entender como funciona el sistema
    2. **La calidad de los datos importa:** Asegurate que las notas esten entre 0.0 y 5.0
    3. **Recall es tu aliado:** Si el Recall es alto (>0.80), el sistema esta detectando bien
    4. **Descarga los resultados:** El CSV te permite trabajar con los datos en Excel
    5. **Reentrena periodicamente:** Con nuevos datos, el modelo mejora
    """)

with tab_manual_func:
    st.header("🔧 Manual de Funcionalidades")
    st.caption("Descripcion detallada de cada componente del sistema")
    
    st.markdown("---")
    
    st.subheader("📊 Modulo 1: Carga y Preprocesamiento de Datos")
    
    with st.expander("Funcionalidades de carga"):
        st.write("""
        **1. Datos de ejemplo integrados:**
        - Genera 220 registros sinteticos realistas
        - Las notas de programacion estan correlacionadas con las de matematicas
        - Incluye ruido aleatorio para simular datos reales
        
        **2. Carga de CSV personalizado:**
        - Valida que existan todas las columnas obligatorias
        - Muestra vista previa de los primeros 20 registros
        
        **3. Descarga de plantilla:**
        - Boton para descargar el dataset de ejemplo como plantilla
        """)
    
    with st.expander("Funcionalidades de limpieza (preprocesamiento)"):
        st.write("""
        **1. Conversion a numerico:**
        - Todas las columnas de notas se convierten a formato numerico
        - Valores no numericos se convierten a NaN (nulos)
        
        **2. Manejo de valores nulos:**
        - Filas con valores nulos en notas se eliminan
        - El sistema reporta cuantas filas se eliminaron
        
        **3. Validacion de rango:**
        - Todas las notas se ajustan al rango [0.0, 5.0]
        - Notas < 0.0 → Se establecen en 0.0
        - Notas > 5.0 → Se establecen en 5.0
        - El sistema reporta cuantas celdas se ajustaron
        
        **4. Creacion de variables derivadas:**
        - `promedio_programacion`: Promedio de las 3 materias de programacion
        - `riesgo_programacion`: Variable binaria (1 = riesgo, 0 = no riesgo)
          - 1 si `promedio_programacion` < 3.0
          - 0 en caso contrario
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Modulo 2: Analisis Exploratorio (EDA)")
    
    with st.expander("Estadistica descriptiva"):
        st.write("""
        **Metricas calculadas por cada columna:**
        - `mean`: Promedio aritmetico
        - `median`: Mediana (valor del medio)
        - `std`: Desviacion estandar (dispersion)
        - `min`: Valor minimo
        - `max`: Valor maximo
        
        **Columnas analizadas:**
        - 4 materias de matematicas (features)
        - 3 materias de programacion
        - `promedio_programacion`
        """)
    
    with st.expander("Visualizaciones"):
        st.write("""
        **1. Grafico de barras - Distribucion de riesgo:**
        - Muestra cuantos estudiantes estan en riesgo vs no en riesgo
        - Permite identificar desbalance de clases
        
        **2. Histogramas (con curva KDE):**
        - Distribucion de frecuencias por cada materia de matematicas
        - 4 graficos en disposicion 2x2
        - Identifica si las notas tienen distribucion normal
        
        **3. Boxplots:**
        - Visualiza mediana, cuartiles, y valores atipicos
        - Identifica outliers (valores extremos)
        - 4 graficos en disposicion 2x2
        
        **4. Heatmap de correlacion:**
        - Matriz de correlacion de Pearson entre todas las variables numericas
        - Valores van de -1.0 a +1.0
        - Colores mas intensos indican mayor correlacion
        - Ayuda a entender que variables tienen mayor relacion
        """)
    
    with st.expander("Analisis de Pearson"):
        st.write("""
        **Objetivo:** Medir la relacion lineal entre cada materia de matematicas y el promedio de programacion.
        
        **Coeficiente de correlacion (r):**
        - +1.0 = Correlacion positiva perfecta
        - 0.0 = Sin correlacion
        - -1.0 = Correlacion negativa perfecta
        
        **Interpretacion cualitativa:**
        - |r| < 0.20 → **Muy debil**
        - 0.20 ≤ |r| < 0.40 → **Debil**
        - 0.40 ≤ |r| < 0.60 → **Moderada**
        - 0.60 ≤ |r| < 0.80 → **Fuerte**
        - |r| ≥ 0.80 → **Muy fuerte**
        
        **Salida:** Tabla ordenada por mayor correlacion
        """)
    
    st.markdown("---")
    
    st.subheader("🤖 Modulo 3: Modelo Predictivo")
    
    with st.expander("Arquitectura del modelo"):
        st.write("""
        **Algoritmo:** Random Forest Classifier
        
        **Hiperparametros principales:**
        - `n_estimators`: 400 arboles de decision
        - `min_samples_leaf`: Minimo 2 muestras por hoja (evita sobreajuste)
        - `class_weight`: "balanced" (pesos inversamente proporcionales a frecuencia de clases)
        
        **Pipeline:**
        1. **SimpleImputer(strategy="median")**: Imputa valores faltantes con la mediana
        2. **RandomForestClassifier**: Modelo de clasificacion
        
        **Por que Random Forest?**
        - Robusto contra sobreajuste
        - Maneja bien datos con diferentes escalas
        - Proporciona importancia de variables
        - Buen rendimiento en problemas de clasificacion
        """)
    
    with st.expander("Entrenamiento y validacion"):
        st.write("""
        **1. Division de datos (Hold-out):**
        - 80% para entrenamiento
        - 20% para prueba
        - **Estratificacion:** Mantiene la proporcion de clases en ambos conjuntos
        
        **2. Busqueda de umbral optimo:**
        - Itera sobre umbrales desde 0.30 hasta 0.70 (paso de 0.01)
        - **Criterio:** Maximizar el **Recall** primero
        - **Desempate:** Mayor F1-score
        
        **3. Validacion cruzada (K-Fold):**
        - 5 folds estratificados
        - Shuffle aleatorio con semilla fija (42) para reproducibilidad
        - Usa `cross_val_predict` con `predict_proba`
        - Resultados mas robustos que hold-out simple
        
        **4. Metricas evaluadas:**
        - **Accuracy**: (VP + VN) / Total
        - **Precision**: VP / (VP + FP)
        - **Recall (Sensitivity)**: VP / (VP + FN) → **PRINCIPAL**
        - **F1-score**: 2 * (Precision * Recall) / (Precision + Recall)
        
        Donde:
        - VP = Verdaderos Positivos (en riesgo, detectados correctamente)
        - VN = Verdaderos Negativos (no en riesgo, detectados correctamente)
        - FP = Falsos Positivos (no en riesgo, marcados como riesgo)
        - FN = Falsos Negativos (en riesgo, NO detectados) → **MAS PELIGROSO**
        """)
    
    with st.expander("Persistencia del modelo"):
        st.write("""
        **Formato:** Bundle serializado con `joblib`
        
        **Contenido del bundle:**
        ```python
        {
            "model": Pipeline entrenado,
            "threshold": Umbral optimo encontrado,
            "features": Lista de nombres de columnas de entrada,
            "target": Nombre de la variable objetivo
        }
        ```
        
        **Ruta por defecto:** `models/modelo_riesgo.pkl`
        
        **Ventajas:**
        - El bundle es **autodescriptivo**
        - Resistente a cambios en el codigo
        - Contiene TODO lo necesario para inferencia
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Modulo 4: Resultados y Reportes")
    
    with st.expander("Metricas en pantalla"):
        st.write("""
        **1. Tarjetas de metricas:**
        - 4 columnas con las metricas principales
        - Formato: 3 decimales
        - Destaca "Recall (principal)"
        
        **2. Informacion del umbral:**
        - Muestra el umbral optimo encontrado
        - En formato: 2 decimales
        
        **3. Matriz de confusion:**
        - DataFrame etiquetado 2x2
        - Eje X: Prediccion del modelo
        - Eje Y: Valor real
        
        **4. Reporte de clasificacion:**
        - Por cada clase (0 = No riesgo, 1 = Riesgo)
        - Precision, Recall, F1-score, Support (cantidad de muestras)
        - Promedios: macro avg, weighted avg, accuracy
        """)
    
    with st.expander("Importancia de variables"):
        st.write("""
        **Fuente:** `feature_importances_` del Random Forest
        
        **Que mide:**
        - Cuanto contribuye cada variable a reducir la impureza (Gini)
        - Valores normalizados (suman 1.0)
        
        **Visualizacion:**
        - Grafico de barras ordenado de mayor a menor importancia
        - Permite identificar que materias de matematicas son mas predictivas
        
        **Ejemplo de interpretacion:**
        - Si `fundamentos_matematicas` tiene la mayor importancia → es la mejor predictora
        """)
    
    with st.expander("Resultados por estudiante y descarga"):
        st.write("""
        **Tabla de resultados:**
        Incluye las siguientes columnas por cada estudiante:
        - `id_estudiante`
        - 4 materias de matematicas (features)
        - `promedio_programacion` (promedio real)
        - `riesgo_programacion` (etiqueta real: 1 = riesgo)
        - `probabilidad_riesgo` (probabilidad predicha: 0.0 - 1.0)
        - `prediccion_riesgo_programacion` (clase predicha: 0 o 1)
        
        **Descarga CSV:**
        - Codificacion: UTF-8
        - Sin indice
        - Nombre de archivo sugerido: `resultados_prediccion_riesgo.csv`
        - Compatible con Excel, Google Sheets, etc.
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 Modulo 5: Inferencia (Prediccion Individual)")
    
    with st.expander("Requisitos previos"):
        st.write("""
        **Para usar el modo inferencia:**
        1. El modelo debe haber sido entrenado al menos una vez
        2. Debe existir el archivo: `models/modelo_riesgo.pkl`
        
        **Verificacion:**
        - Si existe el archivo → Muestra mensaje: "Modelo cargado correctamente"
        - Si NO existe → Muestra advertencia y deshabilita la funcionalidad
        """)
    
    with st.expander("Interfaz de entrada"):
        st.write("""
        **Campos de entrada (4 sliders):**
        1. **Fundamentos Matematicas** (0.0 - 5.0)
        2. **Algebra Lineal** (0.0 - 5.0)
        3. **Calculo Diferencial** (0.0 - 5.0)
        4. **Calculo Integral** (0.0 - 5.0)
        
        **Caracteristicas:**
        - Tipo: `number_input` con paso de 0.1
        - Rango validado: min=0.0, max=5.0
        - Valores por defecto: Datos de ejemplo (2.8, 3.1, 2.5, 2.7)
        - Disposicion: 2 columnas
        """)
    
    with st.expander("Proceso de prediccion"):
        st.write("""
        **Al hacer clic en "Predecir riesgo":**
        
        1. **Construccion del DataFrame:**
           ```python
           pd.DataFrame([{
               "fundamentos_matematicas": valor,
               "algebra_lineal": valor,
               "calculo_diferencial": valor,
               "calculo_integral": valor
           }])
           ```
        
        2. **Prediccion de probabilidad:**
           - `model.predict_proba(sample)[0, 1]`
           - Obtiene la probabilidad de la clase positiva (riesgo = 1)
        
        3. **Aplicacion del umbral:**
           ```python
           pred = int(proba_risk >= threshold)
           ```
           - Si probabilidad >= umbral → pred = 1 (En riesgo)
           - Si probabilidad < umbral → pred = 0 (No en riesgo)
        
        4. **Visualizacion del resultado:**
           - Color **rojo** (`st.error`) para riesgo detectado
           - Color **verde** (`st.success`) para no riesgo
           - Muestra: `Riesgo | Probabilidad: XX.X%`
           - Muestra el umbral de decision usado
        """)
    
    st.markdown("---")
    
    st.subheader("⚙️ Modulo 6: Configuracion")
    
    with st.expander("Opciones en el sidebar"):
        st.write("""
        **1. K-Fold para validacion cruzada:**
        - Selectbox con opciones: `[5]`
        - Por defecto: 5 folds
        - Actualmente es una lista fija (no editable)
        
        **2. Ruta de modelo .pkl:**
        - Campo de texto (`text_input`)
        - Valor por defecto: `"models/modelo_riesgo.pkl"`
        - Permite especificar rutas personalizadas
        
        **Uso:**
        - El valor de `model_path` se usa en:
          - `save_model_bundle(model, threshold, model_path)`
          - Verificacion de existencia: `os.path.exists(model_path)`
          - Carga para inferencia: `load_model_bundle(model_path)`
        """)
    
    st.markdown("---")
    
    st.subheader("🔄 Flujo completo de trabajo")
    
    st.code("""
    [Carga de datos]
         ↓
    [Validacion de columnas]
         ↓
    [Preprocesamiento: numeric → dropna → clip]
         ↓
    [Creacion de target: promedio → binarizacion < 3.0]
         ↓
    [EDA: descriptiva + visualizaciones + Pearson]
         ↓
    [Entrenamiento: split → fit → threshold tuning → CV]
         ↓
    [Evaluacion: metricas + confusion matrix + feature importance]
         ↓
    [Persistencia: joblib dump]
         ↓
    [Inferencia (opcional): formulario → prediccion individual]
    """, language=None)

with st.sidebar:
    st.header("Configuracion")
    cv_folds = st.selectbox("K-Fold para validacion cruzada", options=[5], index=0)
    model_path = st.text_input("Ruta de modelo .pkl", value="models/modelo_riesgo.pkl")

with tab_app:
    st.header("🚀 Aplicacion")
    st.write(
        "Modelo principal: Random Forest Classifier. La metrica prioritaria es **Recall** para detectar "
        "la mayor cantidad posible de estudiantes en riesgo."
    )

    st.subheader("1) Carga de datos")
    demo_df = build_demo_dataset()

    up_a, up_b = st.columns(2)
    with up_a:
        st.download_button(
            "Descargar dataset de ejemplo",
            data=to_csv_bytes(demo_df),
            file_name="datos_ejemplo.csv",
            mime="text/csv",
        )
    with up_b:
        use_demo = st.checkbox("Usar datos de ejemplo", value=True)

    uploaded = st.file_uploader("Sube CSV historico", type=["csv"])
    if use_demo:
        raw_df = demo_df.copy()
    elif uploaded is not None:
        raw_df = pd.read_csv(uploaded)
    else:
        raw_df = pd.DataFrame()

    if raw_df.empty:
        st.info("Carga un CSV o usa datos de ejemplo para continuar.")
        st.stop()

    st.dataframe(raw_df.head(20), use_container_width=True)

    ok, missing = validate_required_columns(raw_df)
    if not ok:
        st.error(f"Faltan columnas obligatorias: {', '.join(missing)}")
        st.stop()

    st.subheader("2) Preprocesamiento")
    clean_result = clean_dataset(raw_df)
    df = add_target_from_programming_average(clean_result.data, threshold=3.0)
    st.write(
        f"Filas eliminadas por nulos/no numericos: **{clean_result.dropped_rows}** | "
        f"Celdas ajustadas a rango [0, 5]: **{clean_result.clipped_cells}**"
    )
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("3) EDA")
    eda_cols = FEATURE_COLS + PROGRAMMING_COLS + [AVG_PROGRAMMING_COL]
    desc = df[eda_cols].describe().T[["mean", "50%", "std", "min", "max"]]
    desc = desc.rename(columns={"50%": "median"})
    st.write("Estadistica descriptiva")
    st.dataframe(desc, use_container_width=True)

    st.write("Distribucion de riesgo")
    risk_counts = df[TARGET_COL].value_counts().rename(index={1: "Riesgo", 0: "No riesgo"})
    st.bar_chart(risk_counts)

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Histogramas", "Boxplots", "Heatmap"])
    with eda_tab1:
        plot_histograms(df, FEATURE_COLS)
    with eda_tab2:
        plot_boxplots(df, FEATURE_COLS)
    with eda_tab3:
        plot_correlation_heatmap(df, eda_cols)

    st.subheader("4) Analisis estadistico (Pearson)")
    pearson_rows = []
    for col in FEATURE_COLS:
        coef = df[[col, AVG_PROGRAMMING_COL]].corr().iloc[0, 1]
        pearson_rows.append(
            {
                "variable": col,
                "pearson": round(float(coef), 4),
                "interpretacion": pearson_interpretation(float(coef)),
            }
        )
    pearson_df = pd.DataFrame(pearson_rows).sort_values(by="pearson", ascending=False)
    st.dataframe(pearson_df, use_container_width=True)

    st.subheader("5) Entrenamiento y validacion")

    if st.button("Entrenar modelo", type="primary"):
        x_train, x_test, y_train, y_test = split_train_test(df)

        model = build_pipeline()
        model.fit(x_train, y_train)

        test_proba = model.predict_proba(x_test)[:, 1]
        threshold_info = find_best_threshold_for_recall(y_test, test_proba)
        threshold = threshold_info["threshold"]

        cv_proba = cross_validated_predictions(build_pipeline(), df[FEATURE_COLS], df[TARGET_COL], folds=cv_folds)
        cv_pred = (cv_proba >= threshold).astype(int)
        metrics = classification_metrics(df[TARGET_COL], cv_pred)

        st.write("Metricas (validacion cruzada, umbral ajustado para recall)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall (principal)", f"{metrics['recall']:.3f}")
        m4.metric("F1", f"{metrics['f1']:.3f}")
        st.info(f"Umbral ajustado para maximizar recall: {threshold:.2f}")

        st.write("Matriz de confusion")
        st.dataframe(confusion_matrix_df(df[TARGET_COL], cv_pred), use_container_width=True)

        st.write("Reporte de clasificacion")
        st.dataframe(report_df(df[TARGET_COL], cv_pred), use_container_width=True)

        importances = model.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({"variable": FEATURE_COLS, "importancia": importances}).sort_values(
            by="importancia", ascending=False
        )
        st.write("Importancia de variables")
        st.bar_chart(importance_df.set_index("variable"))

        save_model_bundle(model, threshold, model_path)
        st.success(f"Modelo guardado en: {model_path}")

        full_proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
        full_pred = (full_proba >= threshold).astype(int)
        results_df = df[[ID_COL, *FEATURE_COLS, AVG_PROGRAMMING_COL, TARGET_COL]].copy()
        results_df["probabilidad_riesgo"] = np.round(full_proba, 4)
        results_df["prediccion_riesgo_programacion"] = full_pred
        st.write("Resultados por estudiante")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            "Descargar resultados",
            data=to_csv_bytes(results_df),
            file_name="resultados_prediccion_riesgo.csv",
            mime="text/csv",
        )

    st.subheader("6) Modo inferencia (usar modelo .pkl)")
    if os.path.exists(model_path):
        bundle = load_model_bundle(model_path)
        st.success("Modelo cargado correctamente.")
        single_prediction_ui(bundle)
    else:
        st.warning("Aun no existe el archivo de modelo. Entrena primero para habilitar inferencia.")

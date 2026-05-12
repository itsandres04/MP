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

st.title("Prototipo funcional de identificacion temprana de riesgo academico")
st.write(
    "Modelo principal: Random Forest Classifier. La metrica prioritaria es **Recall** para detectar "
    "la mayor cantidad posible de estudiantes en riesgo."
)

with st.sidebar:
    st.header("Configuracion")
    cv_folds = st.selectbox("K-Fold para validacion cruzada", options=[5], index=0)
    model_path = st.text_input("Ruta de modelo .pkl", value="models/modelo_riesgo.pkl")

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

"""Aplicación web (Streamlit) — Clasificador de tumores de mama.
Ejecutar:  streamlit run app.py
Proyecto educativo; NO sustituye un diagnóstico médico.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Clasificador de tumores de mama", page_icon="🩺", layout="wide")


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(BASE / "models" / "modelo.joblib")
    meta = json.loads((BASE / "models" / "metadata.json").read_text(encoding="utf-8"))
    return modelo, meta


@st.cache_data
def cargar_ejemplos(features):
    """Valores medianos por clase para cargar ejemplos rápidos."""
    df = pd.read_csv(BASE / "data" / "breast_cancer_wisconsin.csv")
    return {
        "Valores medianos (todos los casos)": df[list(features)].median().to_dict(),
        "Ejemplo típico de caso benigno": df[df["diagnosis"] == 0][list(features)].median().to_dict(),
        "Ejemplo típico de caso maligno": df[df["diagnosis"] == 1][list(features)].median().to_dict(),
    }


modelo, meta = cargar_modelo()
FEATURES = meta["features"]
EJEMPLOS = cargar_ejemplos(tuple(FEATURES))

NOMBRES = {
    "mean radius": "Radio medio",
    "mean texture": "Textura media",
    "mean perimeter": "Perímetro medio",
    "mean area": "Área media",
    "mean smoothness": "Suavidad media",
    "mean compactness": "Compacidad media",
    "mean concavity": "Concavidad media",
    "mean concave points": "Puntos cóncavos medios",
    "mean symmetry": "Simetría media",
    "mean fractal dimension": "Dimensión fractal media",
}


def aplicar_ejemplo():
    for f in FEATURES:
        st.session_state[f"inp_{f}"] = float(EJEMPLOS[st.session_state["ejemplo"]][f])


for f in FEATURES:  # valores iniciales
    st.session_state.setdefault(f"inp_{f}", float(meta["rangos"][f]["mediana"]))

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Acerca del modelo")
    st.write(f"**Algoritmo:** {meta['modelo']}")
    st.write("**Datos:** Breast Cancer Wisconsin (Diagnostic), UCI — 569 casos.")
    m = meta["metricas_prueba"]
    st.metric("Accuracy (prueba)", f"{m['Accuracy']:.1%}")
    st.metric("Recall clase maligna", f"{m['Recall']:.1%}")
    st.metric("Precision clase maligna", f"{m['Precision']:.1%}")
    st.caption("Métricas obtenidas en un conjunto de prueba de 114 casos; tienen incertidumbre.")

# ---------------- Cuerpo ----------------
st.title("🩺 Clasificador de tumores de mama")
st.write("Ingresa las medidas del núcleo celular (biopsia FNA) y el modelo estimará si el tumor es **benigno** o **maligno**.")
st.warning("Herramienta **educativa**. No es un dispositivo médico ni sustituye el diagnóstico de un profesional de la salud.")

st.selectbox("Cargar valores de ejemplo", list(EJEMPLOS.keys()), key="ejemplo", on_change=aplicar_ejemplo)

valores = {}
cols = st.columns(2)
for i, f in enumerate(FEATURES):
    r = meta["rangos"][f]
    paso = max((r["max"] - r["min"]) / 200, 1e-4)
    with cols[i % 2]:
        valores[f] = st.number_input(
            f"{NOMBRES.get(f, f)}",
            min_value=0.0, max_value=float(r["max"] * 1.5),
            step=float(paso), format="%.4f", key=f"inp_{f}",
            help=f"Rango observado en los datos: {r['min']:.4f} – {r['max']:.4f}",
        )
        if not (r["min"] <= valores[f] <= r["max"]):
            st.caption("⚠️ Fuera del rango observado en el entrenamiento; la predicción es menos confiable.")

if st.button("Predecir", type="primary"):
    X = pd.DataFrame([valores])[FEATURES]
    prob_maligno = float(modelo.predict_proba(X)[0, 1])
    clase = int(modelo.predict(X)[0])

    st.subheader("Resultado")
    if clase == 1:
        st.error(f"Predicción: **MALIGNO** (probabilidad estimada {prob_maligno:.1%})")
    else:
        st.success(f"Predicción: **BENIGNO** (probabilidad de ser maligno: {prob_maligno:.1%})")
    st.progress(prob_maligno, text=f"Probabilidad de tumor maligno: {prob_maligno:.1%}")
    st.caption("Cualquier resultado debe ser confirmado por un especialista.")

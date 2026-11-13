"""Aplicación académica, NO clínica: clasificador supervisado de registros WDBC.

Ejecutar: python -m streamlit run app.py
El modelo de sklearn y el orden de variables son exactamente los del proyecto revisado.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Clasificador de muestras | Proyecto IA",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Diseño sobrio y legible, basado en los componentes habituales de Streamlit.
st.markdown(
    """
<style>
    .stApp { background-color: #fafbfc; color: #243443; }
    .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #253747; letter-spacing: normal; }
    h1 { font-size: 2.15rem; }
    h2 { font-size: 1.55rem; }
    div[data-testid="stTabs"] button[role="tab"] { font-weight: 550; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #1e6867; }
    div.stButton > button { border-radius: 6px; }
    div.stButton > button[kind="primary"] {
        background: #22686b; border: 1px solid #22686b; color: white;
        border-radius: 6px; font-weight: 600; box-shadow: none;
    }
    div.stButton > button[kind="primary"]:hover { background: #175759; color: white; }
    div[data-testid="stExpander"] { border: 1px solid #dbe1e5; border-radius: 7px; }
    div[data-testid="stMetric"] { border: 1px solid #dbe1e5; border-radius: 7px; padding: .85rem 1rem; }
    .project-head { border-bottom: 2px solid #22686b; padding: 0 0 19px; margin-bottom: 20px; }
    .project-head .course { color: #567079; font-size: .86rem; font-weight: 600; margin-bottom: 8px; }
    .project-head h1 { font-size: clamp(1.75rem, 3vw, 2.2rem); margin: 0 0 9px; }
    .project-head p { max-width: 880px; color: #526470; font-size: .98rem; line-height: 1.6; margin: 0; }
    .safe-note { background: #f1f5f4; border-left: 3px solid #51817b; color: #324e4d;
        padding: 12px 15px; border-radius: 2px; margin: 10px 0 23px; font-size: .92rem; }
    .section-kicker { color: #557078; font-size: .82rem; font-weight: 650; margin: 15px 0 5px; }
    .section-help { color: #667781; font-size: .92rem; margin-top: -5px; }
    .fact { border: 1px solid #dfe5e8; background: white; border-radius: 7px;
        padding: 12px 15px; margin-bottom: 12px; min-height: 80px; }
    .fact .value { font-size: 1.28rem; color: #263c46; font-weight: 700; }
    .fact .cap { color: #647780; font-size: .81rem; margin-top: 1px; }
    .picked { background: #edf4f1; color: #315f57; border: 1px solid #d0dfd9;
        border-radius: 5px; padding: 10px 13px; font-size: .91rem; margin: 14px 0 17px; }
    .preview { background: white; border: 1px solid #dfe5e8; border-radius: 7px;
        padding: 12px 15px; margin-bottom: 7px; }
    .preview .num { font-size: 1.26rem; font-weight: 700; color: #263d4b; }
    .preview .lbl { font-size: .82rem; color: #61747e; margin-top: 3px; }
    .result { background: #eef5f2; border-left: 4px solid #327b69;
        border-radius: 5px; padding: 18px 21px; margin: 17px 0 11px; }
    .result.maligno { background: #faf1ef; border-left-color: #b16557; }
    .result .r-label { color: #647781; font-weight: 600; font-size: .8rem; }
    .result .r-class { color: #256953; font-size: 1.7rem; font-weight: 750; margin: 4px 0 6px; }
    .result.maligno .r-class { color: #9a524b; }
    .result .r-desc { font-size: .95rem; line-height: 1.55; color: #455a61; }
    .mini-note { color: #687b84; font-size: .85rem; line-height: 1.5; margin-top: 7px; }
    .foot { font-size: .82rem; color: #687a83; margin: 32px 0 5px;
        border-top: 1px solid #dce3e7; padding-top: 14px; }
    @media(max-width: 750px) {
       .block-container { padding-left: 1rem; padding-right: 1rem; }
       .project-head h1 { font-size: 1.65rem; }
    }
</style>
    """,
    unsafe_allow_html=True,
)

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

DESCRIPCIONES = {
    "mean radius": "Tamaño promedio aproximado del núcleo celular.",
    "mean texture": "Variación de tonos de gris observada en la imagen.",
    "mean perimeter": "Medida aproximada del contorno del núcleo.",
    "mean area": "Superficie aproximada del núcleo celular.",
    "mean smoothness": "Variación local de la longitud del radio.",
    "mean compactness": "Relación matemática entre perímetro y área.",
    "mean concavity": "Intensidad de partes hundidas del contorno.",
    "mean concave points": "Cantidad relativa de puntos hundidos del contorno.",
    "mean symmetry": "Simetría del núcleo celular.",
    "mean fractal dimension": "Medida matemática de la complejidad del contorno.",
}

@st.cache_resource
def cargar_modelo():
    return (
        joblib.load(BASE / "models" / "modelo.joblib"),
        json.loads((BASE / "models" / "metadata.json").read_text(encoding="utf-8")),
    )

@st.cache_data
def cargar_ejemplos(features: tuple[str, ...]):
    df = pd.read_csv(BASE / "data" / "breast_cancer_wisconsin.csv")
    campos = list(features)
    return {
        "general": df[campos].median().to_dict(),
        "benigno": df.loc[df["diagnosis"] == 0, campos].median().to_dict(),
        "maligno": df.loc[df["diagnosis"] == 1, campos].median().to_dict(),
    }

@st.cache_data
def cargar_comparacion():
    return pd.read_csv(BASE / "data" / "comparacion_modelos.csv")

try:
    modelo, meta = cargar_modelo()
    FEATURES = meta["features"]
    EJEMPLOS = cargar_ejemplos(tuple(FEATURES))
except (OSError, ValueError, KeyError, ImportError) as exc:
    st.error("No se pueden cargar los datos o el modelo. Revisa las carpetas `models` y `data`.")
    st.code(str(exc))
    st.stop()

PRESETS = {
    "benigno": "Ejemplo tipo benigno",
    "maligno": "Ejemplo tipo maligno",
    "general": "Valores generales",
}


def usar_preset(cual: str) -> None:
    """Actualiza todos los controles *antes* de que Streamlit vuelva a renderizar."""
    ejemplo = EJEMPLOS[cual]
    for feature in FEATURES:
        st.session_state[f"valor::{feature}"] = float(ejemplo[feature])
    st.session_state["ejemplo_activo"] = cual
    st.session_state["resultado"] = None


def cambio_manual() -> None:
    st.session_state["ejemplo_activo"] = "personalizado"
    st.session_state["resultado"] = None


if "ejemplo_activo" not in st.session_state:
    st.session_state["ejemplo_activo"] = "general"
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None
for feature in FEATURES:
    st.session_state.setdefault(f"valor::{feature}", float(EJEMPLOS["general"][feature]))

st.markdown(
    """
<section class="project-head">
  <div class="course">Actividad 3.2.1 · Fundamentos de Inteligencia Artificial</div>
  <h1>Clasificador de muestras de mama</h1>
  <p>Proyecto de aprendizaje automático supervisado. Permite probar un modelo con diez mediciones
  de núcleos celulares y revisar cómo se compararon cuatro algoritmos de clasificación.</p>
</section>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="safe-note"><strong>Uso educativo.</strong> La aplicación muestra una simulación basada en
    datos públicos de investigación. No permite diagnosticar ni descartar enfermedades,
    ni sustituye una evaluación médica. Los ejemplos son valores estadísticos, no casos clínicos individuales.</div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1,c2,c3,c4], ["569","10","4","SVM"],
    ["Registros del dataset", "Variables utilizadas", "Modelos comparados", "Modelo elegido"],
):
    with col:
        st.markdown(f'<div class="fact"><div class="value">{val}</div><div class="cap">{label}</div></div>', unsafe_allow_html=True)

simulador, funcionamiento, resultados = st.tabs([
    "Simulador", "Cómo funciona", "Resultados"
])

with simulador:
    st.markdown('<div class="section-kicker">Paso 1. Selecciona los datos</div>', unsafe_allow_html=True)
    st.subheader("Selecciona un ejemplo")
    st.markdown('<p class="section-help">Selecciona un ejemplo para cargar sus mediciones. También puedes ajustar los valores.</p>', unsafe_allow_html=True)

    p1,p2,p3 = st.columns(3)
    with p1:
        with st.container(border=True):
            st.markdown("#### Ejemplo benigno")
            st.caption("Valores medianos de registros etiquetados como benignos en el conjunto de datos.")
            st.button("Usar datos benignos", key="preset_b", use_container_width=True,
                      on_click=usar_preset, args=("benigno",))
    with p2:
        with st.container(border=True):
            st.markdown("#### Ejemplo maligno")
            st.caption("Valores medianos de registros etiquetados como malignos en el conjunto de datos.")
            st.button("Usar datos malignos", key="preset_m", use_container_width=True,
                      on_click=usar_preset, args=("maligno",))
    with p3:
        with st.container(border=True):
            st.markdown("#### Datos generales")
            st.caption("Valores medianos calculados usando todos los registros, sin distinción de clase.")
            st.button("Usar datos generales", key="preset_g", use_container_width=True,
                      on_click=usar_preset, args=("general",))

    activo = st.session_state["ejemplo_activo"]
    activo_txt = PRESETS.get(activo,"Valores personalizados")
    st.markdown(f'<div class="picked"><strong>Datos seleccionados:</strong> {activo_txt}. Puedes ajustar sus valores abajo.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Paso 2. Revisa los valores</div>', unsafe_allow_html=True)
    st.subheader("Mediciones del ejemplo seleccionado")
    cols = st.columns(4)
    destacados = ("mean radius", "mean texture", "mean area", "mean concavity")
    for col, f in zip(cols,destacados):
        with col:
            numero = st.session_state[f"valor::{f}"]
            formato = f"{numero:.4f}" if numero < 1 else f"{numero:.2f}"
            st.markdown(
                f'<div class="preview"><div class="num">{formato}</div><div class="lbl">{NOMBRES[f]}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption("Estas cifras provienen de mediciones de laboratorio. No es necesario modificarlas para probar el proyecto.")
    with st.expander("Ver o modificar las diez mediciones (opcional)"):
        st.markdown("**Edición opcional:** modifica las mediciones para comparar las respuestas. No introduzcas datos personales de pacientes.")
        groups = [
            ("Tamaño del núcleo", FEATURES[:4]),
            ("Textura y forma", FEATURES[4:7]),
            ("Detalles del contorno", FEATURES[7:]),
        ]
        for titulo, feats in groups:
            st.markdown(f"**{titulo}**")
            gg = st.columns(2)
            for i, f in enumerate(feats):
                rango = meta["rangos"][f]
                with gg[i % 2]:
                    st.number_input(
                        NOMBRES[f], min_value=0.0,
                        max_value=float(max(rango["max"]*2, rango["max"]+1)),
                        step=float(max((rango["max"]-rango["min"])/200, 0.0001)),
                        format="%.4f", key=f"valor::{f}",
                        help=(DESCRIPCIONES[f] + " Rango del conjunto de entrenamiento: "
                              f"{rango['min']:.4f} a {rango['max']:.4f}."),
                        on_change=cambio_manual,
                    )

    valores = {f: float(st.session_state[f"valor::{f}"]) for f in FEATURES}
    fuera = [NOMBRES[f] for f in FEATURES
             if valores[f] < meta["rangos"][f]["min"] or valores[f] > meta["rangos"][f]["max"]]
    if fuera:
        st.warning("Algunos valores quedan fuera del rango de entrenamiento: " + ", ".join(fuera)
                   + ". La salida del modelo es especialmente incierta.")

    st.markdown('<div class="section-kicker">Paso 3. Genera la predicción</div>', unsafe_allow_html=True)
    st.subheader("Resultado del clasificador")
    st.caption("El modelo SVM utilizará las diez características para generar una clasificación de ejemplo.")
    if st.button("Calcular clasificación", key="predict", type="primary", use_container_width=True):
        entrada = pd.DataFrame([valores], columns=FEATURES)
        clase = int(modelo.predict(entrada)[0])
        # El modelo emplea clases 0=benigno y 1=maligno.
        indice_maligno = list(modelo.classes_).index(1)
        prob = float(modelo.predict_proba(entrada)[0, indice_maligno])
        st.session_state["resultado"] = {"clase": clase, "prob": prob, "fuera": bool(fuera)}

    if st.session_state["resultado"] is not None:
        r = st.session_state["resultado"]
        es_maligno = (r["clase"] == 1)
        clase_nombre = "MALIGNO" if es_maligno else "BENIGNO"
        desc = ("El modelo clasificó las mediciones en la categoría maligna del conjunto de investigación."
                if es_maligno else
                "El modelo clasificó las mediciones en la categoría benigna del conjunto de investigación.")
        st.markdown(
            f'''<div class="result {'maligno' if es_maligno else ''}">
                <div class="r-label">RESULTADO DEL MODELO</div>
                <div class="r-class">Clase estimada: {clase_nombre}</div>
                <div class="r-desc">{desc}</div>
            </div>''', unsafe_allow_html=True,
        )
        st.markdown("**Salida numérica del modelo para la clase maligna**")
        st.progress(float(r["prob"]), text=f"{r['prob']:.1%} (cálculo del algoritmo)")
        st.markdown(
            '<div class="mini-note">Este porcentaje es una salida matemática del clasificador, '
            'no es la probabilidad clínica de padecer cáncer. Incluso un valor cercano al 100 % '
            'no significa certeza médica.</div>', unsafe_allow_html=True,
        )
        if r["fuera"]:
            st.warning("Precaución adicional: algunos valores estaban fuera del rango de entrenamiento.")
        with st.expander("¿Cómo interpretar el resultado?"):
            st.write("El sistema compara diez medidas numéricas con los patrones que aprendió de un conjunto histórico de datos etiquetados. "
                     "Devuelve una categoría matemática de ese conjunto, no un diagnóstico.")
            st.write("Para evaluar si el modelo acierta se usan registros de prueba independientes; "
                     "un resultado aislado no permite medir su desempeño real.")
    else:
        st.info("Selecciona un ejemplo y pulsa **Calcular clasificación** para ver el resultado.")

with funcionamiento:
    st.markdown('<div class="section-kicker">Metodología</div>', unsafe_allow_html=True)
    st.subheader("Etapas del proyecto")
    st.write("Aprendizaje **supervisado** significa que un algoritmo estudia ejemplos que ya tienen una etiqueta conocida y aprende patrones para clasificar nuevos valores.")
    pasos = [
        ("01 · Datos", "Se utilizaron 569 registros del conjunto Breast Cancer Wisconsin (Diagnostic)."),
        ("02 · Preparación", "Se seleccionaron 10 características numéricas y se separaron datos de entrenamiento y prueba (455 / 114)."),
        ("03 · Comparación", "Se evaluaron KNN, SVM, Random Forest y Árbol de Decisión mediante validación cruzada."),
        ("04 · Simulación", "El modelo SVM procesa diez valores y devuelve una clase estimada del conjunto de datos."),
    ]
    for i in range(0,4,2):
        cc = st.columns(2)
        for col,(titulo,descripcion) in zip(cc,pasos[i:i+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {titulo}")
                    st.write(descripcion)
    st.markdown("### Conceptos básicos")
    with st.expander("¿Qué es SVM?"):
        st.write("Una máquina de vectores de soporte. Aprende una frontera matemática que busca separar grupos de ejemplos.")
    with st.expander("¿Qué significan benigno y maligno aquí?"):
        st.write("Son las dos etiquetas del conjunto de datos utilizado en el proyecto. La aplicación clasifica mediciones; no evalúa personas.")
    with st.expander("¿Qué es accuracy y qué es recall?"):
        st.write("**Accuracy:** proporción de registros de prueba clasificados correctamente. **Recall de la clase maligna:** proporción de registros malignos del conjunto de prueba que el modelo identificó como tales.")
    st.info("**Limitación clave:** el conjunto de datos tiene 569 registros de una sola fuente y no cuenta con validación clínica externa. No debe usarse en decisiones reales de salud.")

with resultados:
    st.markdown('<div class="section-kicker">Evidencia del trabajo</div>', unsafe_allow_html=True)
    st.subheader("Resultados obtenidos con datos de prueba")
    st.write("Se reservaron **114 registros** para evaluar los modelos. Estos porcentajes describen únicamente su comportamiento en esa muestra, no su eficacia médica.")
    met = meta["metricas_prueba"]
    cc = st.columns(4)
    for col, label, valor in zip(cc,
        ["Accuracy", "Recall (maligno)", "Precision (maligno)", "F1"],
        [met["Accuracy"], met["Recall"], met["Precision"], met["F1"]],
    ):
        with col:
            st.metric(label, f"{valor:.1%}")
    try:
        comparacion = cargar_comparacion()
        st.markdown("### Comparación de los cuatro algoritmos")
        tabla = comparacion[["Modelo","Accuracy","Precision","Recall","F1","ROC_AUC"]].copy()
        tabla.columns = ["Modelo","Exactitud","Precisión","Recall","F1","ROC AUC"]
        st.dataframe(tabla, hide_index=True, use_container_width=True,
                     column_config={col:st.column_config.NumberColumn(col, format="%.3f")
                                    for col in ["Exactitud","Precisión","Recall","F1","ROC AUC"]})
        # No apilar métricas: son puntuaciones independientes en [0, 1].
        # Un selector permite compararlas individualmente sin sumar porcentajes.
        metricas_grafica = {
            "Accuracy": ("Exactitud (Accuracy)", "De todos los registros, qué proporción clasificó correctamente cada modelo."),
            "Precision": ("Precisión (Precision)", "De los registros clasificados como malignos, qué proporción era realmente maligna."),
            "Recall": ("Sensibilidad (Recall)", "De los registros malignos, qué proporción identificó correctamente el modelo."),
            "F1": ("Puntuación F1", "Equilibra precisión y sensibilidad en una sola medida."),
            "ROC_AUC": ("ROC AUC", "Resume la capacidad del modelo de distinguir entre las dos clases."),
        }
        metrica_seleccionada = st.selectbox(
            "Elige una métrica para comparar los algoritmos",
            options=list(metricas_grafica),
            format_func=lambda clave: metricas_grafica[clave][0],
            index=0,
            help="Cada gráfica presenta una sola métrica para evitar sumar puntuaciones diferentes.",
        )
        st.bar_chart(
            comparacion.set_index("Modelo")[[metrica_seleccionada]],
            y_label="Puntuación (0 a 1)",
            x_label="Modelo",
        )
        st.caption(metricas_grafica[metrica_seleccionada][1] + " Valores entre 0 y 1; cuanto más alto, mejor en esta muestra.")
        st.caption("SVM y Random Forest obtuvieron resultados muy cercanos en validación cruzada. "
                   "SVM fue el modelo elegido por el equipo, según los criterios descritos en el informe.")
    except (OSError, KeyError, ValueError):
        st.info("La tabla comparativa está disponible en el archivo `data/comparacion_modelos.csv`.")
    st.markdown("### ¿Qué significa la matriz de confusión?")
    st.write("En los 114 registros de prueba, el SVM clasificó correctamente 71 benignos y 40 malignos; "
             "clasificó incorrectamente 1 benigno y 2 malignos. Estos errores muestran por qué la aplicación no puede usarse clínicamente.")
    confusion = pd.DataFrame(
        [[71,1],[2,40]],
        index=["Real: benigno", "Real: maligno"],
        columns=["Estimado: benigno", "Estimado: maligno"],
    )
    st.dataframe(confusion, use_container_width=True)
    st.caption("Fuente de datos: Breast Cancer Wisconsin (Diagnostic), UCI / scikit-learn. "
               "Proyecto académico de la Actividad 3.2.1.")

st.markdown(
    '<div class="foot">Actividad 3.2.1 · Fundamentos de Inteligencia Artificial · '
    'Universidad Politécnica de Atlautla · Prototipo educativo de aprendizaje supervisado.</div>',
    unsafe_allow_html=True,
)

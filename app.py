"""
CardioAI - Dashboard de Predicción de Riesgo Cardiovascular
============================================================
Aplicación Streamlit para predecir riesgo de enfermedad coronaria
usando el modelo XGBoost entrenado en CardioAI.ipynb

Para ejecutar localmente:
    streamlit run app.py

Para desplegar en Streamlit Community Cloud:
    Ver Streamlit_deploy.ipynb para instrucciones completas
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ AÑADIDO: necesario para cargar modelo .json
import json            # ✅ AÑADIDO: necesario para cargar feature_names.json
import os

# ─── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Riesgo Cardiovascular",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .result-box-positive {
        background-color: #ffeaea;
        border-left: 5px solid #e74c3c;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-box-negative {
        background-color: #eafaf1;
        border-left: 5px solid #27ae60;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #f39c12;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Carga del modelo ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Carga el modelo XGBoost (.json) y las features desde archivos .json."""
    model_path = "model/cardioai_model.json"
    features_path = "model/feature_names.json"

    if not os.path.exists(model_path):
        return None, None, "Archivo `cardioai_model.json` no encontrado. Ejecuta primero `CardioAI.ipynb`."
    if not os.path.exists(features_path):
        return None, None, "Archivo `feature_names.json` no encontrado. Ejecuta primero `CardioAI.ipynb`."

    # ✅ CORREGIDO: usar xgb.XGBClassifier().load_model() en lugar de joblib.load()
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # ✅ CORREGIDO: usar json.load() en lugar de joblib.load()
    with open(features_path, "r") as f:
        features = json.load(f)

    return model, features, None


model, feature_names, load_error = load_model()

# ─── Encabezado ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">CardioAI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predicción de riesgo de enfermedad coronaria · UCI Heart Disease Dataset</p>',
    unsafe_allow_html=True
)

if load_error:
    st.error(load_error)
    st.stop()

# ─── Formulario de entrada ────────────────────────────────────────────────────
st.markdown('<p class="section-header">Datos del paciente</p>', unsafe_allow_html=True)
st.markdown("Ingresa los valores clínicos del paciente para obtener la predicción:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Edad (años)", min_value=20, max_value=90, value=55,
        help="Edad del paciente en años completos"
    )
    sex = st.selectbox(
        "Sexo biológico",
        options=[("Femenino", 0), ("Masculino", 1)],
        format_func=lambda x: x[0],
        help="Sexo biológico del paciente"
    )
    cp_options = {
        "Asintomático (0)": 0,
        "Angina atípica (1)": 1,
        "Dolor no anginoso (2)": 2,
        "Angina típica (3)": 3,
    }
    cp = st.selectbox(
        "Tipo de dolor en el pecho",
        options=list(cp_options.keys()),
        index=0,
        help="Clasificación clínica del dolor torácico"
    )
    trestbps = st.number_input(
        "Presión arterial en reposo (mm Hg)", min_value=80, max_value=220, value=130,
        help="Presión arterial sistólica medida en reposo"
    )
    chol = st.number_input(
        "Colesterol sérico (mg/dl)", min_value=100, max_value=700, value=240,
        help="Nivel de colesterol total en sangre"
    )
    fbs = st.selectbox(
        "Glucosa en ayunas > 120 mg/dl",
        options=[("No (0)", 0), ("Sí (1)", 1)],
        format_func=lambda x: x[0],
        help="¿El nivel de glucosa en ayunas supera los 120 mg/dl?"
    )

with col2:
    restecg_options = {
        "Normal (0)": 0,
        "Anormalidad ST-T (1)": 1,
        "Hipertrofia ventricular izquierda (2)": 2,
    }
    restecg = st.selectbox(
        "Resultado ECG en reposo",
        options=list(restecg_options.keys()),
        index=1,
        help="Resultado del electrocardiograma en reposo"
    )
    thalch = st.number_input(
        "Frecuencia cardíaca máxima (lpm)", min_value=60, max_value=220, value=150,
        help="Frecuencia cardíaca máxima alcanzada durante la prueba de esfuerzo"
    )
    exang = st.selectbox(
        "Angina inducida por ejercicio",
        options=[("No (0)", 0), ("Sí (1)", 1)],
        format_func=lambda x: x[0],
        help="¿Se presentó angina durante la prueba de esfuerzo?"
    )
    oldpeak = st.number_input(
        "Depresión del ST (oldpeak)", min_value=-3.0, max_value=8.0, value=1.0, step=0.1,
        help="Depresión del segmento ST inducida por el ejercicio relativa al reposo"
    )
    slope_options = {
        "Pendiente negativa - downsloping (0)": 0,
        "Pendiente plana - flat (1)": 1,
        "Pendiente positiva - upsloping (2)": 2,
    }
    slope = st.selectbox(
        "Pendiente del segmento ST",
        options=list(slope_options.keys()),
        index=1,
        help="Pendiente del segmento ST peak durante el ejercicio"
    )

# ─── Construcción del vector de entrada ──────────────────────────────────────
input_values = {
    "age": age,
    "sex": sex[1],
    "cp": cp_options[cp],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs[1],
    "restecg": restecg_options[restecg],
    "thalch": thalch,
    "exang": exang[1],
    "oldpeak": oldpeak,
    "slope": slope_options[slope],
}

input_df = pd.DataFrame([input_values])[feature_names]

# ─── Predicción ───────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("Predecir riesgo cardiovascular", type="primary", use_container_width=True)

if predict_btn:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box-positive">
            <h3>⚠️ Riesgo elevado de enfermedad coronaria</h3>
            <p><strong>Probabilidad estimada de enfermedad: {probability:.1%}</strong></p>
            <p>El modelo identifica patrones compatibles con enfermedad coronaria. 
            Se recomienda evaluación médica especializada.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-negative">
            <h3>✅ Bajo riesgo de enfermedad coronaria</h3>
            <p><strong>Probabilidad estimada de enfermedad: {probability:.1%}</strong></p>
            <p>El modelo no identifica patrones de alto riesgo con los datos proporcionados.
            Los controles preventivos regulares siguen siendo recomendables.</p>
        </div>
        """, unsafe_allow_html=True)

    # Barra de probabilidad
    st.markdown("#### Probabilidad de enfermedad")
    st.progress(float(probability))
    st.caption(f"{probability:.1%} — umbral de decisión: 50%")

    # Datos ingresados
    with st.expander("Ver datos ingresados al modelo"):
        st.dataframe(input_df.T.rename(columns={0: "Valor"}), use_container_width=True)

# ─── Aviso legal ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="warning-box">
    ⚕️ <strong>Aviso médico importante:</strong> Esta herramienta es exclusivamente académica y educativa.
    <strong>No sustituye el diagnóstico médico profesional.</strong> Los resultados son orientativos y 
    no deben utilizarse para tomar decisiones clínicas. Ante síntomas cardiovasculares, consulta 
    inmediatamente a un profesional de la salud.
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<br><p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "CardioAI · UCI Heart Disease Dataset · Modelo XGBoost · Proyecto académico"
    "</p>",
    unsafe_allow_html=True
)

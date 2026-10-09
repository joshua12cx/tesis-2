import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de Preeclampsia", page_icon="🤰", layout="centered")

@st.cache_resource
def cargar():
    modelo = joblib.load("modelo_preeclampsia.joblib")
    cfg = json.load(open("config_modelo.json"))
    return modelo, cfg["features"], cfg["threshold"]

modelo, FEATURES, UMBRAL = cargar()

st.title("Detección temprana del riesgo de preeclampsia")
st.caption("Metamodelo Ensemble Stacking · gestantes peruanas · primer control prenatal")
st.warning("Herramienta académica de apoyo. No constituye un diagnóstico médico.")

with st.form("form"):
    c1, c2 = st.columns(2)
    edad = c1.number_input("Edad (años)", 12, 50, 28)
    edad_gest = c2.number_input("Edad gestacional (semanas)", 8, 14, 11)
    peso = c1.number_input("Peso (kg)", 35.0, 160.0, 65.0, 0.1)
    talla = c2.number_input("Talla (cm)", 120.0, 200.0, 155.0, 0.5)
    sis = c1.number_input("P.A. sistólica (mmHg)", 70, 220, 110)
    dia = c2.number_input("P.A. diastólica (mmHg)", 40, 140, 70)
    hb = c1.number_input("Hemoglobina (g/dL)", 5.0, 20.0, 12.0, 0.1)
    st.markdown("**Antecedentes**")
    a, b, c = st.columns(3)
    hta = a.checkbox("Hipertensión")
    dm = b.checkbox("Diabetes")
    fam = c.checkbox("Antec. familiar de hipertensión")
    ok = st.form_submit_button("Calcular riesgo")

if ok:
    imc = peso / (talla / 100) ** 2
    pam = (sis + 2 * dia) / 3
    fila = {"edad": edad, "edad_gestacion": edad_gest, "imc": imc, "pam": pam,
            "hemoglobina": hb, "hipertension": int(hta), "diabetes": int(dm),
            "ant_fam_hiper": int(fam)}
    X = pd.DataFrame([fila])[FEATURES]
    p = float(modelo.predict_proba(X)[0, 1])
    st.metric("Probabilidad estimada", f"{p:.1%}")
    st.caption(f"IMC calculado: {imc:.1f} · PAM calculada: {pam:.1f} mmHg · umbral de decisión: {UMBRAL:.2f}")
    if p >= UMBRAL:
        st.error("RIESGO ALTO: se recomienda evaluación y seguimiento clínico.")
    else:
        st.success("Riesgo bajo según el modelo.")

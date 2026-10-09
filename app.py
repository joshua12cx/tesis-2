import json, warnings
import joblib, numpy as np, pandas as pd, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Riesgo de Preeclampsia · IA explicable", page_icon="🩺", layout="wide")

NOMBRES = {
    "edad": "Edad", "edad_gestacion": "Edad gestacional", "imc": "IMC", "pam": "Presión arterial media",
    "hemoglobina": "Hemoglobina", "hipertension": "Hipertensión", "diabetes": "Diabetes",
    "ant_fam_hiper": "Antec. familiar HTA",
}
ROJO, VERDE, AZUL, AMBAR = "#ff6b8b", "#4ade80", "#60a5fa", "#fbbf24"

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1200px}
.hero{background:linear-gradient(135deg,#1b2236 0%,#2a1f3d 100%);border:1px solid #2c3550;border-radius:18px;padding:26px 30px;margin-bottom:18px}
.hero h1{margin:0;font-size:1.9rem;letter-spacing:-.5px}
.hero p{margin:6px 0 0;color:#aab3c8;font-size:.95rem}
.card{background:#161b26;border:1px solid #262e42;border-radius:16px;padding:20px 22px;height:100%}
.card h4{margin:0 0 10px;font-size:.85rem;text-transform:uppercase;letter-spacing:1.2px;color:#8b95ad;font-weight:600}
.big{font-size:3rem;font-weight:700;line-height:1}
.badge{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:600;font-size:.85rem;margin-top:8px}
.mini{display:flex;justify-content:space-between;color:#aab3c8;font-size:.85rem;padding:5px 0;border-bottom:1px solid #222a3c}
.mini b{color:#e8ecf4}
.kpi{background:#161b26;border:1px solid #262e42;border-radius:14px;padding:14px 16px;text-align:center}
.kpi .v{font-size:1.5rem;font-weight:700}.kpi .l{color:#8b95ad;font-size:.75rem;text-transform:uppercase;letter-spacing:1px}
.aviso{background:#2a2312;border:1px solid #4a3d16;color:#fbd77a;border-radius:12px;padding:10px 14px;font-size:.85rem}
div[data-testid="stForm"]{border:1px solid #262e42;border-radius:16px;background:#12161f}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar():
    modelo = joblib.load("modelo_preeclampsia.joblib")
    cfg = json.load(open("config_modelo.json"))
    bg = pd.read_csv("background.csv")[cfg["features"]]
    glob = json.load(open("shap_global.json"))
    f = lambda a: modelo.predict_proba(pd.DataFrame(a, columns=cfg["features"]))[:, 1]
    explainer = shap.Explainer(f, bg, seed=42)
    return modelo, cfg["features"], cfg["threshold"], explainer, glob


modelo, FEATURES, UMBRAL, explainer, GLOBAL = cargar()


def gauge(p, umbral):
    """Semicirculo SVG: la marca blanca es el umbral de decision."""
    import math
    def pt(frac, r=80):
        a = math.pi * (1 - frac)
        return 100 + r * math.cos(a), 100 - r * math.sin(a)
    x, y = pt(min(max(p, 0.001), 0.999)); tx, ty = pt(umbral, 92); tx2, ty2 = pt(umbral, 68)
    color = ROJO if p >= umbral else VERDE
    return f"""<svg viewBox="0 0 200 118" width="100%">
    <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#252d42" stroke-width="16" stroke-linecap="round"/>
    <path d="M20 100 A80 80 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
    <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#fff" stroke-width="2.5"/>
    <text x="100" y="98" text-anchor="middle" fill="#e8ecf4" font-size="30" font-weight="700">{p:.0%}</text>
    <text x="100" y="114" text-anchor="middle" fill="#8b95ad" font-size="8">probabilidad estimada</text></svg>"""


st.markdown("""<div class="hero"><h1>🩺 Detección temprana del riesgo de preeclampsia</h1>
<p>Metamodelo Ensemble Stacking (RF · CatBoost · XGBoost · LightGBM · LR) con explicabilidad SHAP · gestantes peruanas · primer control prenatal</p></div>""",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Predicción individual", "🔍 Importancia global (SHAP)", "ℹ️ Sobre el modelo"])

with tab1:
    izq, der = st.columns([5, 7], gap="large")
    with izq:
        with st.form("form"):
            st.markdown("##### Datos de la gestante")
            c1, c2 = st.columns(2)
            edad = c1.number_input("Edad (años)", 12, 50, 28)
            edad_g = c2.number_input("Edad gestacional (sem.)", 8, 14, 11)
            peso = c1.number_input("Peso (kg)", 35.0, 160.0, 65.0, 0.1)
            talla = c2.number_input("Talla (cm)", 120.0, 200.0, 155.0, 0.5)
            sis = c1.number_input("P.A. sistólica (mmHg)", 70, 220, 110)
            dia = c2.number_input("P.A. diastólica (mmHg)", 40, 140, 70)
            hb = st.number_input("Hemoglobina (g/dL)", 5.0, 20.0, 12.0, 0.1)
            st.markdown("##### Antecedentes")
            a, b, c = st.columns(3)
            hta = a.checkbox("Hipertensión")
            dm = b.checkbox("Diabetes")
            fam = c.checkbox("Antec. fam. HTA")
            ok = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")
        st.markdown('<div class="aviso">⚠️ Herramienta académica de apoyo. No constituye un diagnóstico médico.</div>',
                    unsafe_allow_html=True)

    with der:
        if not ok:
            st.markdown('<div class="card"><h4>Resultado</h4><p style="color:#aab3c8">Completa el formulario y pulsa '
                        '<b>Calcular riesgo</b> para ver la probabilidad estimada y qué variables la explican.</p></div>',
                        unsafe_allow_html=True)
        else:
            imc = peso / (talla / 100) ** 2
            pam = (sis + 2 * dia) / 3
            fila = pd.DataFrame([{"edad": edad, "edad_gestacion": edad_g, "imc": imc, "pam": pam,
                                  "hemoglobina": hb, "hipertension": int(hta), "diabetes": int(dm),
                                  "ant_fam_hiper": int(fam)}])[FEATURES]
            p = float(modelo.predict_proba(fila)[0, 1])
            alto = p >= UMBRAL
            col, tag = (ROJO, "RIESGO ALTO · requiere evaluación clínica") if alto else (VERDE, "RIESGO BAJO según el modelo")
            g1, g2 = st.columns([6, 5])
            g1.markdown(f'<div class="card"><h4>Riesgo estimado</h4>{gauge(p, UMBRAL)}'
                        f'<span class="badge" style="background:{col}22;color:{col};border:1px solid {col}66">{tag}</span></div>',
                        unsafe_allow_html=True)
            g2.markdown(f"""<div class="card"><h4>Valores calculados</h4>
              <div class="mini"><span>IMC</span><b>{imc:.1f} kg/m²</b></div>
              <div class="mini"><span>PAM</span><b>{pam:.1f} mmHg</b></div>
              <div class="mini"><span>Umbral de decisión</span><b>{UMBRAL:.2f}</b></div>
              <div class="mini"><span>Marca blanca</span><b>= umbral</b></div></div>""", unsafe_allow_html=True)

            st.markdown("##### ¿Por qué este resultado? (SHAP)")
            sv = explainer(fila)
            nombres = [f"{NOMBRES[f]} = {fila.iloc[0][f]:.1f}" if f in ("imc", "pam", "hemoglobina")
                       else f"{NOMBRES[f]} = {int(fila.iloc[0][f])}" for f in FEATURES]
            ex = shap.Explanation(values=sv.values[0], base_values=sv.base_values[0],
                                  data=None, feature_names=nombres)
            with plt.rc_context({"figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117", "text.color": "#e8ecf4",
                                 "axes.labelcolor": "#e8ecf4", "xtick.color": "#aab3c8", "ytick.color": "#e8ecf4"}):
                shap.plots.waterfall(ex, max_display=8, show=False)
                fig = plt.gcf(); fig.set_size_inches(7.5, 4.2); fig.patch.set_facecolor("#0e1117")
                st.pyplot(fig, width="stretch"); plt.close(fig)
            orden = np.argsort(-np.abs(sv.values[0]))[:3]
            frases = []
            for i in orden:
                sube = sv.values[0][i] > 0
                frases.append(f"**{NOMBRES[FEATURES[i]]}** {'aumenta' if sube else 'reduce'} el riesgo "
                              f"({sv.values[0][i]*100:+.1f} pts)")
            st.caption("Rojo = empuja la predicción hacia mayor riesgo · Azul = hacia menor riesgo. "
                       f"Valor base (gestante promedio): {sv.base_values[0]:.1%}.")
            st.info("Principales factores: " + " · ".join(frases))

with tab2:
    imp = pd.Series(GLOBAL["importance"], index=[NOMBRES[f] for f in GLOBAL["features"]]).sort_values()
    st.markdown("##### Importancia global de variables (media |SHAP| en 100 casos de Test)")
    with plt.rc_context({"figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117", "text.color": "#e8ecf4",
                         "axes.labelcolor": "#e8ecf4", "xtick.color": "#aab3c8", "ytick.color": "#e8ecf4"}):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp.index, imp.values, color=[ROJO if v == imp.max() else AZUL for v in imp.values])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#2c3550"); ax.set_xlabel("Contribución media a la probabilidad")
        for i, v in enumerate(imp.values):
            ax.text(v + imp.max() * 0.01, i, f"{v:.3f}", va="center", color="#e8ecf4", fontsize=9)
        st.pyplot(fig, width="stretch"); plt.close(fig)
    st.markdown(f"La **presión arterial media (PAM)** domina las predicciones del modelo, muy por encima del resto de variables. "
                "Esta importancia describe la lógica del modelo, no relaciones causales.")

with tab3:
    k = st.columns(5)
    for col, (v, l) in zip(k, [("93.7 %", "Recall (Test)"), ("71.8 %", "Precision (Test)"), ("0.813", "F1 (Test)"),
                               ("0.983", "ROC-AUC (Test)"), ("0.946", "PR-AUC (Test)")]):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.markdown("""
**Arquitectura.** Preprocesamiento (imputación + estandarización) → SMOTE (solo en entrenamiento) → 5 modelos base
(Random Forest, CatBoost, XGBoost, LightGBM, Regresión Logística) → metamodelo de Regresión Logística.

**Umbral de decisión: 0.55.** Elegido en validación priorizando Recall ≥ 0.93 y maximizando F1, para minimizar falsos negativos
(en Test: 74 de 79 casos detectados, 5 falsos negativos, 29 falsos positivos).

**Variables de entrada:** edad, edad gestacional, IMC, presión arterial media, hemoglobina, hipertensión, diabetes y antecedente familiar de hipertensión.

**Equipo:** Hidalgo Jauregui Karla Monica · Pedraza Perez Joshua Josue · Quispe Mamani Deyvis.
""")

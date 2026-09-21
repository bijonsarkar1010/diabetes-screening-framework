"""
streamlit_app.py — clinician-facing dashboard on top of orchestrator.py.

Run with: streamlit run streamlit_app.py

Lets a clinician enter whatever patient data they have (EHR fields are
required as the baseline; retinal image and wearable data are optional),
and shows the fused risk score, explanation, and referral recommendation —
this is the "human oversight layer" from the paper made concrete.
"""

import json
import tempfile
import os

import streamlit as st
from orchestrator import PatientInput, run_screening

st.set_page_config(page_title="Diabetes Screening Assistant", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 640px;}
    div[data-testid="stMetricValue"] {font-weight: 500;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Multimodal diabetes screening — clinician view")
st.caption(
    "Prototype implementation of the framework in *Toward Trustworthy Multimodal AI "
    "for Chronic Disease Screening*. This tool produces a triage recommendation only — "
    "it does not diagnose. A clinician makes the final call."
)

if not (os.path.exists("ehr_model.json") or os.path.exists("imaging_model.pt")):
    st.warning(
        "⚠️ Demo mode: no trained model files found (ehr_model.json / imaging_model.pt). "
        "Scores shown below are placeholder heuristics, not validated clinical outputs. "
        "Train ehr_model.py / imaging_model.py and place the resulting files in this "
        "folder to switch to real model inference.",
        icon="⚠️",
    )

st.subheader("1. Patient EHR data (required)")
col1, col2, col3 = st.columns(3)
with col1:
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=400, value=120)
with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.1)
with col3:
    age = st.number_input("Age", min_value=1, max_value=110, value=45)

st.subheader("2. Retinal image (optional)")
uploaded_image = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])

st.subheader("3. Wearable data (optional)")
col4, col5 = st.columns(2)
with col4:
    has_wearable = st.checkbox("Patient has wearable data")
with col5:
    pass
avg_hr, steps = None, None
if has_wearable:
    avg_hr = st.number_input("Average heart rate (bpm, last 7 days)", min_value=30, max_value=200, value=75)
    steps = st.number_input("Average daily steps (last 7 days)", min_value=0, max_value=30000, value=5000)

patient_id = st.text_input("Patient ID", value="P0001")

if st.button("Run screening", type="primary"):
    image_path = None
    if uploaded_image is not None:
        suffix = os.path.splitext(uploaded_image.name)[1] or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_image.read())
        tmp.close()
        image_path = tmp.name

    patient = PatientInput(
        ehr_data={"Glucose": glucose, "BMI": bmi, "Age": age},
        image_path=image_path,
        wearable_data={"avg_heart_rate": avg_hr, "steps": steps} if has_wearable else None,
    )

    try:
        result = run_screening(patient, patient_id=patient_id)
    except ValueError as e:
        st.error(str(e))
    else:
        st.subheader("Result")

        category_colors = {"low": "green", "moderate": "orange", "high": "red"}
        color = category_colors.get(result["risk_category"], "gray")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Risk score", f"{result['risk_score']} / 100")
        with c2:
            st.markdown(
                f"**Risk category:** :{color}[{result['risk_category'].upper()}]  \n"
                f"**Recommended action:** {result['recommended_action']}"
            )

        st.markdown("**Modalities used:** " + ", ".join(result["modalities_used"]))

        st.markdown("**Modality contributions**")
        contrib = {k: v for k, v in result["modality_contributions"].items() if v is not None}
        st.bar_chart(contrib)

        st.markdown("**Explanation**")
        st.write(result["explanation_summary"])

        if result["demo_mode"]:
            st.info(
                "This result used one or more demo placeholder scores, not trained "
                "models. Do not use for anything beyond a UI walkthrough.",
                icon="ℹ️",
            )

        with st.expander("Full structured output (audit log record)"):
            st.code(json.dumps(result, indent=2), language="json")

    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

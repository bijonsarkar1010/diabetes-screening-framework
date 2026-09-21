# Trustworthy multimodal AI for chronic disease screening

Reference implementation and companion paper for a regulation-compliant,
explainable AI framework for multimodal (EHR + retinal imaging + wearable)
type 2 diabetes screening in low-resource settings, mapped to EU AI Act and
India CDSCO medical device software obligations.

## Repo layout

```
orchestrator.py        Core system — modality detection, model routing, fusion
streamlit_app.py        Clinician-facing dashboard UI (run with `streamlit run streamlit_app.py`)
.streamlit/config.toml   Minimal custom theme for the dashboard
training/                Offline scripts to train the EHR and imaging models
docs/                    Companion research paper (.docx) and architecture diagram
requirements.txt
```

## Quick start (demo mode, no trained models needed)

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens a browser dashboard where you can enter patient EHR data, optionally
upload a retinal image, and optionally add wearable data. The system detects
which modalities are present and fuses only those. Without trained model
files, every score is a clearly-labeled placeholder heuristic — never a
validated clinical output.

## Training real models

See `training/README.md` for dataset download instructions (Pima Indians
Diabetes Database, APTOS 2019 Blindness Detection) and how to run
`training/ehr_model.py` / `training/imaging_model.py`. Once
`ehr_model.json` and/or `imaging_model.pt` exist in the repo root,
`orchestrator.py` automatically uses them instead of the placeholder logic
— no code changes needed.

## Paper

`docs/Trustworthy_Multimodal_AI_Diabetes_Screening_Framework.docx` — the
full framework paper, targeting IEEE JBHI's "AI-Driven Multimodal
Intelligence for Sustainable Healthcare Systems" special issue. Covers the
five-layer regulatory framework, related work, the case study, and the
system design in this repo (Section VI).

## Status

Research prototype. The fusion and orchestration logic is tested and
working; the EHR and imaging models have not yet been trained on real data
in this repo — see `training/README.md` to do that next.

## License

MIT — see `LICENSE`.

# Training scripts

Offline, one-time scripts that produce the trained model files the main
system (`../orchestrator.py`) picks up automatically. Not run by end users.

## 1. Datasets

**Pima Indians Diabetes Database** (for `ehr_model.py`)
- `pip install ucimlrepo` then `from ucimlrepo import fetch_ucirepo; ds = fetch_ucirepo(id=891)`
- Or download the CSV and place it at `../data/pima.csv`.

**APTOS 2019 Blindness Detection** (for `imaging_model.py`)
- Kaggle competition — needs a free Kaggle account + API token (`kaggle.json`).
- `kaggle competitions download -c aptos2019-blindness-detection`
- Unzip so images sit at `../data/aptos/train_images/*.png` and labels at
  `../data/aptos/train.csv`.

## 2. Run (from the repo root, with requirements.txt installed)

```
python training/ehr_model.py
python training/imaging_model.py
python training/fusion.py
```

`ehr_model.py` and `imaging_model.py` save `ehr_model.json` and
`imaging_model.pt` respectively into the current working directory — run
them from the repo root so `orchestrator.py` finds them automatically.

`fusion.py` is a standalone reference for the fusion math and output
schema; the same logic is implemented properly (with modality detection)
in `../orchestrator.py`, which is what the Streamlit app actually uses.

## 3. GPU

`imaging_model.py` fine-tunes a ResNet18 — run this on Colab/Kaggle with a
free GPU, not on a laptop CPU, or training will be very slow.

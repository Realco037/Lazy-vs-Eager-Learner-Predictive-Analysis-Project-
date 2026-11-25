# Lazy vs Eager Learning: KNN vs Decision Tree

An interactive Data Science project comparing K-Nearest Neighbors (lazy learning) and Decision Tree (eager learning) on the Wine Recognition dataset.

## Run locally

Use Python 3.11 or 3.12 for a predictable environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/simulator.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select this repository and the `main` branch.
3. Set the app file path to `app/simulator.py`.
4. Deploy. Streamlit reads dependencies from the root `requirements.txt`.

## Dataset location

The app resolves paths relative to `app/simulator.py`, so it works regardless of the shell's current working directory. It checks:
- `dataset/wine_classification.csv`
- `notebooks and dataset/wine_classification.csv`
- `notebooks_and_dataset/wine_classification.csv`

The current CSV is in `notebooks and dataset/`.

## Project contents

- `app/simulator.py`: interactive Streamlit simulator
- `notebooks and dataset/lazy_vs_eager_learner.ipynb`: predictive analysis notebook
- `notebooks and dataset/wine_classification.csv`: dataset
- `results/`: experiment outputs
- `screenshots/`: plots and evaluation images
- `report/`: report and presentation

Both models use the same stratified 75/25 train/test split (`random_state=42`). KNN uses StandardScaler inside a Pipeline; the scaler is fitted only on training data. Decision Tree uses unscaled features. Timing depends on the machine and is not a universal ranking of the algorithms.

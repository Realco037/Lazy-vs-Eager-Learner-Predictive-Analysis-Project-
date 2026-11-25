# Lazy vs Eager Learning: Interactive Wine Classifier

An interactive Streamlit project comparing K-Nearest Neighbors (lazy learning) and Decision Tree (eager learning) using the Wine Recognition dataset.

## Simulator
Enter the 13 chemical measurements for a wine sample and submit the same input to both trained models. The app shows:
- Each model's predicted class
- Estimated class probabilities in side-by-side charts
- Whether the models agree or disagree
- Your submitted values alongside dataset minimum, median, and maximum values

This is an educational classification demonstration. Model probabilities are not guarantees or laboratory measurements and should not be used to judge real-world wine quality, safety, origin, or authenticity.

## Run locally
Use Python 3.11 or 3.12 for a predictable environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/simulator.py
```

On macOS/Linux: `source .venv/bin/activate`.

## Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select this repository and the `main` branch.
3. Set the app file path to `app/simulator.py`.
4. Deploy. Commits to the selected branch should trigger a redeploy.

## Dataset and models
The app resolves the CSV relative to `app/simulator.py` and checks:
- `dataset/wine_classification.csv`
- `notebooks and dataset/wine_classification.csv`
- `notebooks_and_dataset/wine_classification.csv`

The current CSV is in `notebooks and dataset/`. Both models are trained on the full reference dataset for interactive inference:
- KNN: StandardScaler + KNeighborsClassifier (k=5)
- Decision Tree: DecisionTreeClassifier (max_depth=4, random_state=42)

KNN scaling is applied through a Pipeline at both training and prediction time.

## Project files
- `app/simulator.py`: interactive Streamlit app
- `notebooks and dataset/lazy_vs_eager_learner.ipynb`: predictive analysis notebook
- `notebooks and dataset/wine_classification.csv`: dataset
- `results/`, `screenshots/`, `report/`: project outputs and documentation

# Lazy vs Eager Learning: KNN vs Decision Tree

A separate Data Science project comparing K-Nearest Neighbors (lazy learning) and Decision Tree (eager learning) on the UCI Wine Recognition dataset.

## Dataset
- 178 records
- 13 independent numerical features
- 3 wine classes
- CSV: `dataset/wine_classification.csv`
- Original source: UCI Machine Learning Repository, Wine dataset.

## Objectives
1. Explain lazy and eager learning.
2. Inspect a real dataset with at least 12 independent attributes.
3. Preprocess data and create a reproducible stratified split.
4. Train KNN as the lazy learner.
5. Train and visualize Decision Tree as the eager learner using `plot_tree`.
6. Compare accuracy, macro precision, recall, F1, training time and prediction time.
7. Interpret findings and limitations.

## Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open `notebooks/predictive_analysis.ipynb` and run all cells.

Run the interactive simulator:
```bash
streamlit run app/simulator.py
```

## Methodology
Both models use the same stratified 75/25 split (random_state=42). KNN uses StandardScaler inside a Pipeline; Decision Tree uses unscaled features because tree splits are not scale-dependent. The scaler is fitted only on training data. Timing is hardware-dependent and is not a universal property of the algorithms.

## Structure
- `dataset/`: CSV data
- `notebooks/`: predictive analysis notebook
- `app/`: Streamlit simulator
- `results/`: measured model metrics and predictions
- `screenshots/`: generated charts and Decision Tree plot
- `report/`: report and presentation

Student, registration number, course code and supervisor details must be completed before submission.

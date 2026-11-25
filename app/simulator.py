import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Lazy vs Eager Learning", page_icon="⚖️", layout="wide")
st.title("⚖️ Lazy vs Eager Learning Simulator")
st.caption("K-Nearest Neighbors (Lazy) vs Decision Tree (Eager)")

# Resolve paths from the repository, not the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "wine_classification.csv",
    PROJECT_ROOT / "notebooks and dataset" / "wine_classification.csv",
    PROJECT_ROOT / "notebooks_and_dataset" / "wine_classification.csv",
]
DATA_PATH = next((candidate for candidate in DATA_CANDIDATES if candidate.is_file()), None)

if DATA_PATH is None:
    checked_paths = "\n".join(str(candidate) for candidate in DATA_CANDIDATES)
    st.error(
        "Dataset not found. Ensure wine_classification.csv is committed to the repository. "
        "Paths checked:\n" + checked_paths
    )
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Unable to read dataset at {DATA_PATH}: {exc}")
    st.stop()

if "target_class" not in df.columns:
    st.error(
        "Expected a 'target_class' column in the CSV. "
        f"Found columns: {', '.join(map(str, df.columns))}"
    )
    st.stop()

X = df.drop(columns=["target_class"])
y = df["target_class"]

if X.empty or y.nunique() < 2:
    st.error("The dataset must contain at least one feature and two target classes.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5)),
])
tree = DecisionTreeClassifier(max_depth=4, random_state=42)

st.caption(f"Dataset loaded: {len(df)} records, {X.shape[1]} features, {y.nunique()} classes.")
st.markdown(
    "Both models receive exactly the same held-out test records. "
    "Prediction timing measures the batch prediction call only; table rendering is excluded."
)

if "simulation" not in st.session_state:
    st.session_state.simulation = None

if st.button("Run comparison", type="primary"):
    models = {
        "KNN — Lazy learner": knn,
        "Decision Tree — Eager learner": tree,
    }
    output = {}

    for name, model in models.items():
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        # Exclude a warm-up call from the measured prediction time.
        model.predict(X_test.iloc[:1])
        prediction_start = time.perf_counter()
        predictions = model.predict(X_test)
        prediction_time = time.perf_counter() - prediction_start

        rows = []
        for index, prediction in enumerate(predictions):
            actual = y_test.iloc[index]
            rows.append({
                "Test record": index + 1,
                "Predicted class": str(prediction),
                "Actual class": str(actual),
                "Correct": bool(prediction == actual),
            })

        output[name] = {
            "fit_time": fit_time,
            "prediction_time": prediction_time,
            "rows": rows,
        }

    st.session_state.simulation = output

if st.session_state.simulation:
    result = st.session_state.simulation
    col1, col2 = st.columns(2)

    for column, name in zip((col1, col2), result.keys()):
        data = result[name]
        correct = sum(row["Correct"] for row in data["rows"])
        total = len(data["rows"])

        with column:
            st.subheader(name)
            st.metric("Correct predictions", f"{correct}/{total}")
            st.metric("Accuracy", f"{correct / total * 100:.2f}%")
            st.metric("Training time", f"{data['fit_time']:.6f} s")
            st.metric("Prediction time (test batch)", f"{data['prediction_time']:.6f} s")
            st.progress(1.0, text=f"{total} / {total} test records processed")
            st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True, height=300)

    st.subheader("Side-by-side performance")
    summary = pd.DataFrame([
        {
            "Model": name,
            "Training time (s)": values["fit_time"],
            "Prediction time (s)": values["prediction_time"],
            "Accuracy (%)": (
                sum(row["Correct"] for row in values["rows"])
                / len(values["rows"]) * 100
            ),
        }
        for name, values in result.items()
    ])
    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary.set_index("Model")[["Accuracy (%)"]])
else:
    st.info("Select **Run comparison** to train both models and process the same test records.")

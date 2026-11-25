from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Lazy vs Eager Learning Lab", page_icon="⚖️", layout="wide")
st.title("⚖️ Lazy vs Eager Learning Lab")
st.write("Enter the chemical measurements of a wine sample. Both models classify the exact same input so you can compare their predictions.")

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ROOT / "dataset" / "wine_classification.csv",
    ROOT / "notebooks and dataset" / "wine_classification.csv",
    ROOT / "notebooks_and_dataset" / "wine_classification.csv",
]
CSV_PATH = next((p for p in CANDIDATES if p.is_file()), None)
if CSV_PATH is None:
    st.error("Dataset not found. Checked:\n" + "\n".join(map(str, CANDIDATES)))
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def build_models(path):
    data = load_data(path)
    if "target_class" not in data.columns:
        raise ValueError("CSV must contain target_class.")
    features = data.drop(columns="target_class")
    target = data["target_class"]
    if features.empty or target.nunique() < 2:
        raise ValueError("Dataset needs features and at least two target classes.")
    knn_model = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=min(5, len(data)))),
    ])
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    knn_model.fit(features, target)
    tree_model.fit(features, target)
    return data, features, target, knn_model, tree_model

try:
    data, X, y, knn, tree = build_models(str(CSV_PATH))
except Exception as exc:
    st.error(f"Could not load/train models: {exc}")
    st.stop()

classes = list(knn.classes_)
labels = [f"Class {c}" for c in classes]
st.caption(f"Reference data: {len(data)} samples · {X.shape[1]} attributes · {len(classes)} classes")

with st.expander("How the two learning approaches work"):
    st.markdown("""
    - **KNN (lazy learner):** stores the training examples and compares your sample with its nearest neighbours when you request a prediction. Features are standardised for distance calculations.
    - **Decision Tree (eager learner):** learns decision rules from the training data before you submit a sample.
    - Both models are trained on the same full dataset and receive the same values from this form.
    - Probability bars show model-estimated class probabilities, not guaranteed correctness or laboratory measurements.
    """)

pretty = {
    "alcohol":"Alcohol","malic_acid":"Malic acid","ash":"Ash",
    "alcalinity_of_ash":"Alcalinity of ash","magnesium":"Magnesium",
    "total_phenols":"Total phenols","flavanoids":"Flavanoids",
    "nonflavanoid_phenols":"Nonflavanoid phenols",
    "proanthocyanins":"Proanthocyanins","color_intensity":"Color intensity",
    "hue":"Hue","od280/od315_of_diluted_wines":"OD280/OD315 of diluted wines",
    "proline":"Proline",
}

st.header("1. Enter your wine sample")
st.caption("Starting values are the dataset medians. Inputs are bounded by the observed dataset ranges.")
with st.form("sample_form"):
    entered = {}
    cols = st.columns(3)
    for i, feature in enumerate(X.columns):
        values = pd.to_numeric(X[feature], errors="coerce").dropna()
        lo, hi, default = float(values.min()), float(values.max()), float(values.median())
        if lo == hi:
            lo, hi = lo - 0.01, hi + 0.01
        step = max((hi - lo) / 1000, 0.001)
        digits = 3 if step < 0.01 else 2
        with cols[i % 3]:
            entered[feature] = st.number_input(
                pretty.get(feature, feature.replace("_"," ").title()),
                min_value=lo, max_value=hi, value=min(max(default,lo),hi),
                step=step, format=f"%.{digits}f",
                help=f"Observed range: {lo:.3f} – {hi:.3f}",
            )
    submitted = st.form_submit_button("🔬 Classify with both models", type="primary", use_container_width=True)

if submitted:
    sample = pd.DataFrame([[entered[c] for c in X.columns]], columns=X.columns)
    st.session_state["prediction_result"] = {
        "sample": sample,
        "knn_class": knn.predict(sample)[0],
        "tree_class": tree.predict(sample)[0],
        "knn_probs": knn.predict_proba(sample)[0],
        "tree_probs": tree.predict_proba(sample)[0],
    }

if "prediction_result" in st.session_state:
    result = st.session_state["prediction_result"]
    kp, tp = result["knn_class"], result["tree_class"]
    st.divider()
    st.header("2. Prediction results")
    if kp == tp:
        st.success(f"Both models agree: Class {kp}")
    else:
        st.warning(f"Models disagree — KNN predicts Class {kp}; Decision Tree predicts Class {tp}.")

    left, right = st.columns(2)
    for col, title, pred, probs in [
        (left, "KNN · Lazy learner", kp, result["knn_probs"]),
        (right, "Decision Tree · Eager learner", tp, result["tree_probs"]),
    ]:
        with col:
            st.subheader(title)
            st.metric("Predicted class", f"Class {pred}")
            st.metric("Top class probability", f"{np.max(probs)*100:.1f}%")
            chart = pd.DataFrame({"Estimated probability": probs}, index=labels)
            st.bar_chart(chart)

    st.subheader("3. Probability comparison")
    comparison = pd.DataFrame({
        "KNN (%)": result["knn_probs"] * 100,
        "Decision Tree (%)": result["tree_probs"] * 100,
    }, index=labels)
    st.bar_chart(comparison)
    st.dataframe(comparison.style.format("{:.1f}%"), use_container_width=True)

    with st.expander("See the values you submitted and dataset reference values"):
        reference = pd.DataFrame({
            "Your input": result["sample"].iloc[0],
            "Dataset minimum": X.min(),
            "Dataset median": X.median(),
            "Dataset maximum": X.max(),
        })
        st.dataframe(reference, use_container_width=True)

    if kp != tp:
        st.info("Disagreement is useful: the models have learned different decision mechanisms and may separate borderline samples differently.")
else:
    st.info("Complete the measurements above and submit the form to compare both predictions.")

st.divider()
st.caption("Educational demonstration only. Predictions do not establish a real wine's origin, quality, safety, or authenticity.")

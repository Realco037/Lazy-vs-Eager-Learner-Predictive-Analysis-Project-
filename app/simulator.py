import streamlit as st
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Lazy vs Eager Learning", page_icon="⚖️", layout="wide")
st.title("⚖️ Lazy vs Eager Learning Simulator")
st.caption("K-Nearest Neighbors (Lazy) vs Decision Tree (Eager)")

DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "wine_classification.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target_class"])
y = df["target_class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
class_names = ["class_0", "class_1", "class_2"]

knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))])
tree = DecisionTreeClassifier(max_depth=4, random_state=42)

if "simulation" not in st.session_state:
    st.session_state.simulation = None

st.markdown("Both models receive exactly the same held-out test records. Model prediction is timed as a batch; table rendering is excluded.")
if st.button("Run comparison", type="primary"):
    models = {"KNN — Lazy learner": knn, "Decision Tree — Eager learner": tree}
    output = {}
    for name, model in models.items():
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        # Warm-up is excluded from the measured prediction call.
        model.predict(X_test.iloc[:1])
        pred_start = time.perf_counter()
        predictions = model.predict(X_test)
        prediction_time = time.perf_counter() - pred_start

        rows = []
        for i, prediction in enumerate(predictions):
            rows.append({
                "Test question / record": i + 1,
                "Predicted class": class_names[int(prediction)],
                "Actual class": class_names[int(y_test.iloc[i])],
                "Correct": bool(prediction == y_test.iloc[i])
            })
        output[name] = {
            "fit_time": fit_time,
            "prediction_time": prediction_time,
            "rows": rows
        }
    st.session_state.simulation = output

if st.session_state.simulation:
    result = st.session_state.simulation
    c1, c2 = st.columns(2)
    for col, name in zip([c1,c2], result.keys()):
        data = result[name]
        correct = sum(r["Correct"] for r in data["rows"])
        with col:
            st.subheader(name)
            st.metric("Correct predictions", f"{correct}/{len(data['rows'])}")
            st.metric("Accuracy", f"{correct/len(data['rows'])*100:.2f}%")
            st.metric("Training time", f"{data['fit_time']:.6f} s")
            st.metric("Prediction time (test batch)", f"{data['prediction_time']:.6f} s")
            st.progress(1.0, text=f"{len(data['rows'])} / {len(data['rows'])} records processed")
            st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True, height=300)

    st.subheader("Side-by-side performance")
    summary = pd.DataFrame([
        {"Model": name, "Training time (s)": v["fit_time"],
         "Prediction time (s)": v["prediction_time"],
         "Accuracy (%)": sum(r["Correct"] for r in v["rows"])/len(v["rows"])*100}
        for name,v in result.items()
    ])
    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary.set_index("Model")[["Accuracy (%)"]])
else:
    st.info("Select **Run comparison** to train both models and process the same test records.")

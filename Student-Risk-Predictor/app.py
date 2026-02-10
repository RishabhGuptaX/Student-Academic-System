import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Academic Risk Predictor",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["📊 Dashboard", "🔮 Risk Predictor", "📋 Students List", "👤 Student Details"]
)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload students.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.title("🎓 Student Academic Risk Prediction System")
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(uploaded_file)

REQUIRED = ["attendance", "half_yearly_marks", "final_marks"]
if not all(col in df.columns for col in REQUIRED):
    st.error("CSV must contain: attendance, half_yearly_marks, final_marks")
    st.stop()

# ---------------- FEATURE ENGINEERING ----------------
df["Average Marks"] = ((df["half_yearly_marks"] + df["final_marks"]) / 2).round(2)
df["marks_drop"] = df["half_yearly_marks"] - df["final_marks"]
df["low_attendance"] = (df["attendance"] < 75).astype(int)
df["poor_final"] = (df["final_marks"] < 40).astype(int)

# ---------------- TARGET CREATION ----------------
def risk_level(row):
    if row["attendance"] < 65 or row["final_marks"] < 35:
        return "HIGH"
    elif row["attendance"] < 75 or row["Average Marks"] < 50:
        return "MEDIUM"
    else:
        return "LOW"

df["Risk"] = df.apply(risk_level, axis=1)

# ---------------- MODEL TRAINING ----------------
features = [
    "attendance",
    "half_yearly_marks",
    "final_marks",
    "marks_drop",
    "low_attendance",
    "poor_final"
]

X = df[features]
le = LabelEncoder()
y = le.fit_transform(df["Risk"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- PREDICT FOR ALL STUDENTS ----------------
probs_all = model.predict_proba(X)
pred_all = model.predict(X)

df["Predicted_Risk"] = le.inverse_transform(pred_all)
df["Prob_LOW (%)"] = (probs_all[:, 0] * 100).round(2)
df["Prob_MEDIUM (%)"] = (probs_all[:, 1] * 100).round(2)
df["Prob_HIGH (%)"] = (probs_all[:, 2] * 100).round(2)

class_average = df["Average Marks"].mean().round(2)

# ===================== DASHBOARD =====================
if page == "📊 Dashboard":

    st.title("📊 Academic Risk Dashboard")
    st.success(f"🤖 Model Accuracy: {accuracy*100:.2f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", len(df))
    c2.metric("High Risk Students", (df["Predicted_Risk"] == "HIGH").sum())
    c3.metric("Low Risk Students", (df["Predicted_Risk"] == "LOW").sum())

    st.subheader("📈 Risk Distribution")
    st.bar_chart(df["Predicted_Risk"].value_counts())

    st.subheader("🤖 ML Risk Predictions")
    st.dataframe(
        df[
            [
                "attendance",
                "half_yearly_marks",
                "final_marks",
                "Predicted_Risk",
                "Prob_LOW (%)",
                "Prob_MEDIUM (%)",
                "Prob_HIGH (%)"
            ]
        ],
        use_container_width=True
    )

    st.subheader("⬇️ Download Predictions")
    csv_all = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download All Student Predictions",
        csv_all,
        "student_risk_predictions.csv",
        "text/csv"
    )

    high_risk_df = df[df["Predicted_Risk"] == "HIGH"]
    csv_high = high_risk_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "🚨 Download HIGH Risk Students Only",
        csv_high,
        "high_risk_students.csv",
        "text/csv"
    )

# ===================== RISK PREDICTOR =====================
elif page == "🔮 Risk Predictor":

    st.title("🔮 Individual Student Risk Predictor")

    col1, col2 = st.columns(2)

    with col1:
        att = st.slider("📊 Attendance (%)", 0, 100, 75)
        hy = st.slider("📝 Half-Yearly Marks", 0, 100, 65)
        fm = st.slider("📘 Final Marks", 0, 100, 60)

    with col2:
        has_backlog = st.toggle("📉 Has Backlogs?")
        poor_assignments = st.toggle("📂 Poor Assignment Performance")

    md = hy - fm
    la = int(att < 75)
    pf = int(fm < 40)

    if st.button("🔍 Predict Risk"):
        probs = model.predict_proba([[att, hy, fm, md, la, pf]])[0]
        pred = le.inverse_transform(
            model.predict([[att, hy, fm, md, la, pf]])
        )[0]

        risk_score = probs[2] * 100

        st.subheader("🚨 Risk Meter")
        st.progress(int(risk_score))

        if risk_score > 70:
            st.error(f"🔴 HIGH RISK ({risk_score:.1f}%)")
        elif risk_score > 40:
            st.warning(f"🟠 MEDIUM RISK ({risk_score:.1f}%)")
        else:
            st.success(f"🟢 LOW RISK ({risk_score:.1f}%)")

        st.subheader("📊 Probability Distribution")
        prob_df = pd.DataFrame({
            "Risk Level": ["LOW", "MEDIUM", "HIGH"],
            "Probability (%)": [p * 100 for p in probs]
        })
        st.bar_chart(prob_df.set_index("Risk Level"))

        st.subheader("🧠 Risk Factors")
        reasons = []
        if att < 75: reasons.append("⚠️ Low attendance")
        if fm < 40: reasons.append("⚠️ Low final marks")
        if md > 15: reasons.append("⚠️ Marks dropped significantly")
        if has_backlog: reasons.append("⚠️ Existing backlogs")
        if poor_assignments: reasons.append("⚠️ Poor assignment performance")

        if reasons:
            for r in reasons:
                st.write(r)
        else:
            st.success("✅ No major academic risk factors detected")

# ===================== STUDENTS LIST =====================
elif page == "📋 Students List":

    st.title("📋 Students Academic Data")
    st.dataframe(df, use_container_width=True)

# ===================== STUDENT DETAILS =====================
elif page == "👤 Student Details":

    st.title("👤 Individual Student Details")
    idx = st.selectbox("Select Student Index", df.index)
    s = df.loc[idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Attendance", s["attendance"])
    col2.metric("Final Marks", s["final_marks"])
    col3.metric("Predicted Risk", s["Predicted_Risk"])

    st.subheader("📊 Student vs Class Average")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(
        ["Student Avg", "Class Avg"],
        [s["Average Marks"], class_average]
    )
    st.pyplot(fig)

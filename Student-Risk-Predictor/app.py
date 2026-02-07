import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Academic Evaluation System", layout="wide")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["📊 Dashboard", "📋 Students List", "👤 Student Details"]
)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload students.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.title("🎓 Academic Evaluation & Debar System")
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(uploaded_file)

REQUIRED = ["attendance", "half_yearly_marks", "final_marks"]
if not all(col in df.columns for col in REQUIRED):
    st.error("CSV must contain attendance, half_yearly_marks, final_marks")
    st.stop()

# ---------------- ACADEMIC LOGIC ----------------
df["Average Marks"] = ((df["half_yearly_marks"] + df["final_marks"]) / 2).round(2)

df["Final_Debarred"] = np.where(df["attendance"] < 75, "YES", "NO")

df["Failed"] = np.where(
    (df["Final_Debarred"] == "YES") & (df["final_marks"] < 40),
    "YES",
    "NO"
)

def grade(row):
    if row["Failed"] == "YES":
        return "F"
    m = row["Average Marks"]
    if m >= 91:
        return "A"
    elif m >= 81:
        return "B"
    elif m >= 71:
        return "C"
    elif m >= 61:
        return "D"
    else:
        return "E"

df["Grade"] = df.apply(grade, axis=1)

class_average = df["Average Marks"].mean().round(2)

# ===================== DASHBOARD =====================
if page == "📊 Dashboard":

    st.title("📊 Academic Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", len(df))
    c2.metric("Failed Students", (df["Failed"] == "YES").sum())
    c3.metric("F Grade Students", (df["Grade"] == "F").sum())

    # ---- PIE: GRADE DISTRIBUTION (SMALL) ----
    st.subheader("🎓 Grade Distribution")
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    df["Grade"].value_counts().sort_index().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax1
    )
    ax1.set_ylabel("")
    st.pyplot(fig1)

    # ---- PIE: FAILED STUDENTS (SMALL) ----
    st.subheader("❌ Failed Students")
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
    df["Failed"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# ===================== STUDENTS LIST =====================
elif page == "📋 Students List":

    st.title("📋 Students Academic List")
    st.caption("Complete academic overview")

    st.dataframe(
        df.drop(columns=["Final_Debarred"]),
        use_container_width=True
    )

# ===================== STUDENT DETAILS =====================
elif page == "👤 Student Details":

    st.title("👤 Individual Student Details")

    idx = st.selectbox("Select Student Index", df.index)
    s = df.loc[idx]

    st.subheader("📘 Student Marks")

    col1, col2, col3 = st.columns(3)
    col1.metric("Half-Yearly Marks", s["half_yearly_marks"])
    col2.metric("Final Marks", s["final_marks"])
    col3.metric("Student Average", s["Average Marks"])

    st.subheader("📊 Comparison with Class Average")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(
        ["Student Avg", "Class Avg"],
        [s["Average Marks"], class_average],
        color=["#1f77b4", "#ff7f0e"]
    )
    ax.set_ylabel("Marks")
    st.pyplot(fig)

    st.subheader("📄 Result Summary")
    st.info(
        f"Grade: {s['Grade']} | "
        f"Status: {'FAILED' if s['Failed']=='YES' else 'PASSED'}"
    )

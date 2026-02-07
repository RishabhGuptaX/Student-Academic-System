Academic Evaluation System:

An interactive Streamlit-based academic analytics system designed to evaluate student performance using attendance-based debar rules, exam-wise marks analysis, grading logic, and visual dashboards. The system also supports dataset generation and downloadable individual student reports.

📌 Project Overview

Educational institutions often need transparent and rule-based systems to evaluate student performance.
This project simulates a real academic evaluation workflow, combining attendance policies, exam performance, grading, and analytics into a single dashboard.

The system is designed as a decision-support tool for colleges and universities.

✨ Key Features

📊 Interactive Dashboard

Grade distribution

Failed students overview

Academic performance insights

🧮 Academic Rules Engine

Attendance-based debar logic

Failure condition based on debar + exam performance

Automatic grading (A–F)

📋 Students List View

Complete academic data table

Calculated fields (average marks, grade, status)

👤 Individual Student Analysis

Exam marks comparison

Student vs class average performance

Clear pass/fail and grade summary

🧪 Dataset Generator

Generates realistic student data

User-defined number of students

CSV saved in project folder

📄 Student Report Export

Downloadable PDF report for each student

🛠️ Tech Stack

Language: Python

Framework: Streamlit

Libraries:

Pandas

NumPy

Matplotlib

ReportLab

📂 Project Structure
Student-Academic-System/
│
├── app.py                 # Main Streamlit application
├── generate_students.py   # CSV dataset generator
├── pdf_utils.py           # PDF report utility
├── requirements.txt       # Project dependencies
├── students.csv           # Generated dataset (optional)

▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Generate student dataset
python generate_students.py

3️⃣ Start the Streamlit app
python -m streamlit run app.py


Open in browser:

http://localhost:8501

📄 CSV Format

The system expects the following columns:

attendance,half_yearly_marks,final_marks


All evaluation logic (debar, fail, grade) is handled inside the application.

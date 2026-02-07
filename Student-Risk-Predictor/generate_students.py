import pandas as pd
import numpy as np

print("📊 Student CSV Generator")
print("------------------------")

while True:
    try:
        n = int(input("Enter number of students to generate: "))
        if n > 0:
            break
        else:
            print("Enter a number greater than 0")
    except ValueError:
        print("Please enter a valid integer")

np.random.seed(42)

data = []

for _ in range(n):
    attendance = np.random.randint(40, 101)

    half_yearly = int(np.clip(np.random.normal(attendance * 0.9, 10), 30, 100))
    final = int(np.clip(np.random.normal(attendance * 0.95, 8), 30, 100))

    data.append([attendance, half_yearly, final])

df = pd.DataFrame(
    data,
    columns=["attendance", "half_yearly_marks", "final_marks"]
)

df.to_csv("students.csv", index=False)

print("\n✅ students.csv generated successfully")
print(f"📁 Total students: {n}")
print("📄 File saved in project folder")

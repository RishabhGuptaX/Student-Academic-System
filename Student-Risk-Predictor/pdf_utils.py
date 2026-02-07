from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_student_pdf(student, filename="student_report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Student Academic Report")

    c.setFont("Helvetica", 12)
    y = height - 100

    for key, value in student.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()

# Dictionary operations
student_grades = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78,
    "Diana": 96
}

# Find average grade
average = sum(student_grades.values()) / len(student_grades)
print(f"Average grade: {average:.2f}")

# Find top student
top_student = max(student_grades, key=student_grades.get)
print(f"Top student: {top_student} with {student_grades[top_student]}")
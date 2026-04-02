import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load models
career_model = joblib.load("career_model.pkl")
selection_model = joblib.load("selection_model.pkl")
mlb = joblib.load("mlb.pkl")
le = joblib.load("label_encoder.pkl")
exp_encoder = joblib.load("exp_encoder.pkl")

st.title("🎯 Student Skill Gap Analyzer")

# Input
skills_input = st.text_input("Enter your skills (comma separated)")
experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])

# Suggestions
suggestions_dict = {
    "python": "Learn Pandas, NumPy, Projects",
    "machine learning": "Take Andrew Ng Course",
    "sql": "Practice queries",
    "react": "Build projects",
    "java": "Learn Spring Boot"
}

# Courses
courses_dict = {
    "machine learning": "Andrew Ng - Coursera",
    "python": "FreeCodeCamp YouTube"
}

def plot_chart(matched, missing):
    labels = ['Matched', 'Missing']
    values = [len(matched), len(missing)]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

if st.button("Analyze"):

    user_skills = [s.strip().lower() for s in skills_input.split(",")]

    # Convert to ML format
    X_skills = mlb.transform([user_skills])
    exp = exp_encoder.transform([experience])

    X_input = np.hstack((X_skills, [[exp[0]]]))

    # Predictions
    role = le.inverse_transform(career_model.predict(X_input))[0]
    prob = selection_model.predict_proba(X_input)[0][1]

    # Output
    st.subheader("🏆 Career Recommendation")
    st.success(role)

    st.subheader("📈 Selection Chance")
    st.write(f"{prob*100:.2f}%")

    # Score meter
    if prob < 0.4:
        level = "Beginner"
    elif prob < 0.7:
        level = "Intermediate"
    else:
        level = "Advanced"

    st.write(f"Level: {level}")
    st.progress(int(prob * 100))

    # Skill analysis
    required_skills = list(mlb.classes_)

    matched = set(user_skills).intersection(required_skills)
    missing = set(required_skills) - set(user_skills)

    st.subheader("📊 Skill Analysis")
    st.write("Matched:", list(matched))
    st.write("Missing:", list(missing))

    plot_chart(matched, missing)

    # Suggestions
    st.subheader("💡 Suggestions")
    for skill in missing:
        tip = suggestions_dict.get(skill, f"Learn {skill}")
        st.write(f"{skill} → {tip}")

    # Courses
    st.subheader("📚 Course Recommendations")
    for skill in missing:
        course = courses_dict.get(skill, "Search online")
        st.write(f"{skill} → {course}")

    # Roadmap
    st.subheader("📅 Learning Roadmap")
    for i, skill in enumerate(missing):
        st.write(f"Week {i+1}: Learn {skill}")

    # Internship readiness
    st.subheader("🧑‍💼 Internship Readiness")
    st.write(f"{prob*100:.2f}% ready")
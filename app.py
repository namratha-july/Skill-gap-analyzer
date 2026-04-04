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


skills_input = st.text_input("Enter your skills (comma separated)")
experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])

#  ROLE SKILLS 

role_skills_map = {
    "Data Scientist": ["python", "pandas", "numpy", "machine learning", "statistics", "sql"],
    "Web Developer": ["html", "css", "javascript"],
    "Backend Developer": ["java", "spring", "sql", "rest api"],
    "Frontend Developer": ["html", "css", "javascript", "react", "redux"],
    "AI Engineer": ["python", "deep learning", "nlp", "tensorflow"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau"],
    "ML Engineer": ["python", "machine learning", "scikit-learn", "tensorflow"]
}

#  SUGGESTIONS 

suggestions_dict = {
    "python": "Learn Pandas, NumPy, Projects",
    "machine learning": "Take Andrew Ng Course",
    "sql": "Practice queries",
    "react": "Build projects",
    "java": "Learn Spring Boot"
}

#  COURSES 

courses_dict = {
    "machine learning": "Andrew Ng - Coursera",
    "python": "FreeCodeCamp YouTube"
}

#  CHART 
def plot_chart(matched, missing):
    labels = ['Matched', 'Missing']
    values = [len(matched), len(missing)]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

#  MAIN 

if st.button("Analyze"):

    user_skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

    if not user_skills:
        st.warning("Please enter at least one skill")

    else:
 # HYBRID CAREER PREDICTION 

        role_scores = {}

        for r, skills in role_skills_map.items():
            match_count = len(set(user_skills).intersection(skills))
            role_scores[r] = match_count

        # Best matching role
        role = max(role_scores, key=role_scores.get)

        #  ML FOR SELECTION 

        X_skills = mlb.transform([user_skills])
        exp = exp_encoder.transform([experience])
        X_input = np.hstack((X_skills, [[exp[0]]]))

        prob = selection_model.predict_proba(X_input)[0][1]

        #  OUTPUT

        st.subheader("🏆 Career Recommendation")
        st.success(role)

        st.subheader("📈 Selection Chance")
        st.write(f"{prob*100:.2f}%")

        # Level
        if prob < 0.4:
            level = "Beginner"
        elif prob < 0.7:
            level = "Intermediate"
        else:
            level = "Advanced"

        st.write(f"Level: {level}")
        st.progress(int(prob * 100))

        #  SKILL ANALYSIS

        required_skills = role_skills_map.get(role, [])

        matched = sorted(set(user_skills).intersection(required_skills))
        missing = sorted(set(required_skills) - set(user_skills))

        st.subheader("📊 Skill Analysis")
        st.write("✅ Matched Skills:", matched)
        st.write("❌ Missing Skills:", missing)

        plot_chart(matched, missing)

        #  SUGGESTIONS 

        st.subheader("💡 Suggestions")
        for skill in missing:
            tip = suggestions_dict.get(skill, f"Learn {skill}")
            st.write(f"{skill} → {tip}")

        #  COURSES 

        st.subheader("📚 Course Recommendations")
        for skill in missing:
            course = courses_dict.get(skill, "Search online")
            st.write(f"{skill} → {course}")

        #  ROADMAP 

        st.subheader("📅 Learning Roadmap")
        for i, skill in enumerate(missing):
            st.write(f"Week {i+1}: Learn {skill}")

        #  INTERNSHIP READINESS 

        st.subheader("🧑‍💼 Internship Readiness")
        st.write(f"{prob*100:.2f}% ready")

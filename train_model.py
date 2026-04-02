import pandas as pd
import joblib
import os
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# FIX PATH PROPERLY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "C:\\vs\\Skill gap analyzer\\asserts\\dataset.csv")

df = pd.read_csv(DATA_PATH)

# Debug (optional - remove later)
print("Looking for dataset at:", DATA_PATH)

# Check if file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f" dataset.csv not found at: {DATA_PATH}")

# Load dataset
df = pd.read_csv("C:\\vs\\Skill gap analyzer\\asserts\\dataset.csv")


# DATA PREPROCESSING


# Convert skills to list
df['skills'] = df['skills'].apply(lambda x: str(x).split(";"))

# Encode skills
mlb = MultiLabelBinarizer()
X_skills = mlb.fit_transform(df['skills'])

# Encode experience level
exp_encoder = LabelEncoder()
df['experience_level'] = exp_encoder.fit_transform(df['experience_level'])

# Combine features
X = np.hstack((X_skills, df[['experience_level']].values))

# Encode career labels
le = LabelEncoder()
y = le.fit_transform(df['career_label'])

# Encode selection status
df['selection_status'] = df['selection_status'].map({
    "Selected": 1,
    "Not Selected": 0
})


# TRAIN MODELS

career_model = RandomForestClassifier()
career_model.fit(X, y)

selection_model = RandomForestClassifier()
selection_model.fit(X, df['selection_status'])


# SAVE MODELS (same folder)


joblib.dump(career_model, os.path.join(BASE_DIR, "career_model.pkl"))
joblib.dump(selection_model, os.path.join(BASE_DIR, "selection_model.pkl"))
joblib.dump(mlb, os.path.join(BASE_DIR, "mlb.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
joblib.dump(exp_encoder, os.path.join(BASE_DIR, "exp_encoder.pkl"))

print("✅ Models trained successfully!")
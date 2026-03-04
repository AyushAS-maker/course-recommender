import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("coursera.csv")# Load the data
df = pd.read_csv('coursera.csv')

# Add this line to fix the titles automatically! ✨
df = df.rename(columns={'title': 'course_title', 'category-subject-area': 'skills'})

# Now your original code will work perfectly:
df["combined"] = df["course_title"] + " " + df["skills"] + " " + df["description"]

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Print columns for debugging (you can remove later)
print(df.columns)

# ---- CHANGE THESE BASED ON YOUR DATASET ----
# Common Coursera dataset columns:
# course_name
# course_description
# skills

df["combined"] = (
    df["course_name"].fillna('') + " " +
    df["skills"].fillna('') + " " +
    df["course_description"].fillna('')
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(course_name):
    idx = df[df["course_name"] == course_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    course_indices = [i[0] for i in sim_scores]
    return df["course_name الدور"].iloc[course_indices]

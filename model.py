import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("coursera.csv")

df["combined"] = df["course_title"] + " " + df["skills"] + " " + df["description"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(course_name):
    idx = df[df["course_title"] == course_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    course_indices = [i[0] for i in sim_scores]
    return df["course_title"].iloc[course_indices]

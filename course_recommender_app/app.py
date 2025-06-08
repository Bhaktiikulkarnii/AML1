import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data/EdX1.csv")

# Fill missing values
df.fillna("", inplace=True)

# Combine features into a single string
df["content"] = (
    df["Name"] + " " +
    df["University"] + " " +
    df["Difficulty Level"] + " " +
    df["Course Description"] + " " +
    df["About"]
)

# Vectorize the content
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["content"])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit UI
st.set_page_config(page_title="Course Recommender", layout="centered")
st.title("🎓 EdX Course Recommendation System")

# Dropdown to select a course
selected_course = st.selectbox("Select a course you liked:", df["Name"].values)

if selected_course:
    # Find index of selected course
    idx = df[df["Name"] == selected_course].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations

    st.subheader("🔍 Recommended Courses:")
    for i, score in sim_scores:
        st.markdown(f"**📘 {df.iloc[i]['Name']}**")
        st.markdown(f"🏫 {df.iloc[i]['University']}")
        st.markdown(f"🎯 Difficulty: {df.iloc[i]['Difficulty Level']}")
        st.markdown(f"🔗 [View Course]({df.iloc[i]['Link']})")
        st.markdown("---")

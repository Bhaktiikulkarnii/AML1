import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI
st.set_page_config(page_title="Course Recommender", layout="centered")
st.title("🎓 EdX Course Recommendation System")

# File uploader widget
uploaded_file = st.file_uploader("Upload your course data CSV file", type="csv")

if uploaded_file:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Check for necessary columns
        required_columns = ["Name", "University", "Difficulty Level", "Course Description", "About", "Link"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required columns: {', '.join(set(required_columns) - set(df.columns))}")
        else:
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
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
    

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
df = pd.read_csv("amazon_machine_learning_books.csv")

# Clean data
df['Title'] = df['Title'].fillna('').astype(str).str.lower().str.strip()
df['Authors'] = df['Authors'].fillna('').astype(str).str.lower().str.strip()

# ✅ Combine text fields
df['content'] = df['Title'] + " " + df['Authors']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['content'])

# Cosine similarity between books
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_input, df, vectorizer, tfidf_matrix, cosine_sim):
    user_input = user_input.lower().strip()

    # Convert user input to TF-IDF vector
    user_vec = vectorizer.transform([user_input])

    # Find best matching book
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_match_idx = similarity_scores.argmax()

    if similarity_scores[best_match_idx] < 0.1:
        return "No similar book found"

    # Get similar books to best match
    sim_scores = list(enumerate(cosine_sim[best_match_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    book_indices = [i[0] for i in sim_scores]

    return df.loc[book_indices, ['Title', 'Authors']]

st.title("📚 Book Recommendation Engine")
st.write("Enter a book title and get similar recommendations")

select_book = st.text_input("Book Title")

if select_book:
    result = get_recommendations(
        select_book,
        df,
        vectorizer,
        tfidf_matrix,
        cosine_sim
    )

    if isinstance(result, pd.DataFrame):
        st.table(result)
    else:
        st.warning(result)

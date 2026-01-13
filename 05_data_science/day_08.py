import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("amazon_machine_learning_books.csv")

vectorizer = TfidfVectorizer(stop_words='english')
tfid_matrix = vectorizer.fit_transform(df['Title'])

cosine_sim = cosine_similarity(tfid_matrix, tfid_matrix)

indices = pd.Series(df.index, index=df['Title'])

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    boox_indices = [i[0] for i in sim_scores]
    return df['Title','Authors'].iloc[boox_indices]
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("youtube_comments.csv")

x_train, x_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(f"Model trained. Accuracy: {round(acc * 100, 2)}%")
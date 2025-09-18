# -----------------------------------------
# Feature Extraction: BoW & TF-IDF
# -----------------------------------------

# Step 1: Import Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Step 2: Example Dataset (Replace with movie reviews, tweets, etc.)
corpus = [
    "I love this movie, it was fantastic!",
    "The movie was terrible and boring.",
    "What a great and thrilling film!",
    "I did not like the movie, it was dull.",
    "Amazing acting and story, I enjoyed it."
]

# Step 3: Bag of Words (BoW)
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(corpus)

df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
print("Bag-of-Words Representation:\n")
print(df_bow)

# Step 4: Term Frequency - Inverse Document Frequency (TF-IDF)
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(corpus)

df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
print("\nTF-IDF Representation:\n")
print(df_tfidf.round(2))

# Step 5: Save results (Optional)
df_bow.to_csv("bow_output.csv", index=False)
df_tfidf.to_csv("tfidf_output.csv", index=False)

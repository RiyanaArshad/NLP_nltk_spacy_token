# text_preprocessing_nltk_spacy.py

# Import libraries
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# Example text corpus
# ------------------------------
corpus = [
    "Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence.",
    "Social media posts often contain slang and abbreviations that make NLP challenging.",
    "News articles provide structured text data for topic modeling and analysis."
]

print("=== Original Corpus ===")
for doc in corpus:
    print(doc)

# ------------------------------
# Tokenization
# ------------------------------
print("\n=== Tokenization (NLTK) ===")
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
for tokens in tokenized_corpus:
    print(tokens)

# ------------------------------
# Stopword Removal
# ------------------------------
print("\n=== Stopword Removal (NLTK) ===")
stop_words = set(stopwords.words("english"))
filtered_corpus = [[word for word in tokens if word.isalpha() and word not in stop_words] for tokens in tokenized_corpus]
for words in filtered_corpus:
    print(words)

# ------------------------------
# Stemming
# ------------------------------
print("\n=== Stemming (PorterStemmer, NLTK) ===")
stemmer = PorterStemmer()
stemmed_corpus = [[stemmer.stem(word) for word in words] for words in filtered_corpus]
for stems in stemmed_corpus:
    print(stems)

# ------------------------------
# Lemmatization
# ------------------------------
print("\n=== Lemmatization (SpaCy) ===")
lemmatized_corpus = []
for doc in corpus:
    spacy_doc = nlp(doc.lower())
    lemmatized_corpus.append([token.lemma_ for token in spacy_doc if token.is_alpha and not token.is_stop])

for lemmas in lemmatized_corpus:
    print(lemmas)

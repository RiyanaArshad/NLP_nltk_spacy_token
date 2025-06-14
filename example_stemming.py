# File: preprocess_stemming.py

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download necessary stuff
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------- 🔹 NLTK Preprocessing 🔹 -----------

text = input("Enter a sentence: ")

# Tokenize the sentence
tokens_nltk = word_tokenize(text.lower())

# Remove stopwords
filtered_nltk = [word for word in tokens_nltk if word not in stop_words and word.isalpha()]

# Apply stemming
stemmed_nltk = [stemmer.stem(word) for word in filtered_nltk]

print("\n🔹 NLTK Stemming Output:")
print("Original Tokens:", tokens_nltk)
print("After Stemming:", stemmed_nltk)

# ----------- 🔹 spaCy Preprocessing (with NLTK stemmer) 🔹 -----------

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
doc = nlp(text.lower())

# spaCy tokenization + stemming with NLTK
tokens_spacy = [token.text for token in doc if not token.is_stop and token.is_alpha]
stemmed_spacy = [stemmer.stem(token) for token in tokens_spacy]

print("\n🔹 spaCy + NLTK Stemming Output:")
print("Original Tokens:", tokens_spacy)
print("After Stemming:", stemmed_spacy)

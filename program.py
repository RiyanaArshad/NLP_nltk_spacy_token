import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data files (run only once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Input sentence
text = input("Enter a sentence: ")

# Tokenize the sentence
words = word_tokenize(text.lower())

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

# Count word frequency
word_counts = Counter(filtered_words)

# Display result
print("\nWord Frequencies (excluding stopwords):")
for word, count in word_counts.items():
    print(f"{word}: {count}")

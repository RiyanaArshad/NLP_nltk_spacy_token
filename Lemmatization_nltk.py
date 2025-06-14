from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

text = input("Enter a sentence: ")
words = word_tokenize(text)

lemmatized = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized Words:", lemmatized)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

text = input("Enter a sentence: ")
words = word_tokenize(text)

stemmed = [stemmer.stem(word) for word in words]
print("Stemmed Words:", stemmed)

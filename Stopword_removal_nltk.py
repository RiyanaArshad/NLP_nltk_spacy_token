from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')

text = input("Enter a sentence: ")
words = word_tokenize(text)

filtered = [word for word in words if word.lower() not in stopwords.words('english')]
print("Filtered Words:", filtered)

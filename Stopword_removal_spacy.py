import spacy

nlp = spacy.load("en_core_web_sm")

text = input("Enter a sentence: ")
doc = nlp(text)

filtered = [token.text for token in doc if not token.is_stop]
print("Filtered Words:", filtered)

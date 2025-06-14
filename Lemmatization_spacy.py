import spacy

nlp = spacy.load("en_core_web_sm")

text = input("Enter a sentence: ")
doc = nlp(text)

lemmatized = [token.lemma_ for token in doc]
print("Lemmatized Words:", lemmatized)

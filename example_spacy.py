import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Input text
text = input("Enter a sentence: ")

# Process the text
doc = nlp(text)

print("\n--- Named Entities Found ---")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")

# -----------------------------------------
# Named Entity Recognition (NER) with spaCy
# -----------------------------------------

import spacy
import pandas as pd

# -----------------------------------------
# Step 1: Load pre-trained spaCy model
# -----------------------------------------
# Download the model once with: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# -----------------------------------------
# Step 2: Example dataset (unstructured text)
# Replace this with your dataset: job postings, scientific articles, etc.
# -----------------------------------------
documents = [
    "John Doe is a data scientist at Microsoft Research.",
    "Dr. Smith published a paper on Machine Learning at Stanford University.",
    "Alice Johnson works at Google as a software engineer.",
    "Professor Brown from Harvard collaborated with IBM Research."
]

# -----------------------------------------
# Step 3: Apply NER model
# -----------------------------------------
extracted_data = []

for doc in documents:
    spacy_doc = nlp(doc)
    person = None
    org = None
    
    for ent in spacy_doc.ents:
        if ent.label_ == "PERSON":
            person = ent.text
        elif ent.label_ == "ORG":
            org = ent.text
    
    extracted_data.append({"Person": person, "Organization": org})

# -----------------------------------------
# Step 4: Create structured table
# -----------------------------------------
df = pd.DataFrame(extracted_data)
print("Extracted Structured Information:\n")
print(df)

# -----------------------------------------
# Step 5: Save results to CSV (optional for GitHub)
# -----------------------------------------
df.to_csv("ner_results.csv", index=False)

# -----------------------------------------
# Topic Modeling with LDA using Gensim
# -----------------------------------------

# Step 1: Import required libraries
import nltk
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download stopwords (only once)
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------------------
# Step 2: Example text corpus
# (You can replace this with your dataset: news articles, research papers, etc.)
# -----------------------------------------
corpus = [
    "Artificial Intelligence is transforming the healthcare industry with faster diagnosis.",
    "Machine Learning algorithms are applied in finance for fraud detection.",
    "Natural Language Processing helps in building intelligent chatbots.",
    "Sports analytics use data science to improve player performance.",
    "Climate change is studied using large amounts of data and predictive models.",
    "Deep Learning models are widely used in image recognition applications."
]

# -----------------------------------------
# Step 3: Preprocessing
# - Tokenization
# - Stopword removal
# - Lowercasing
# -----------------------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())  # tokenize and convert to lowercase
    tokens = [t for t in tokens if t.isalpha()]  # keep only words (remove punctuation/numbers)
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    return tokens

processed_corpus = [preprocess(doc) for doc in corpus]

print("Sample Preprocessed Document:", processed_corpus[0])

# -----------------------------------------
# Step 4: Create Dictionary and Corpus
# -----------------------------------------
dictionary = corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

# -----------------------------------------
# Step 5: Apply LDA Model
# -----------------------------------------
lda_model = gensim.models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=3,   # choose number of topics
    random_state=42,
    passes=15,
    per_word_topics=True
)

# Display the topics
print("\nLDA Model Topics:")
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")

# -----------------------------------------
# Step 6: Visualization with WordCloud
# -----------------------------------------
for i in range(3):
    plt.figure(figsize=(6, 4))
    plt.title(f"Topic {i}")
    words = dict(lda_model.show_topic(i, 20))  # top 20 words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

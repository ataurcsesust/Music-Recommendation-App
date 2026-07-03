import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Load dataset
df = pd.read_csv("spotify_millsongdata.csv")

# Dataset বড় হলে sample নাও
df = df.sample(5000, random_state=42).reset_index(drop=True)

df = df.drop(columns=["link"], errors="ignore")

df["cleaned_text"] = df["text"].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=5000)

tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

def recommend_songs(song_name, top_n=5):

    idx = df[df["song"].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        return None

    idx = idx[0]

    similarity = linear_kernel(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    scores = list(enumerate(similarity))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    indices = [i[0] for i in scores]

    result = df[["artist", "song"]].iloc[indices].reset_index(drop=True)

    result.index += 1
    result.index.name = "S.No."

    return result
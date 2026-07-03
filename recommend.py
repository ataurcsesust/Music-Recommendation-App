import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Download only if missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()

    # No NLTK tokenizer needed
    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Load dataset
df = pd.read_csv("spotify_millsongdata.csv")

# Sample 5000 songs
df = df.sample(5000, random_state=42).reset_index(drop=True)

df = df.drop(columns=["link"], errors="ignore")

df["cleaned_text"] = df["text"].apply(preprocess_text)

# TF-IDF
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

    scores = sorted(
        enumerate(similarity),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    song_indices = [i[0] for i in scores]

    result = (
        df[["artist", "song"]]
        .iloc[song_indices]
        .reset_index(drop=True)
    )

    result.index += 1
    result.index.name = "S.No."

    return result
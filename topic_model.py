import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

from nlp_utils import clean_text


# ==========================================
# LOAD SOCIAL MEDIA DATA
# ==========================================

df = pd.read_csv("data/social_media.csv")

df["text"] = df["text"].fillna("")
df["clean_text"] = df["text"].apply(clean_text)


# ==========================================
# CUSTOM STOP WORDS
# ==========================================

# "AI" appears in almost every post.
# Removing it helps LDA discover more meaningful topics.

custom_stop_words = list(
    ENGLISH_STOP_WORDS.union({
        "ai"
    })
)


# ==========================================
# CREATE DOCUMENT-TERM MATRIX
# ==========================================

vectorizer = CountVectorizer(
    stop_words=custom_stop_words,
    max_features=1000,
    min_df=1,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["clean_text"])


# ==========================================
# LDA TOPIC MODEL
# ==========================================

lda = LatentDirichletAllocation(
    n_components=3,
    random_state=42,
    learning_method="batch"
)

lda.fit(X)


# ==========================================
# EXTRACT TOPIC KEYWORDS
# ==========================================

feature_names = vectorizer.get_feature_names_out()

topic_keywords = {}

print("\n==============================")
print("DETECTED TOPICS")
print("==============================")


for topic_idx, topic in enumerate(lda.components_):

    top_words = [
        feature_names[i]
        for i in topic.argsort()[-8:][::-1]
    ]

    topic_id = topic_idx + 1

    topic_keywords[topic_id] = top_words

    print(f"\nTopic {topic_id}:")
    print(" • ".join(top_words))


# ==========================================
# ASSIGN DOMINANT TOPIC TO EACH POST
# ==========================================

topic_probabilities = lda.transform(X)

df["topic_id"] = topic_probabilities.argmax(axis=1) + 1


# ==========================================
# HUMAN-READABLE TOPIC NAMES
# ==========================================

topic_names = {
    1: "AI Automation & Productivity",
    2: "ChatGPT, Coding & Education",
    3: "AI, Jobs & Future Technology"
}

df["topic_label"] = df["topic_id"].map(topic_names)


# ==========================================
# TOPIC CONFIDENCE
# ==========================================

df["topic_confidence"] = topic_probabilities.max(axis=1)


# ==========================================
# ENGAGEMENT
# ==========================================

df["likes"] = pd.to_numeric(
    df["likes"],
    errors="coerce"
).fillna(0)

df["shares"] = pd.to_numeric(
    df["shares"],
    errors="coerce"
).fillna(0)

df["engagement"] = df["likes"] + df["shares"]


# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(
    lda,
    "topic_model.pkl"
)

joblib.dump(
    vectorizer,
    "topic_vectorizer.pkl"
)


# ==========================================
# SAVE ANALYZED DATASET
# ==========================================

df.to_csv(
    "data/topic_analyzed_social_media.csv",
    index=False
)


# ==========================================
# DISPLAY SUMMARY
# ==========================================

print("\n==============================")
print("TOPIC MODELING COMPLETE")
print("==============================")

print("\nTopic Distribution:")

print(
    df["topic_label"]
    .value_counts()
)

print("\nSaved Files:")

print("✓ topic_model.pkl")
print("✓ topic_vectorizer.pkl")
print("✓ data/topic_analyzed_social_media.csv")

print("\nTopic Keywords:")

for topic_id, words in topic_keywords.items():

    print(
        f"\n{topic_names[topic_id]}:"
    )

    print(
        " • ".join(words)
    )
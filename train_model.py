import pandas as pd
import joblib

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from nlp_utils import clean_text


print("========================================")
print("   TRENDSENSE AI - SENTIMENT TRAINING")
print("========================================")


# =========================================================
# 1. DOWNLOAD SOCIAL MEDIA SENTIMENT DATASET
# =========================================================

print("\nDownloading TweetEval sentiment dataset...")

dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "sentiment"
)

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])


# =========================================================
# 2. CONVERT LABELS
# =========================================================

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

train_df["sentiment"] = train_df["label"].map(label_map)
test_df["sentiment"] = test_df["label"].map(label_map)


train_df = train_df[["text", "sentiment"]]
test_df = test_df[["text", "sentiment"]]


# =========================================================
# 3. SAVE DATASET
# =========================================================

train_df.to_csv(
    "data/sentiment_train.csv",
    index=False
)

test_df.to_csv(
    "data/sentiment_test.csv",
    index=False
)


print("\nDataset downloaded successfully!")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

print("\nTraining sentiment distribution:")
print(train_df["sentiment"].value_counts())


# =========================================================
# 4. CLEAN TEXT
# =========================================================

print("\nCleaning text...")

train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# =========================================================
# 5. TF-IDF
# =========================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


X_train = vectorizer.fit_transform(
    train_df["clean_text"]
)

X_test = vectorizer.transform(
    test_df["clean_text"]
)


y_train = train_df["sentiment"]
y_test = test_df["sentiment"]


print(f"Training features: {X_train.shape}")
print(f"Testing features: {X_test.shape}")


# =========================================================
# 6. TRAIN MODEL
# =========================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# 7. EVALUATION
# =========================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========================================")
print("         MODEL PERFORMANCE")
print("========================================")

print(
    f"\nAccuracy: {accuracy:.2%}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 8. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


print("\n========================================")
print("       MODEL SAVED SUCCESSFULLY")
print("========================================")

print("\nFiles created:")

print("✅ sentiment_model.pkl")
print("✅ tfidf_vectorizer.pkl")

print("\nTrendSense sentiment model is ready!")
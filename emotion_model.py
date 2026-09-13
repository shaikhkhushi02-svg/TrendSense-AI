import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from datasets import load_dataset

from nlp_utils import clean_text


print("\n====================================")
print("TREND SENSE AI - EMOTION MODEL")
print("====================================")


# ------------------------------------
# 1. LOAD EMOTION DATASET
# ------------------------------------

print("\nLoading emotion dataset...")

dataset = load_dataset(
    "dair-ai/emotion"
)

train_df = pd.DataFrame(
    dataset["train"]
)

test_df = pd.DataFrame(
    dataset["test"]
)

print(
    f"Training samples: {len(train_df):,}"
)

print(
    f"Testing samples: {len(test_df):,}"
)


# ------------------------------------
# 2. EMOTION LABELS
# ------------------------------------

emotion_names = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

train_df["emotion"] = train_df[
    "label"
].map(emotion_names)

test_df["emotion"] = test_df[
    "label"
].map(emotion_names)


# ------------------------------------
# 3. CLEAN TEXT
# ------------------------------------

print("\nCleaning text...")

train_df["clean_text"] = train_df[
    "text"
].apply(clean_text)

test_df["clean_text"] = test_df[
    "text"
].apply(clean_text)


# ------------------------------------
# 4. TF-IDF
# ------------------------------------

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

y_train = train_df["emotion"]
y_test = test_df["emotion"]


print(
    f"TF-IDF features: {X_train.shape[1]:,}"
)


# ------------------------------------
# 5. TRAIN MODEL
# ------------------------------------

print("\nTraining emotion classifier...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------
# 6. EVALUATE MODEL
# ------------------------------------

accuracy = model.score(
    X_test,
    y_test
)

print("\n====================================")
print("EMOTION MODEL RESULTS")
print("====================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ------------------------------------
# 7. SAVE MODEL
# ------------------------------------

joblib.dump(
    model,
    "emotion_model.pkl"
)

joblib.dump(
    vectorizer,
    "emotion_vectorizer.pkl"
)


# ------------------------------------
# 8. SAVE DATASETS
# ------------------------------------

train_df[
    ["text", "emotion"]
].to_csv(
    "data/emotion_train.csv",
    index=False
)

test_df[
    ["text", "emotion"]
].to_csv(
    "data/emotion_test.csv",
    index=False
)


# ------------------------------------
# 9. TEST PREDICTIONS
# ------------------------------------

sample_posts = [
    "I am extremely happy today!",
    "I am so angry about this!",
    "I am worried about losing my job.",
    "This is the worst day ever.",
    "I really love this new technology!",
    "Wow! I did not expect this!"
]


print("\n====================================")
print("SAMPLE EMOTION PREDICTIONS")
print("====================================")


sample_clean = [
    clean_text(text)
    for text in sample_posts
]

sample_features = vectorizer.transform(
    sample_clean
)

predictions = model.predict(
    sample_features
)

probabilities = model.predict_proba(
    sample_features
)


for text, prediction, probability in zip(
    sample_posts,
    predictions,
    probabilities
):

    confidence = probability.max() * 100

    print(
        f"\nPost: {text}"
    )

    print(
        f"Emotion: {prediction}"
    )

    print(
        f"Confidence: {confidence:.1f}%"
    )


print("\n====================================")
print("EMOTION MODEL TRAINING COMPLETE")
print("====================================")

print("\nCreated files:")

print("✓ emotion_model.pkl")
print("✓ emotion_vectorizer.pkl")
print("✓ data/emotion_train.csv")
print("✓ data/emotion_test.csv")
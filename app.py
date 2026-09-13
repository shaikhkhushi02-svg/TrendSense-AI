import os
import io
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrendSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "social_media.csv")

SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
SENTIMENT_VECTOR_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
SENTIMENT_TEST_PATH = os.path.join(
    BASE_DIR, "data", "sentiment_test.csv"
)

EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.pkl")
EMOTION_VECTOR_PATH = os.path.join(BASE_DIR, "emotion_vectorizer.pkl")
EMOTION_TEST_PATH = os.path.join(
    BASE_DIR, "data", "emotion_test.csv"
)

TOPIC_MODEL_PATH = os.path.join(BASE_DIR, "topic_model.pkl")
TOPIC_VECTOR_PATH = os.path.join(BASE_DIR, "topic_vectorizer.pkl")

# ============================================================
# HTML HELPER
# ============================================================

def render_html(html):
    try:
        st.html(html)
    except Exception:
        st.markdown(html, unsafe_allow_html=True)


# ============================================================
# PREMIUM DARK CSS
# ============================================================

render_html("""
<style>

.stApp {
    background: #050505;
    color: #f5f5f5;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #080808;
    border-right: 1px solid #222;
}

section[data-testid="stSidebar"] * {
    color: #eeeeee;
}

.hero {
    padding: 30px;
    border: 1px solid #292929;
    border-radius: 22px;
    background: linear-gradient(135deg, #111111, #080808);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 43px;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.hero-sub {
    color: #a5a5a5;
    font-size: 16px;
    margin-top: 8px;
    line-height: 1.6;
}

.section-title {
    font-size: 26px;
    font-weight: 750;
    margin: 28px 0 8px;
}

.section-sub {
    color: #999;
    margin-bottom: 18px;
}

.kpi {
    background: #101010;
    border: 1px solid #292929;
    border-radius: 18px;
    padding: 20px;
    min-height: 120px;
}

.kpi-label {
    color: #8f8f8f;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.kpi-value {
    color: #fff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #777;
    font-size: 11px;
    margin-top: 5px;
}

.card {
    background: #0e0e0e;
    border: 1px solid #292929;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 14px;
}

.card-title {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 7px;
}

.card-text {
    color: #a5a5a5;
    line-height: 1.55;
}

.insight {
    background: #111;
    border-left: 3px solid #eee;
    border-radius: 12px;
    padding: 15px 17px;
    margin: 12px 0;
}

.warning {
    background: #151515;
    border: 1px solid #404040;
    border-radius: 14px;
    padding: 14px 16px;
    color: #d0d0d0;
    margin: 12px 0;
}

.success {
    background: #111;
    border: 1px solid #353535;
    border-radius: 14px;
    padding: 14px 16px;
    color: #ddd;
    margin: 12px 0;
}

.footer {
    color: #666;
    text-align: center;
    padding: 35px 0 10px;
    font-size: 12px;
}

</style>
""")

# ============================================================
# PLOTLY THEME
# ============================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="#eeeeee"),
    margin=dict(l=20, r=20, t=45, b=20),
)


# ============================================================
# LABEL MAPPINGS
# ============================================================

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}

EMOTION_LABELS = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
    "0": "Sadness",
    "1": "Joy",
    "2": "Love",
    "3": "Anger",
    "4": "Fear",
    "5": "Surprise",
    "sadness": "Sadness",
    "joy": "Joy",
    "love": "Love",
    "anger": "Anger",
    "fear": "Fear",
    "surprise": "Surprise",
}

SENTIMENT_ORDER = [
    "Negative",
    "Neutral",
    "Positive"
]

EMOTION_ORDER = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_label(value, mapping):
    if isinstance(value, (int, np.integer)):
        return mapping.get(int(value), "Unknown")

    value = str(value).strip()

    if value in mapping:
        return mapping[value]

    value_lower = value.lower()

    if value_lower in mapping:
        return mapping[value_lower]

    try:
        return mapping.get(int(float(value)), "Unknown")
    except Exception:
        return value.title() if value else "Unknown"


def normalize_sentiment(value):
    return normalize_label(value, SENTIMENT_LABELS)


def normalize_emotion(value):
    return normalize_label(value, EMOTION_LABELS)


def compact_number(value):
    try:
        value = float(value)

        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"

        if value >= 1_000:
            return f"{value / 1_000:.1f}K"

        return f"{int(value):,}"

    except Exception:
        return "0"


def detect_column(df, candidates):
    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for column in df.columns:
        for candidate in candidates:
            if candidate.lower() in str(column).lower():
                return column

    return None


def safe_probability(model, X):
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            return np.max(probabilities, axis=1)
    except Exception:
        pass

    return np.full(len(X), np.nan)


def extract_hashtags(text):
    return [
        tag.lower()
        for tag in re.findall(r"#\w+", str(text))
    ]


def kpi_card(label, value, note=""):
    return f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


# ============================================================
# NLP CLEANING
# ============================================================

def fallback_clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return " ".join(
        word
        for word in text.split()
        if word not in ENGLISH_STOP_WORDS
    )


try:
    from nlp_utils import clean_text
except Exception:
    clean_text = fallback_clean_text


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    paths = {
        "sentiment_model": SENTIMENT_MODEL_PATH,
        "sentiment_vectorizer": SENTIMENT_VECTOR_PATH,
        "emotion_model": EMOTION_MODEL_PATH,
        "emotion_vectorizer": EMOTION_VECTOR_PATH,
        "topic_model": TOPIC_MODEL_PATH,
        "topic_vectorizer": TOPIC_VECTOR_PATH,
    }

    loaded = {}

    for name, path in paths.items():

        if os.path.exists(path):

            try:
                loaded[name] = joblib.load(path)
            except Exception:
                loaded[name] = None

        else:
            loaded[name] = None

    return loaded


models = load_models()


# ============================================================
# TEST DATA EVALUATION
# ============================================================

def get_file_signature(path):

    try:
        return os.path.getmtime(path)
    except Exception:
        return 0


def detect_test_columns(df):

    text_column = detect_column(
        df,
        [
            "text",
            "tweet",
            "sentence",
            "content",
            "comment",
            "post"
        ]
    )

    label_column = detect_column(
        df,
        [
            "label",
            "sentiment",
            "emotion",
            "target",
            "class"
        ]
    )

    return text_column, label_column


@st.cache_data(show_spinner=False)
def evaluate_model(
    test_path,
    model_path,
    vectorizer_path,
    task,
    test_signature,
    model_signature,
    vectorizer_signature
):

    result = {
        "available": False,
        "error": ""
    }

    if not all(
        os.path.exists(path)
        for path in [
            test_path,
            model_path,
            vectorizer_path
        ]
    ):
        result["error"] = "Required evaluation files are missing."
        return result

    try:

        test_df = pd.read_csv(test_path)

        text_column, label_column = detect_test_columns(test_df)

        if text_column is None or label_column is None:

            result["error"] = (
                "Could not identify text and label columns."
            )

            return result

        test_df = test_df[
            [text_column, label_column]
        ].dropna()

        test_df[text_column] = (
            test_df[text_column].astype(str)
        )

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        X = vectorizer.transform(
            test_df[text_column].apply(clean_text)
        )

        predictions = model.predict(X)

        if task == "sentiment":

            y_true = test_df[label_column].apply(
                normalize_sentiment
            )

            y_pred = pd.Series(
                predictions
            ).apply(
                normalize_sentiment
            )

            class_order = SENTIMENT_ORDER

        else:

            y_true = test_df[label_column].apply(
                normalize_emotion
            )

            y_pred = pd.Series(
                predictions
            ).apply(
                normalize_emotion
            )

            class_order = EMOTION_ORDER

        valid = (
            y_true.isin(class_order)
            & y_pred.isin(class_order)
        )

        y_true = y_true[valid]
        y_pred = y_pred[valid]

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            labels=class_order,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            labels=class_order,
            average="weighted",
            zero_division=0
        )

        f1_weighted = f1_score(
            y_true,
            y_pred,
            labels=class_order,
            average="weighted",
            zero_division=0
        )

        f1_macro = f1_score(
            y_true,
            y_pred,
            labels=class_order,
            average="macro",
            zero_division=0
        )

        f2_weighted = fbeta_score(
            y_true,
            y_pred,
            labels=class_order,
            beta=2,
            average="weighted",
            zero_division=0
        )

        f2_macro = fbeta_score(
            y_true,
            y_pred,
            labels=class_order,
            beta=2,
            average="macro",
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=class_order
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=class_order,
            target_names=class_order,
            output_dict=True,
            zero_division=0
        )

        report_df = (
            pd.DataFrame(report)
            .T
            .reset_index()
            .rename(columns={"index": "Class"})
        )

        result.update({
            "available": True,
            "samples": len(y_true),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "f2_weighted": f2_weighted,
            "f2_macro": f2_macro,
            "confusion_matrix": cm,
            "class_order": class_order,
            "report_df": report_df
        })

        return result

    except Exception as error:

        result["error"] = str(error)

        return result


sentiment_eval = evaluate_model(
    SENTIMENT_TEST_PATH,
    SENTIMENT_MODEL_PATH,
    SENTIMENT_VECTOR_PATH,
    "sentiment",
    get_file_signature(SENTIMENT_TEST_PATH),
    get_file_signature(SENTIMENT_MODEL_PATH),
    get_file_signature(SENTIMENT_VECTOR_PATH)
)

emotion_eval = evaluate_model(
    EMOTION_TEST_PATH,
    EMOTION_MODEL_PATH,
    EMOTION_VECTOR_PATH,
    "emotion",
    get_file_signature(EMOTION_TEST_PATH),
    get_file_signature(EMOTION_MODEL_PATH),
    get_file_signature(EMOTION_VECTOR_PATH)
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div style="padding:8px 0 20px;">
        <div style="font-size:26px;font-weight:800;">
            TrendSense AI
        </div>

        <div style="color:#777;font-size:12px;margin-top:4px;">
            Social Intelligence Dashboard
        </div>
    </div>
    """)

    st.markdown("### Data Source")

    source = st.radio(
        "Choose data",
        [
            "Demo Dataset",
            "Upload CSV"
        ]
    )

    uploaded_file = None

    if source == "Upload CSV":

        uploaded_file = st.file_uploader(
            "Upload social-media CSV",
            type=["csv"]
        )

    st.markdown("---")

    st.markdown("### Filters")

    search_query = st.text_input(
        "Search posts",
        placeholder="Search keywords..."
    )

    sentiment_filter = st.multiselect(
        "Sentiment",
        SENTIMENT_ORDER
    )

    emotion_filter = st.multiselect(
        "Emotion",
        EMOTION_ORDER
    )

    min_likes = st.number_input(
        "Minimum likes",
        min_value=0,
        value=0,
        step=10
    )

    st.markdown("---")

    st.markdown("### Model Stack")

    stack = [
        (
            "Sentiment",
            models["sentiment_model"] is not None
        ),
        (
            "Emotion",
            models["emotion_model"] is not None
        ),
        (
            "Topic LDA",
            models["topic_model"] is not None
        ),
        (
            "Model Evaluation",
            sentiment_eval["available"]
            or emotion_eval["available"]
        )
    ]

    for name, active in stack:

        symbol = "✓" if active else "○"

        st.markdown(
            f"""
            <div style="margin:5px 0;color:#aaa;">
                {symbol} {name}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# LOAD DATA
# ============================================================

if (
    source == "Upload CSV"
    and uploaded_file is not None
):

    raw_df = pd.read_csv(uploaded_file)

    data_source_label = "Uploaded Dataset"

else:

    if not os.path.exists(DATA_PATH):

        st.error(
            f"Dataset not found: {DATA_PATH}"
        )

        st.stop()

    raw_df = pd.read_csv(DATA_PATH)

    data_source_label = "Demo Dataset"


df = raw_df.copy()


# ============================================================
# STANDARDIZE DATA
# ============================================================

text_column = detect_column(
    df,
    [
        "text",
        "post",
        "content",
        "tweet",
        "message"
    ]
)

likes_column = detect_column(
    df,
    [
        "likes",
        "like",
        "likes_count"
    ]
)

shares_column = detect_column(
    df,
    [
        "shares",
        "share",
        "retweets",
        "retweet_count"
    ]
)

date_column = detect_column(
    df,
    [
        "date",
        "created_at",
        "timestamp",
        "time"
    ]
)

if text_column is None:

    st.error(
        "No text column found in the dataset."
    )

    st.stop()

df["text"] = (
    df[text_column]
    .fillna("")
    .astype(str)
)

if likes_column:

    df["likes"] = pd.to_numeric(
        df[likes_column],
        errors="coerce"
    ).fillna(0)

else:

    df["likes"] = 0


if shares_column:

    df["shares"] = pd.to_numeric(
        df[shares_column],
        errors="coerce"
    ).fillna(0)

else:

    df["shares"] = 0


if date_column:

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

else:

    df["date"] = pd.NaT


if df["date"].isna().all():

    df["date"] = pd.date_range(
        end=pd.Timestamp.today(),
        periods=len(df),
        freq="D"
    )

else:

    df["date"] = (
        df["date"]
        .ffill()
        .bfill()
    )


df["clean_text"] = (
    df["text"]
    .apply(clean_text)
)


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

if (
    models["sentiment_model"] is not None
    and
    models["sentiment_vectorizer"] is not None
):

    try:

        X_sentiment = (
            models["sentiment_vectorizer"]
            .transform(df["clean_text"])
        )

        sentiment_predictions = (
            models["sentiment_model"]
            .predict(X_sentiment)
        )

        df["sentiment"] = (
            pd.Series(
                sentiment_predictions,
                index=df.index
            )
            .apply(normalize_sentiment)
        )

        df["sentiment_confidence"] = (
            safe_probability(
                models["sentiment_model"],
                X_sentiment
            )
        )

    except Exception:

        df["sentiment"] = "Unknown"
        df["sentiment_confidence"] = np.nan

else:

    df["sentiment"] = "Unknown"
    df["sentiment_confidence"] = np.nan


# ============================================================
# EMOTION PREDICTION
# ============================================================

if (
    models["emotion_model"] is not None
    and
    models["emotion_vectorizer"] is not None
):

    try:

        X_emotion = (
            models["emotion_vectorizer"]
            .transform(df["clean_text"])
        )

        emotion_predictions = (
            models["emotion_model"]
            .predict(X_emotion)
        )

        df["emotion"] = (
            pd.Series(
                emotion_predictions,
                index=df.index
            )
            .apply(normalize_emotion)
        )

        df["emotion_confidence"] = (
            safe_probability(
                models["emotion_model"],
                X_emotion
            )
        )

    except Exception:

        df["emotion"] = "Unknown"
        df["emotion_confidence"] = np.nan

else:

    df["emotion"] = "Unknown"
    df["emotion_confidence"] = np.nan


# ============================================================
# TOPIC MODEL
# ============================================================

topic_names = {
    1: "AI Automation & Productivity",
    2: "ChatGPT, Coding & Education",
    3: "AI, Jobs & Future Technology"
}


if (
    models["topic_model"] is not None
    and
    models["topic_vectorizer"] is not None
):

    try:

        X_topic = (
            models["topic_vectorizer"]
            .transform(df["clean_text"])
        )

        topic_probabilities = (
            models["topic_model"]
            .transform(X_topic)
        )

        df["topic_id"] = (
            topic_probabilities
            .argmax(axis=1)
            + 1
        )

        df["topic_label"] = (
            df["topic_id"]
            .map(topic_names)
            .fillna(
                df["topic_id"]
                .apply(
                    lambda x: f"Topic {x}"
                )
            )
        )

        df["topic_confidence"] = (
            topic_probabilities
            .max(axis=1)
        )

    except Exception:

        df["topic_id"] = 0
        df["topic_label"] = "Unknown"
        df["topic_confidence"] = np.nan

else:

    df["topic_id"] = 0
    df["topic_label"] = "Unknown"
    df["topic_confidence"] = np.nan


# ============================================================
# ENGAGEMENT + TREND SCORE
# ============================================================

df["engagement"] = (
    df["likes"]
    + df["shares"]
)

df["trend_score"] = (
    df["engagement"]
    *
    (
        0.5
        +
        df["sentiment_confidence"]
        .fillna(0.5)
        *
        0.5
    )
)

df["hashtags"] = (
    df["text"]
    .apply(extract_hashtags)
)


# ============================================================
# FILTERS
# ============================================================

filtered_df = df.copy()

if search_query.strip():

    filtered_df = filtered_df[
        filtered_df["text"].str.contains(
            re.escape(search_query.strip()),
            case=False,
            na=False
        )
    ]

if sentiment_filter:

    filtered_df = filtered_df[
        filtered_df["sentiment"]
        .isin(sentiment_filter)
    ]

if emotion_filter:

    filtered_df = filtered_df[
        filtered_df["emotion"]
        .isin(emotion_filter)
    ]

filtered_df = filtered_df[
    filtered_df["likes"] >= min_likes
]


# ============================================================
# HERO
# ============================================================

render_html(f"""
<div class="hero">

    <div class="hero-title">
        TrendSense AI
    </div>

    <div class="hero-sub">
        AI-powered social-media intelligence for
        sentiment, emotion, topic discovery,
        engagement and emerging trends.
    </div>

    <div style="
        margin-top:14px;
        color:#777;
        font-size:12px;
    ">
        Source: {data_source_label}
        &nbsp;•&nbsp;
        {len(filtered_df):,} posts currently in view
    </div>

</div>
""")


# ============================================================
# KPI CARDS
# ============================================================

total_posts = len(filtered_df)

total_likes = (
    filtered_df["likes"].sum()
)

total_engagement = (
    filtered_df["engagement"].sum()
)

confidence_values = pd.concat(
    [
        filtered_df[
            "sentiment_confidence"
        ].dropna(),

        filtered_df[
            "emotion_confidence"
        ].dropna()
    ],
    ignore_index=True
)

average_confidence = (
    confidence_values.mean()
    if len(confidence_values)
    else np.nan
)

columns = st.columns(4)

with columns[0]:

    render_html(
        kpi_card(
            "Posts",
            compact_number(total_posts),
            "Current filtered view"
        )
    )

with columns[1]:

    render_html(
        kpi_card(
            "Total Likes",
            compact_number(total_likes),
            "Across visible posts"
        )
    )

with columns[2]:

    render_html(
        kpi_card(
            "Engagement",
            compact_number(total_engagement),
            "Likes + shares"
        )
    )

with columns[3]:

    confidence_text = (
        f"{average_confidence * 100:.1f}%"
        if pd.notna(average_confidence)
        else "N/A"
    )

    render_html(
        kpi_card(
            "Avg. Prediction Confidence",
            confidence_text,
            "Not model accuracy"
        )
    )


# ============================================================
# CONFIDENCE MESSAGE
# ============================================================

render_html("""
<div class="warning">

<b>Important:</b>
Prediction confidence represents how confident the model is
for its current predictions. It is NOT the same as accuracy.

Use the <b>Model Performance</b> tab for Accuracy,
Precision, Recall, F1 and F2.

</div>
""")


# ============================================================
# AI INSIGHTS
# ============================================================

render_html(
    '<div class="section-title">AI Insights</div>'
)

if len(filtered_df):

    emotion_counts = (
        filtered_df["emotion"]
        .value_counts()
    )

    topic_counts = (
        filtered_df["topic_label"]
        .value_counts()
    )

    dominant_emotion = (
        emotion_counts.index[0]
        if len(emotion_counts)
        else "Unknown"
    )

    leading_topic = (
        topic_counts.index[0]
        if len(topic_counts)
        else "Unknown"
    )

    all_tags = [
        tag
        for tags in filtered_df["hashtags"]
        for tag in tags
    ]

    trending_hashtag = (
        pd.Series(all_tags)
        .value_counts()
        .index[0]
        if all_tags
        else "No hashtag data"
    )

    top_post = (
        filtered_df
        .sort_values(
            "trend_score",
            ascending=False
        )
        .iloc[0]
    )

    columns = st.columns(4)

    with columns[0]:

        render_html(f"""
        <div class="card">
            <div class="card-title">
                Dominant Emotion
            </div>

            <div class="card-text">
                {dominant_emotion}
            </div>
        </div>
        """)

    with columns[1]:

        render_html(f"""
        <div class="card">
            <div class="card-title">
                Leading Topic
            </div>

            <div class="card-text">
                {leading_topic}
            </div>
        </div>
        """)

    with columns[2]:

        render_html(f"""
        <div class="card">
            <div class="card-title">
                Trending Hashtag
            </div>

            <div class="card-text">
                {trending_hashtag}
            </div>
        </div>
        """)

    with columns[3]:

        render_html(f"""
        <div class="card">
            <div class="card-title">
                Top Conversation
            </div>

            <div class="card-text">
                {str(top_post["text"])[:80]}...
            </div>
        </div>
        """)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 Overview",
        "💭 Sentiment & Emotion",
        "🔥 Topics & Trends",
        "🧪 Model Performance",
        "🔎 Post Explorer",
        "📄 Reports"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    render_html(
        '<div class="section-title">'
        'Conversation Overview'
        '</div>'
    )

    render_html(
        '<div class="section-sub">'
        'High-level view of sentiment, emotion and engagement.'
        '</div>'
    )

    column1, column2 = st.columns(2)

    with column1:

        sentiment_counts = (
            filtered_df["sentiment"]
            .value_counts()
            .reindex(
                SENTIMENT_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.55,
            title="Sentiment Distribution"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with column2:

        emotion_counts = (
            filtered_df["emotion"]
            .value_counts()
            .reset_index()
        )

        emotion_counts.columns = [
            "Emotion",
            "Count"
        ]

        fig = px.bar(
            emotion_counts,
            x="Emotion",
            y="Count",
            title="Emotion Distribution"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=400
        )

        fig.update_xaxes(
            tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if len(filtered_df):

        trend_df = (
            filtered_df
            .groupby(
                "date",
                as_index=False
            )["engagement"]
            .sum()
            .sort_values("date")
        )

        fig = px.line(
            trend_df,
            x="date",
            y="engagement",
            markers=True,
            title="Engagement Over Time"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=420
        )

        fig.update_xaxes(title="")
        fig.update_yaxes(
            title="Engagement"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# SENTIMENT & EMOTION
# ============================================================

with tabs[1]:

    render_html(
        '<div class="section-title">'
        'Sentiment & Emotion Intelligence'
        '</div>'
    )

    render_html(
        '<div class="section-sub">'
        'Understand what people feel and how confident the prediction pipeline is.'
        '</div>'
    )

    column1, column2 = st.columns(2)

    with column1:

        sentiment_counts = (
            filtered_df["sentiment"]
            .value_counts()
            .reindex(
                SENTIMENT_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Sentiment Count"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with column2:

        emotion_counts = (
            filtered_df["emotion"]
            .value_counts()
            .reset_index()
        )

        emotion_counts.columns = [
            "Emotion",
            "Count"
        ]

        fig = px.bar(
            emotion_counts,
            x="Emotion",
            y="Count",
            title="Emotion Count"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=390
        )

        fig.update_xaxes(
            tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    confidence_df = pd.DataFrame({
        "Sentiment Confidence":
            filtered_df[
                "sentiment_confidence"
            ],

        "Emotion Confidence":
            filtered_df[
                "emotion_confidence"
            ]
    })

    confidence_df = (
        confidence_df
        .melt(
            var_name="Model",
            value_name="Confidence"
        )
        .dropna()
    )

    if len(confidence_df):

        fig = px.box(
            confidence_df,
            x="Model",
            y="Confidence",
            title="Prediction Confidence"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TOPICS & TRENDS
# ============================================================

with tabs[2]:

    render_html(
        '<div class="section-title">'
        'Topics & Trends'
        '</div>'
    )

    render_html(
        '<div class="section-sub">'
        'Discover conversation themes, engagement hotspots and hashtags.'
        '</div>'
    )

    column1, column2 = st.columns(2)

    with column1:

        topic_counts = (
            filtered_df["topic_label"]
            .value_counts()
            .reset_index()
        )

        topic_counts.columns = [
            "Topic",
            "Posts"
        ]

        fig = px.bar(
            topic_counts,
            x="Posts",
            y="Topic",
            orientation="h",
            title="Topic Distribution"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with column2:

        topic_engagement = (
            filtered_df
            .groupby(
                "topic_label",
                as_index=False
            )["engagement"]
            .sum()
            .sort_values(
                "engagement",
                ascending=False
            )
        )

        fig = px.bar(
            topic_engagement,
            x="topic_label",
            y="engagement",
            title="Engagement by Topic"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=430
        )

        fig.update_xaxes(
            tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    all_tags = [
        tag
        for tags in filtered_df["hashtags"]
        for tag in tags
    ]

    if all_tags:

        tag_counts = (
            pd.Series(all_tags)
            .value_counts()
            .head(15)
            .reset_index()
        )

        tag_counts.columns = [
            "Hashtag",
            "Count"
        ]

        fig = px.bar(
            tag_counts,
            x="Hashtag",
            y="Count",
            title="Top Hashtags"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=400
        )

        fig.update_xaxes(
            tickangle=-35
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    corpus = " ".join(
        filtered_df[
            "clean_text"
        ]
        .astype(str)
        .tolist()
    ).strip()

    if corpus:

        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color="black",
            stopwords=set(
                ENGLISH_STOP_WORDS
            ),
            collocations=False
        ).generate(corpus)

        figure, axis = plt.subplots(
            figsize=(14, 5)
        )

        axis.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        axis.axis("off")

        st.pyplot(
            figure,
            use_container_width=True
        )

        plt.close(figure)

    render_html("""
    <div class="warning">

    <b>Topic-model note:</b>
    The current demo dataset contains a small number of posts,
    so topic modeling is mainly a prototype demonstration.
    A larger real-world dataset will produce stronger topic discovery.

    </div>
    """)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with tabs[3]:

    render_html(
        '<div class="section-title">'
        'Model Performance'
        '</div>'
    )

    render_html("""
    <div class="section-sub">
        Objective evaluation using held-out test datasets.
        These metrics are separate from the demo/uploaded posts.
    </div>

    <div class="success">

    <b>Evaluation methodology:</b>

    Accuracy, Precision, Recall, F1 and F2 are calculated
    using the saved models against the held-out test datasets.

    F2 gives more importance to recall than F1.

    </div>
    """)

    performance_tabs = st.tabs(
        [
            "Sentiment Evaluation",
            "Emotion Evaluation"
        ]
    )

    # --------------------------------------------------------
    # SENTIMENT PERFORMANCE
    # --------------------------------------------------------

    with performance_tabs[0]:

        if not sentiment_eval["available"]:

            st.warning(
                "Sentiment evaluation unavailable: "
                + sentiment_eval["error"]
            )

        else:

            s = sentiment_eval

            columns = st.columns(6)

            metrics = [
                (
                    "Test Samples",
                    f"{s['samples']:,}",
                    ""
                ),
                (
                    "Accuracy",
                    f"{s['accuracy'] * 100:.2f}%",
                    ""
                ),
                (
                    "Precision",
                    f"{s['precision'] * 100:.2f}%",
                    "Weighted"
                ),
                (
                    "Recall",
                    f"{s['recall'] * 100:.2f}%",
                    "Weighted"
                ),
                (
                    "F1",
                    f"{s['f1_weighted'] * 100:.2f}%",
                    "Weighted"
                ),
                (
                    "F2",
                    f"{s['f2_weighted'] * 100:.2f}%",
                    "Recall-focused"
                )
            ]

            for column, metric in zip(
                columns,
                metrics
            ):

                with column:

                    render_html(
                        kpi_card(
                            metric[0],
                            metric[1],
                            metric[2]
                        )
                    )

            st.markdown("")

            column1, column2 = st.columns(
                [1.05, 1]
            )

            with column1:

                matrix = pd.DataFrame(
                    s["confusion_matrix"],
                    index=s["class_order"],
                    columns=s["class_order"]
                )

                fig = px.imshow(
                    matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Sentiment Confusion Matrix",
                    labels={
                        "x": "Predicted Label",
                        "y": "Actual Label",
                        "color": "Count"
                    }
                )

                fig.update_layout(
                    PLOTLY_LAYOUT,
                    height=430
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with column2:

                st.markdown(
                    "#### Per-class Performance"
                )

                st.dataframe(
                    s["report_df"],
                    use_container_width=True,
                    hide_index=True
                )

            column1, column2 = st.columns(2)

            with column1:

                render_html(f"""
                <div class="card">

                    <div class="card-title">
                        Macro F1
                    </div>

                    <div class="card-text">
                        {s["f1_macro"] * 100:.2f}%

                        <br>

                        <span style="color:#777;">
                        Gives every class equal importance.
                        </span>
                    </div>

                </div>
                """)

            with column2:

                render_html(f"""
                <div class="card">

                    <div class="card-title">
                        Macro F2
                    </div>

                    <div class="card-text">
                        {s["f2_macro"] * 100:.2f}%

                        <br>

                        <span style="color:#777;">
                        Gives every class equal importance
                        while emphasizing recall.
                        </span>
                    </div>

                </div>
                """)


    # --------------------------------------------------------
    # EMOTION PERFORMANCE
    # --------------------------------------------------------

    with performance_tabs[1]:

        if not emotion_eval["available"]:

            st.warning(
                "Emotion evaluation unavailable: "
                + emotion_eval["error"]
            )

        else:

            e = emotion_eval

            columns = st.columns(6)

            metrics = [
                (
                    "Test Samples",
                    f"{e['samples']:,}",
                    ""
                ),
                (
                    "Accuracy",
                    f"{e['accuracy'] * 100:.2f}%",
                    ""
                ),
                (
                    "Precision",
                    f"{e['precision'] * 100:.2f}%",
                    "Weighted"
                ),
                (
                    "Recall",
                    f"{e['recall'] * 100:.2f}%",
                    "Weighted"
                ),
                (
                    "F1",
                    f"{e['f1_weighted'] * 100:.2f}%",
                    "Weighted"
                ),
                (
                    "F2",
                    f"{e['f2_weighted'] * 100:.2f}%",
                    "Recall-focused"
                )
            ]

            for column, metric in zip(
                columns,
                metrics
            ):

                with column:

                    render_html(
                        kpi_card(
                            metric[0],
                            metric[1],
                            metric[2]
                        )
                    )

            st.markdown("")

            column1, column2 = st.columns(
                [1.05, 1]
            )

            with column1:

                matrix = pd.DataFrame(
                    e["confusion_matrix"],
                    index=e["class_order"],
                    columns=e["class_order"]
                )

                fig = px.imshow(
                    matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Emotion Confusion Matrix",
                    labels={
                        "x": "Predicted Emotion",
                        "y": "Actual Emotion",
                        "color": "Count"
                    }
                )

                fig.update_layout(
                    PLOTLY_LAYOUT,
                    height=500
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with column2:

                st.markdown(
                    "#### Per-class Performance"
                )

                st.dataframe(
                    e["report_df"],
                    use_container_width=True,
                    hide_index=True
                )

            column1, column2 = st.columns(2)

            with column1:

                render_html(f"""
                <div class="card">

                    <div class="card-title">
                        Macro F1
                    </div>

                    <div class="card-text">
                        {e["f1_macro"] * 100:.2f}%

                        <br>

                        <span style="color:#777;">
                        Gives every emotion equal importance.
                        </span>
                    </div>

                </div>
                """)

            with column2:

                render_html(f"""
                <div class="card">

                    <div class="card-title">
                        Macro F2
                    </div>

                    <div class="card-text">
                        {e["f2_macro"] * 100:.2f}%

                        <br>

                        <span style="color:#777;">
                        Equal class weighting with extra emphasis on recall.
                        </span>
                    </div>

                </div>
                """)

    render_html("""
    <div class="warning">

    <b>Remember:</b>
    High confidence for one prediction does not mean the model is highly accurate.
    Model quality should be discussed using the held-out evaluation metrics above.

    </div>
    """)


# ============================================================
# POST EXPLORER
# ============================================================

with tabs[4]:

    render_html(
        '<div class="section-title">'
        'Post Explorer'
        '</div>'
    )

    render_html(
        '<div class="section-sub">'
        'Inspect individual predictions, topics, confidence and engagement.'
        '</div>'
    )

    display_columns = [
        column
        for column in [
            "id",
            "post_id",
            "text",
            "date",
            "likes",
            "shares",
            "engagement",
            "sentiment",
            "sentiment_confidence",
            "emotion",
            "emotion_confidence",
            "topic_label",
            "topic_confidence",
            "trend_score"
        ]
        if column in filtered_df.columns
    ]

    display_df = filtered_df[
        display_columns
    ].copy()

    for column in [
        "sentiment_confidence",
        "emotion_confidence",
        "topic_confidence"
    ]:

        if column in display_df.columns:

            display_df[column] = (
                display_df[column]
                .round(3)
            )

    if "trend_score" in display_df.columns:

        display_df["trend_score"] = (
            display_df["trend_score"]
            .round(2)
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    if len(filtered_df):

        st.markdown(
            "#### Highest-trend posts"
        )

        top_posts = (
            filtered_df
            .sort_values(
                "trend_score",
                ascending=False
            )
            .head(5)
            [
                [
                    "text",
                    "sentiment",
                    "emotion",
                    "topic_label",
                    "engagement",
                    "trend_score"
                ]
            ]
        )

        st.dataframe(
            top_posts,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def create_csv_report(report_df):

    return report_df.to_csv(
        index=False
    ).encode("utf-8")


def create_excel_report(report_df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        report_df.to_excel(
            writer,
            sheet_name="TrendSense Results",
            index=False
        )

    output.seek(0)

    return output.getvalue()


def create_pdf_report(text):

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer
        )
        from reportlab.lib.styles import (
            getSampleStyleSheet
        )

        output = io.BytesIO()

        document = SimpleDocTemplate(
            output,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        story = []

        for line in text.split("\n"):

            if line.strip():

                story.append(
                    Paragraph(
                        line.replace(
                            "&",
                            "&amp;"
                        ),
                        styles["BodyText"]
                    )
                )

                story.append(
                    Spacer(1, 8)
                )

        document.build(story)

        output.seek(0)

        return output.getvalue()

    except Exception:

        return None


# ============================================================
# REPORTS
# ============================================================

with tabs[5]:

    render_html(
        '<div class="section-title">'
        'Reports'
        '</div>'
    )

    render_html(
        '<div class="section-sub">'
        'Export the current analysis for presentation or documentation.'
        '</div>'
    )

    report_columns = [
        column
        for column in [
            "text",
            "date",
            "likes",
            "shares",
            "engagement",
            "sentiment",
            "sentiment_confidence",
            "emotion",
            "emotion_confidence",
            "topic_label",
            "topic_confidence",
            "trend_score"
        ]
        if column in filtered_df.columns
    ]

    report_df = filtered_df[
        report_columns
    ].copy()

    csv_data = create_csv_report(
        report_df
    )

    excel_data = create_excel_report(
        report_df
    )

    sentiment_accuracy = (
        f"{sentiment_eval['accuracy'] * 100:.2f}%"
        if sentiment_eval["available"]
        else "Unavailable"
    )

    emotion_accuracy = (
        f"{emotion_eval['accuracy'] * 100:.2f}%"
        if emotion_eval["available"]
        else "Unavailable"
    )

    summary = f"""
TrendSense AI Report

Data Source: {data_source_label}

Posts in current view:
{len(filtered_df)}

Total Likes:
{int(filtered_df["likes"].sum())}

Total Engagement:
{int(filtered_df["engagement"].sum())}

Sentiment Accuracy:
{sentiment_accuracy}

Emotion Accuracy:
{emotion_accuracy}

Important:
Prediction confidence is not model accuracy.
Model performance is calculated using held-out test datasets.
"""

    pdf_data = create_pdf_report(
        summary
    )

    columns = st.columns(3)

    with columns[0]:

        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="trendsense_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with columns[1]:

        st.download_button(
            "⬇️ Download Excel",
            data=excel_data,
            file_name="trendsense_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    with columns[2]:

        if pdf_data:

            st.download_button(
                "⬇️ Download PDF",
                data=pdf_data,
                file_name="trendsense_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        else:

            st.warning(
                "PDF export unavailable."
            )

    render_html("""
    <div class="card">

        <div class="card-title">
            Executive Summary
        </div>

        <div class="card-text">

            TrendSense AI combines NLP-based sentiment
            and emotion classification with topic discovery,
            engagement analysis, hashtag tracking and trend scoring.

            <br><br>

            The dashboard also provides objective model evaluation
            using held-out test datasets.

        </div>

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

TrendSense AI • NLP & Social Intelligence Project

<br>

Sentiment • Emotion • Topic Modeling • Model Evaluation

</div>
""")
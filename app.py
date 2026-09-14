import os
import io
import re
import html
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

DATA_PATH = os.path.join(
    BASE_DIR, "data", "social_media.csv"
)

SENTIMENT_MODEL_PATH = os.path.join(
    BASE_DIR, "sentiment_model.pkl"
)

SENTIMENT_VECTOR_PATH = os.path.join(
    BASE_DIR, "tfidf_vectorizer.pkl"
)

EMOTION_MODEL_PATH = os.path.join(
    BASE_DIR, "emotion_model.pkl"
)

EMOTION_VECTOR_PATH = os.path.join(
    BASE_DIR, "emotion_vectorizer.pkl"
)

TOPIC_MODEL_PATH = os.path.join(
    BASE_DIR, "topic_model.pkl"
)

TOPIC_VECTOR_PATH = os.path.join(
    BASE_DIR, "topic_vectorizer.pkl"
)

SENTIMENT_TEST_PATH = os.path.join(
    BASE_DIR, "data", "sentiment_test.csv"
)

EMOTION_TEST_PATH = os.path.join(
    BASE_DIR, "data", "emotion_test.csv"
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content):
    try:
        st.html(content)
    except Exception:
        st.markdown(
            content,
            unsafe_allow_html=True
        )


# ============================================================
# PREMIUM LIGHT UI
# ============================================================

render_html("""
<style>

.stApp {
    background: #f5f7fb;
    color: #172033;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

section[data-testid="stSidebar"] * {
    color: #344054;
}


/* HERO */

.hero {
    padding: 35px;
    border-radius: 25px;
    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #f5f0ff 50%,
            #edf5ff 100%
        );
    border: 1px solid #e5e7eb;
    margin-bottom: 25px;
    box-shadow:
        0 10px 35px
        rgba(79, 70, 229, 0.08);
}

.hero-title {
    font-size: 47px;
    font-weight: 850;
    letter-spacing: -2px;
    color: #17132b;
}

.hero-gradient {
    background:
        linear-gradient(
            90deg,
            #6d28d9,
            #2563eb
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #667085;
    font-size: 16px;
    line-height: 1.7;
    margin-top: 10px;
    max-width: 950px;
}

.source-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 15px;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid #ddd6fe;
    color: #6d28d9;
    font-size: 12px;
    font-weight: 750;
}


/* SECTION */

.section-title {
    font-size: 27px;
    font-weight: 800;
    color: #17132b;
    margin: 30px 0 7px;
}

.section-sub {
    color: #667085;
    margin-bottom: 20px;
}


/* KPI */

.kpi {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 21px;
    min-height: 125px;
    box-shadow:
        0 5px 20px
        rgba(15, 23, 42, 0.045);
}

.kpi-label {
    color: #667085;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .09em;
    font-weight: 750;
}

.kpi-value {
    color: #17132b;
    font-size: 30px;
    font-weight: 850;
    margin-top: 8px;
}

.metric-note {
    color: #98a2b3;
    font-size: 11px;
    margin-top: 5px;
}


/* CARDS */

.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 21px;
    margin-bottom: 15px;
    box-shadow:
        0 5px 20px
        rgba(15, 23, 42, 0.035);
}

.card-title {
    font-size: 17px;
    font-weight: 750;
    color: #1f2937;
    margin-bottom: 8px;
}

.card-text {
    color: #667085;
    line-height: 1.6;
}


/* STATUS */

.success-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    border-radius: 14px;
    padding: 15px 17px;
    margin: 14px 0;
}

.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    border-radius: 14px;
    padding: 15px 17px;
    margin: 14px 0;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
    border-radius: 14px;
    padding: 15px 17px;
    margin: 14px 0;
}


/* BUTTONS */

.stButton > button {
    border-radius: 11px;
    border: 1px solid #ddd6fe;
    background: #ffffff;
    color: #5b21b6;
    font-weight: 650;
}

.stButton > button:hover {
    border-color: #8b5cf6;
    background: #f5f3ff;
}


/* FILE UPLOAD */

[data-testid="stFileUploader"] {
    background: #fafaff;
    border-radius: 15px;
}


/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}


/* FOOTER */

.footer {
    color: #98a2b3;
    text-align: center;
    padding: 40px 0 10px;
    font-size: 12px;
}

</style>
""")


# ============================================================
# PLOTLY THEME
# ============================================================

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(
        color="#344054"
    ),
    margin=dict(
        l=30,
        r=25,
        t=60,
        b=30
    ),
)


# ============================================================
# LABELS
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

SENTIMENT_ORDER = [
    "Negative",
    "Neutral",
    "Positive"
]


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

EMOTION_ORDER = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


TOPIC_NAMES = {
    1: "AI Automation & Productivity",
    2: "ChatGPT, Coding & Education",
    3: "AI, Jobs & Future Technology",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_label(value, mapping):

    if isinstance(
        value,
        (int, np.integer)
    ):
        return mapping.get(
            int(value),
            "Unknown"
        )

    value = str(value).strip()

    if value in mapping:
        return mapping[value]

    lower = value.lower()

    if lower in mapping:
        return mapping[lower]

    try:
        return mapping.get(
            int(float(value)),
            "Unknown"
        )
    except Exception:
        return (
            value.title()
            if value
            else "Unknown"
        )


def normalize_sentiment(value):
    return normalize_label(
        value,
        SENTIMENT_LABELS
    )


def normalize_emotion(value):
    return normalize_label(
        value,
        EMOTION_LABELS
    )


def compact_number(value):

    try:

        value = float(value)

        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"

        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"

        if value >= 1_000:
            return f"{value / 1_000:.1f}K"

        return f"{int(value):,}"

    except Exception:

        return "0"


def detect_column(
    df,
    candidates
):

    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    # Exact match first
    for candidate in candidates:

        if candidate.lower() in columns:

            return columns[
                candidate.lower()
            ]

    # Partial match second
    for column in df.columns:

        column_lower = (
            str(column)
            .strip()
            .lower()
        )

        for candidate in candidates:

            if candidate.lower() in column_lower:

                return column

    return None


def parse_metric(value):

    if pd.isna(value):
        return 0.0

    if isinstance(
        value,
        (int, float, np.number)
    ):
        return float(value)

    value = str(value).strip()

    if not value:
        return 0.0

    value = (
        value
        .replace(",", "")
        .replace(" ", "")
        .replace("+", "")
    )

    multiplier = 1

    last = value[-1:].upper()

    if last == "K":
        multiplier = 1_000
        value = value[:-1]

    elif last == "M":
        multiplier = 1_000_000
        value = value[:-1]

    elif last == "B":
        multiplier = 1_000_000_000
        value = value[:-1]

    try:
        return float(value) * multiplier

    except Exception:
        return 0.0


def extract_hashtags(text):

    return [
        tag.lower()
        for tag in re.findall(
            r"#\w+",
            str(text)
        )
    ]


def clean_fallback(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = [
        word
        for word in text.split()
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)


try:

    from nlp_utils import clean_text

except Exception:

    clean_text = clean_fallback


def kpi_card(
    label,
    value,
    note=""
):

    return f"""
    <div class="kpi">

        <div class="kpi-label">
            {html.escape(str(label))}
        </div>

        <div class="kpi-value">
            {html.escape(str(value))}
        </div>

        <div class="metric-note">
            {html.escape(str(note))}
        </div>

    </div>
    """


# ============================================================
# PLATFORM DETECTION
# ============================================================

def detect_platform(
    df,
    filename=""
):

    filename_lower = (
        str(filename)
        .lower()
    )

    column_text = " ".join(
        str(c).lower()
        for c in df.columns
    )

    # Explicit platform column
    platform_column = detect_column(
        df,
        [
            "platform",
            "social_platform",
            "network",
            "source"
        ]
    )

    if platform_column:

        values = (
            df[platform_column]
            .dropna()
            .astype(str)
            .str.lower()
        )

        detected = []

        for value in values.unique():

            if "youtube" in value:
                detected.append(
                    "YouTube"
                )

            elif (
                "instagram" in value
                or value == "ig"
            ):
                detected.append(
                    "Instagram"
                )

            elif (
                "twitter" in value
                or value == "x"
                or "twitter/x" in value
            ):
                detected.append(
                    "Twitter/X"
                )

            elif "reddit" in value:
                detected.append(
                    "Reddit"
                )

        if detected:
            return sorted(
                set(detected)
            )

    # YouTube signals
    youtube_signals = [
        "video_id",
        "channel_title",
        "channel_id",
        "video_views",
        "subscriber_count",
        "author_channel_id",
    ]

    if (
        any(
            signal in column_text
            for signal in youtube_signals
        )
        or
        "youtube" in filename_lower
    ):
        return ["YouTube"]

    # Instagram signals
    instagram_signals = [
        "media_id",
        "shortcode",
        "owner_username",
        "is_video",
        "carousel",
        "save_count",
        "saves",
    ]

    if (
        any(
            signal in column_text
            for signal in instagram_signals
        )
        or
        "instagram" in filename_lower
        or
        "insta" in filename_lower
    ):
        return ["Instagram"]

    # Twitter/X signals
    twitter_signals = [
        "tweet_id",
        "retweet_count",
        "favorite_count",
        "favourites",
        "user_followers",
        "tweet",
    ]

    if (
        any(
            signal in column_text
            for signal in twitter_signals
        )
        or
        "twitter" in filename_lower
        or
        "tweet" in filename_lower
    ):
        return ["Twitter/X"]

    # Reddit signals
    reddit_signals = [
        "subreddit",
        "num_comments",
        "score",
        "upvote_ratio",
        "reddit_id",
        "permalink",
    ]

    if (
        any(
            signal in column_text
            for signal in reddit_signals
        )
        or
        "reddit" in filename_lower
    ):
        return ["Reddit"]

    return ["Unknown"]


# ============================================================
# DATA STANDARDIZATION
# ============================================================

def standardize_social_data(
    raw_df,
    filename=""
):

    df = raw_df.copy()

    detected_platforms = detect_platform(
        df,
        filename
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_column = detect_column(
        df,
        [
            "text",
            "post",
            "content",
            "tweet",
            "message",
            "comment",
            "caption",
            "description",
            "body",
            "review",
            "title"
        ]
    )

    if text_column is None:

        raise ValueError(
            "No text column found. "
            "Use a column such as text, "
            "content, tweet, comment, "
            "caption or post."
        )

    df["text"] = (
        df[text_column]
        .fillna("")
        .astype(str)
    )


    # --------------------------------------------------------
    # PLATFORM
    # --------------------------------------------------------

    platform_column = detect_column(
        df,
        [
            "platform",
            "social_platform",
            "network",
            "source"
        ]
    )

    if platform_column:

        df["platform"] = (
            df[platform_column]
            .fillna(
                detected_platforms[0]
            )
            .astype(str)
        )

        def normalize_platform(value):

            value = (
                str(value)
                .strip()
                .lower()
            )

            if "youtube" in value:
                return "YouTube"

            if (
                "instagram" in value
                or value == "ig"
            ):
                return "Instagram"

            if (
                "twitter" in value
                or value == "x"
                or "twitter/x" in value
            ):
                return "Twitter/X"

            if "reddit" in value:
                return "Reddit"

            if value in [
                "",
                "nan",
                "none"
            ]:
                return "Unknown"

            return str(value).title()

        df["platform"] = (
            df["platform"]
            .apply(normalize_platform)
        )

    else:

        df["platform"] = (
            detected_platforms[0]
            if len(detected_platforms) == 1
            else "Multi-platform"
        )


    # --------------------------------------------------------
    # POST ID
    # --------------------------------------------------------

    post_id_column = detect_column(
        df,
        [
            "post_id",
            "tweet_id",
            "video_id",
            "comment_id",
            "media_id",
            "reddit_id",
            "id"
        ]
    )

    if post_id_column:

        df["post_id"] = (
            df[post_id_column]
            .astype(str)
        )

    else:

        df["post_id"] = [
            f"TS-{i:06d}"
            for i in range(
                1,
                len(df) + 1
            )
        ]


    # --------------------------------------------------------
    # LIKES
    # --------------------------------------------------------

    likes_column = detect_column(
        df,
        [
            "likes",
            "like",
            "likes_count",
            "like_count",
            "favorite_count",
            "favourites",
            "favorites",
            "reactions",
            "total_likes"
        ]
    )

    if likes_column:

        df["likes"] = (
            df[likes_column]
            .apply(parse_metric)
        )

    else:

        df["likes"] = 0.0


    # --------------------------------------------------------
    # COMMENTS
    # --------------------------------------------------------

    comments_column = detect_column(
        df,
        [
            "comments",
            "comment_count",
            "comments_count",
            "num_comments",
            "replies",
            "reply_count"
        ]
    )

    if comments_column:

        df["comments"] = (
            df[comments_column]
            .apply(parse_metric)
        )

    else:

        df["comments"] = 0.0


    # --------------------------------------------------------
    # SHARES
    # --------------------------------------------------------

    shares_column = detect_column(
        df,
        [
            "shares",
            "share",
            "shares_count",
            "retweets",
            "retweet_count",
            "reposts",
            "repost_count"
        ]
    )

    if shares_column:

        df["shares"] = (
            df[shares_column]
            .apply(parse_metric)
        )

    else:

        df["shares"] = 0.0


    # --------------------------------------------------------
    # VIEWS
    # --------------------------------------------------------

    views_column = detect_column(
        df,
        [
            "views",
            "view_count",
            "views_count",
            "video_views",
            "play_count",
            "plays",
            "impressions"
        ]
    )

    if views_column:

        df["views"] = (
            df[views_column]
            .apply(parse_metric)
        )

    else:

        df["views"] = 0.0


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_column = detect_column(
        df,
        [
            "date",
            "created_at",
            "timestamp",
            "time",
            "datetime",
            "published_at",
            "published",
            "upload_date",
            "posted_at"
        ]
    )

    if date_column:

        df["date"] = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

    else:

        df["date"] = pd.NaT


    # --------------------------------------------------------
    # AUTHOR
    # --------------------------------------------------------

    author_column = detect_column(
        df,
        [
            "author",
            "username",
            "user",
            "creator",
            "channel",
            "author_name"
        ]
    )

    if author_column:

        df["author"] = (
            df[author_column]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        df["author"] = "Unknown"


    # --------------------------------------------------------
    # FOLLOWERS
    # --------------------------------------------------------

    follower_column = detect_column(
        df,
        [
            "author_followers",
            "followers",
            "follower_count",
            "user_followers",
            "subscribers",
            "subscriber_count"
        ]
    )

    if follower_column:

        df["author_followers"] = (
            df[follower_column]
            .apply(parse_metric)
        )

    else:

        df["author_followers"] = 0.0


    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    language_column = detect_column(
        df,
        [
            "language",
            "lang"
        ]
    )

    if language_column:

        df["language"] = (
            df[language_column]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        df["language"] = "Unknown"


    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    url_column = detect_column(
        df,
        [
            "source_url",
            "url",
            "link",
            "permalink"
        ]
    )

    if url_column:

        df["source_url"] = (
            df[url_column]
            .fillna("")
            .astype(str)
        )

    else:

        df["source_url"] = ""


    # --------------------------------------------------------
    # POST TYPE
    # --------------------------------------------------------

    type_column = detect_column(
        df,
        [
            "post_type",
            "content_type",
            "media_type",
            "type"
        ]
    )

    if type_column:

        df["post_type"] = (
            df[type_column]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        df["post_type"] = "Unknown"


    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    df["clean_text"] = (
        df["text"]
        .apply(clean_text)
    )


    # --------------------------------------------------------
    # HASHTAGS
    # --------------------------------------------------------

    df["hashtags"] = (
        df["text"]
        .apply(extract_hashtags)
    )


    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    df["engagement"] = (
        df["likes"]
        +
        df["comments"]
        +
        df["shares"]
    )


    # --------------------------------------------------------
    # ENGAGEMENT RATE
    # --------------------------------------------------------

    df["engagement_rate"] = np.where(
        df["views"] > 0,
        (
            df["engagement"]
            /
            df["views"]
        ) * 100,
        np.nan
    )


    return (
        df,
        detected_platforms
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    paths = {
        "sentiment_model":
            SENTIMENT_MODEL_PATH,

        "sentiment_vectorizer":
            SENTIMENT_VECTOR_PATH,

        "emotion_model":
            EMOTION_MODEL_PATH,

        "emotion_vectorizer":
            EMOTION_VECTOR_PATH,

        "topic_model":
            TOPIC_MODEL_PATH,

        "topic_vectorizer":
            TOPIC_VECTOR_PATH,
    }

    for name, path in paths.items():

        if os.path.exists(path):

            try:

                models[name] = (
                    joblib.load(path)
                )

            except Exception:

                models[name] = None

        else:

            models[name] = None

    return models


models = load_models()


# ============================================================
# AI ANALYSIS
# ============================================================

def run_ai_analysis(df):

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    if (
        models["sentiment_model"]
        is not None
        and
        models["sentiment_vectorizer"]
        is not None
    ):

        try:

            X = (
                models[
                    "sentiment_vectorizer"
                ]
                .transform(
                    df["clean_text"]
                )
            )

            predictions = (
                models[
                    "sentiment_model"
                ]
                .predict(X)
            )

            df["sentiment"] = (
                pd.Series(
                    predictions,
                    index=df.index
                )
                .apply(
                    normalize_sentiment
                )
            )

            try:

                probabilities = (
                    models[
                        "sentiment_model"
                    ]
                    .predict_proba(X)
                )

                df[
                    "sentiment_confidence"
                ] = (
                    probabilities
                    .max(axis=1)
                )

            except Exception:

                df[
                    "sentiment_confidence"
                ] = np.nan

        except Exception:

            df["sentiment"] = "Unknown"

            df[
                "sentiment_confidence"
            ] = np.nan

    else:

        df["sentiment"] = "Unknown"

        df[
            "sentiment_confidence"
        ] = np.nan


    # --------------------------------------------------------
    # EMOTION
    # --------------------------------------------------------

    if (
        models["emotion_model"]
        is not None
        and
        models["emotion_vectorizer"]
        is not None
    ):

        try:

            X = (
                models[
                    "emotion_vectorizer"
                ]
                .transform(
                    df["clean_text"]
                )
            )

            predictions = (
                models[
                    "emotion_model"
                ]
                .predict(X)
            )

            df["emotion"] = (
                pd.Series(
                    predictions,
                    index=df.index
                )
                .apply(
                    normalize_emotion
                )
            )

            try:

                probabilities = (
                    models[
                        "emotion_model"
                    ]
                    .predict_proba(X)
                )

                df[
                    "emotion_confidence"
                ] = (
                    probabilities
                    .max(axis=1)
                )

            except Exception:

                df[
                    "emotion_confidence"
                ] = np.nan

        except Exception:

            df["emotion"] = "Unknown"

            df[
                "emotion_confidence"
            ] = np.nan

    else:

        df["emotion"] = "Unknown"

        df[
            "emotion_confidence"
        ] = np.nan


    # --------------------------------------------------------
    # TOPICS
    # --------------------------------------------------------

    if (
        models["topic_model"]
        is not None
        and
        models["topic_vectorizer"]
        is not None
    ):

        try:

            X = (
                models[
                    "topic_vectorizer"
                ]
                .transform(
                    df["clean_text"]
                )
            )

            topic_probabilities = (
                models[
                    "topic_model"
                ]
                .transform(X)
            )

            topic_ids = (
                topic_probabilities
                .argmax(axis=1)
                + 1
            )

            df["topic_id"] = topic_ids

            df["topic"] = (
                df["topic_id"]
                .map(TOPIC_NAMES)
                .fillna(
                    df["topic_id"]
                    .apply(
                        lambda x:
                        f"Topic {x}"
                    )
                )
            )

            df[
                "topic_confidence"
            ] = (
                topic_probabilities
                .max(axis=1)
            )

        except Exception:

            df["topic_id"] = 0

            df["topic"] = (
                "Unknown"
            )

            df[
                "topic_confidence"
            ] = np.nan

    else:

        df["topic_id"] = 0

        df["topic"] = "Unknown"

        df[
            "topic_confidence"
        ] = np.nan


    # --------------------------------------------------------
    # TREND SCORE
    # --------------------------------------------------------

    confidence = (
        df[
            "sentiment_confidence"
        ]
        .fillna(0.5)
    )

    df["trend_score"] = (
        df["engagement"]
        *
        (
            0.5
            +
            confidence * 0.5
        )
    )


    # --------------------------------------------------------
    # TREND SCORE NORMALIZED
    # --------------------------------------------------------

    max_score = (
        df["trend_score"]
        .max()
    )

    if (
        pd.notna(max_score)
        and
        max_score > 0
    ):

        df["trend_score_normalized"] = (
            df["trend_score"]
            /
            max_score
        ) * 100

    else:

        df[
            "trend_score_normalized"
        ] = 0


    return df


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div style="
        padding:8px 0 22px;
    ">

        <div style="
            font-size:27px;
            font-weight:850;
            color:#17132b;
        ">
            TrendSense AI
        </div>

        <div style="
            color:#667085;
            font-size:12px;
            margin-top:5px;
        ">
            Social Intelligence Platform
        </div>

    </div>
    """)

    st.markdown("### 📂 Data Source")

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
            type=["csv"],
            help=(
                "Works with YouTube, Instagram, "
                "Twitter/X, Reddit and generic "
                "social-media datasets."
            )
        )

    st.markdown("---")

    st.markdown("### ⚙️ Filters")

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


# ============================================================
# LOAD DATA
# ============================================================

if (
    source == "Upload CSV"
    and
    uploaded_file is not None
):

    try:

        raw_df = pd.read_csv(
            uploaded_file
        )

        filename = (
            uploaded_file.name
        )

        data_source_label = (
            filename
        )

    except Exception as error:

        st.error(
            f"Could not read CSV: {error}"
        )

        st.stop()

else:

    if not os.path.exists(
        DATA_PATH
    ):

        st.error(
            "Demo dataset not found. "
            "Upload a CSV dataset instead."
        )

        st.stop()

    raw_df = pd.read_csv(
        DATA_PATH
    )

    filename = (
        "social_media.csv"
    )

    data_source_label = (
        "Demo Dataset"
    )


# ============================================================
# STANDARDIZE
# ============================================================

try:

    df, detected_platforms = (
        standardize_social_data(
            raw_df,
            filename
        )
    )

except Exception as error:

    st.error(
        f"Dataset standardization failed: {error}"
    )

    st.info(
        "Make sure your CSV contains a text/content/comment/post "
        "column."
    )

    st.stop()


# ============================================================
# AI PIPELINE
# ============================================================

with st.spinner(
    "Running TrendSense AI analysis..."
):

    df = run_ai_analysis(
        df
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if search_query.strip():

    filtered_df = filtered_df[
        filtered_df["text"]
        .str.contains(
            re.escape(
                search_query.strip()
            ),
            case=False,
            na=False
        )
    ]


if sentiment_filter:

    filtered_df = filtered_df[
        filtered_df[
            "sentiment"
        ].isin(
            sentiment_filter
        )
    ]


if emotion_filter:

    filtered_df = filtered_df[
        filtered_df[
            "emotion"
        ].isin(
            emotion_filter
        )
    ]


filtered_df = filtered_df[
    filtered_df["likes"]
    >=
    min_likes
]


# ============================================================
# HERO
# ============================================================

platform_text = " • ".join(
    detected_platforms
)

render_html(
    f"""
    <div class="hero">

        <div class="hero-title">
            <span class="hero-gradient">
                TrendSense AI
            </span>
        </div>

        <div class="hero-sub">

            Turn social-media conversations
            into intelligent business insights.

            TrendSense detects sentiment,
            emotions, topics, hashtags,
            engagement patterns and emerging trends
            across multiple platforms.

        </div>

        <div class="source-pill">

            📂 {html.escape(str(data_source_label))}
            &nbsp; • &nbsp;
            📱 {html.escape(platform_text)}
            &nbsp; • &nbsp;
            {len(df):,} records

        </div>

    </div>
    """
)


# ============================================================
# PIPELINE VISUAL
# ============================================================

render_html("""
<div class="card">

    <div class="card-title">
        🚀 TrendSense Intelligence Pipeline
    </div>

    <div class="card-text">

        CSV
        →
        Platform Detection
        →
        NLP
        →
        Sentiment
        →
        Emotion
        →
        Topics
        →
        Hashtags
        →
        Trend Score
        →
        Platform Comparison
        →
        Business Insights

    </div>

</div>
""")


# ============================================================
# DATA STATUS
# ============================================================

render_html(
    f"""
    <div class="success-box">

        <b>✓ Dataset connected successfully</b><br>

        Detected platform:
        <b>{html.escape(platform_text)}</b>
        &nbsp; • &nbsp;

        Raw records:
        <b>{len(raw_df):,}</b>
        &nbsp; • &nbsp;

        Analyzed records:
        <b>{len(df):,}</b>

    </div>
    """
)


# ============================================================
# KPI
# ============================================================

total_posts = len(
    filtered_df
)

total_likes = (
    filtered_df["likes"]
    .sum()
)

total_comments = (
    filtered_df["comments"]
    .sum()
)

total_engagement = (
    filtered_df["engagement"]
    .sum()
)

columns = st.columns(4)


with columns[0]:

    render_html(
        kpi_card(
            "Posts",
            compact_number(
                total_posts
            ),
            "Filtered records"
        )
    )


with columns[1]:

    render_html(
        kpi_card(
            "Likes",
            compact_number(
                total_likes
            ),
            "Total likes"
        )
    )


with columns[2]:

    render_html(
        kpi_card(
            "Engagement",
            compact_number(
                total_engagement
            ),
            "Likes + comments + shares"
        )
    )


with columns[3]:

    avg_trend = (
        filtered_df[
            "trend_score_normalized"
        ].mean()
        if len(filtered_df)
        else 0
    )

    render_html(
        kpi_card(
            "Trend Index",
            f"{avg_trend:.1f}",
            "Average trend score"
        )
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 Overview",
        "💭 Sentiment & Emotion",
        "🔥 Topics & Trends",
        "🌐 Platform Comparison",
        "💼 Business Insights",
        "🧪 Model Performance",
        "🔎 Post Explorer",
        "📄 Reports"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:

    render_html(
        """
        <div class="section-title">
            Social Intelligence Overview
        </div>

        <div class="section-sub">
            A high-level view of what your audience
            is talking about and how strongly
            they are engaging.
        </div>
        """
    )

    col1, col2 = st.columns(2)


    with col1:

        sentiment_df = (
            filtered_df[
                "sentiment"
            ]
            .value_counts()
            .reindex(
                SENTIMENT_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        sentiment_df.columns = [
            "Sentiment",
            "Count"
        ]

        fig = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            hole=0.58,
            title="Sentiment Distribution"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        emotion_df = (
            filtered_df[
                "emotion"
            ]
            .value_counts()
            .reset_index()
        )

        emotion_df.columns = [
            "Emotion",
            "Count"
        ]

        fig = px.bar(
            emotion_df,
            x="Emotion",
            y="Count",
            title="Emotion Distribution"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Engagement over time

    date_df = (
        filtered_df
        .dropna(
            subset=["date"]
        )
    )

    if len(date_df):

        trend_df = (
            date_df
            .groupby(
                "date",
                as_index=False
            )[
                "engagement"
            ]
            .sum()
            .sort_values(
                "date"
            )
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
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 2 — SENTIMENT & EMOTION
# ============================================================

with tabs[1]:

    render_html(
        """
        <div class="section-title">
            💭 Audience Sentiment & Emotion
        </div>

        <div class="section-sub">
            Understand how people feel about the
            conversations represented in your dataset.
        </div>
        """
    )

    col1, col2 = st.columns(2)


    with col1:

        sentiment_df = (
            filtered_df[
                "sentiment"
            ]
            .value_counts()
            .reindex(
                SENTIMENT_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        sentiment_df.columns = [
            "Sentiment",
            "Posts"
        ]

        fig = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Posts",
            title="Sentiment Count"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        emotion_df = (
            filtered_df[
                "emotion"
            ]
            .value_counts()
            .reset_index()
        )

        emotion_df.columns = [
            "Emotion",
            "Posts"
        ]

        fig = px.bar(
            emotion_df,
            x="Emotion",
            y="Posts",
            title="Emotion Count"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Sentiment by platform

    platform_sentiment = (
        filtered_df
        .groupby(
            [
                "platform",
                "sentiment"
            ]
        )
        .size()
        .reset_index(
            name="Posts"
        )
    )

    if len(platform_sentiment):

        fig = px.bar(
            platform_sentiment,
            x="platform",
            y="Posts",
            color="sentiment",
            barmode="group",
            title="Sentiment by Platform"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 3 — TOPICS & TRENDS
# ============================================================

with tabs[2]:

    render_html(
        """
        <div class="section-title">
            🔥 Topics, Hashtags & Trends
        </div>

        <div class="section-sub">
            Identify what conversations are gaining
            attention and which themes are driving engagement.
        </div>
        """
    )

    col1, col2 = st.columns(2)


    with col1:

        topic_df = (
            filtered_df[
                "topic"
            ]
            .value_counts()
            .reset_index()
        )

        topic_df.columns = [
            "Topic",
            "Posts"
        ]

        fig = px.bar(
            topic_df,
            x="Posts",
            y="Topic",
            orientation="h",
            title="Topic Distribution"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=440
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        topic_engagement = (
            filtered_df
            .groupby(
                "topic",
                as_index=False
            )[
                "engagement"
            ]
            .sum()
            .sort_values(
                "engagement",
                ascending=False
            )
        )

        fig = px.bar(
            topic_engagement,
            x="topic",
            y="engagement",
            title="Engagement by Topic"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=440
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # HASHTAGS

    render_html(
        """
        <div class="section-title">
            #️⃣ Trending Hashtags
        </div>
        """
    )

    all_hashtags = []

    for tags in filtered_df[
        "hashtags"
    ]:

        all_hashtags.extend(
            tags
        )

    if all_hashtags:

        hashtag_df = (
            pd.Series(
                all_hashtags,
                name="Hashtag"
            )
            .value_counts()
            .head(20)
            .reset_index()
        )

        hashtag_df.columns = [
            "Hashtag",
            "Posts"
        ]

        fig = px.bar(
            hashtag_df,
            x="Posts",
            y="Hashtag",
            orientation="h",
            title="Top 20 Hashtags"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No hashtags were detected in this dataset."
        )


    # TOP TRENDING POSTS

    render_html(
        """
        <div class="section-title">
            🚀 Top Trending Posts
        </div>
        """
    )

    top_posts = (
        filtered_df
        .sort_values(
            "trend_score",
            ascending=False
        )
        .head(10)
        [
            [
                "platform",
                "text",
                "sentiment",
                "emotion",
                "topic",
                "engagement",
                "trend_score_normalized"
            ]
        ]
    )

    top_posts = top_posts.copy()

    top_posts[
        "trend_score_normalized"
    ] = top_posts[
        "trend_score_normalized"
    ].round(2)

    st.dataframe(
        top_posts,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 4 — PLATFORM COMPARISON
# ============================================================

with tabs[3]:

    render_html(
        """
        <div class="section-title">
            🌐 Platform Comparison
        </div>

        <div class="section-sub">
            Compare how different social platforms
            perform in terms of conversation,
            engagement and sentiment.
        </div>
        """
    )

    platform_summary = (
        filtered_df
        .groupby(
            "platform",
            as_index=False
        )
        .agg(
            Posts=(
                "text",
                "count"
            ),
            Likes=(
                "likes",
                "sum"
            ),
            Comments=(
                "comments",
                "sum"
            ),
            Shares=(
                "shares",
                "sum"
            ),
            Engagement=(
                "engagement",
                "sum"
            ),
            Average_Trend=(
                "trend_score_normalized",
                "mean"
            )
        )
    )

    platform_summary[
        "Average_Trend"
    ] = platform_summary[
        "Average_Trend"
    ].round(2)


    if len(platform_summary):

        st.dataframe(
            platform_summary,
            use_container_width=True,
            hide_index=True
        )


        col1, col2 = st.columns(2)


        with col1:

            fig = px.bar(
                platform_summary,
                x="platform",
                y="Engagement",
                title="Total Engagement by Platform"
            )

            fig.update_layout(
                PLOTLY_LAYOUT,
                height=420
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            fig = px.bar(
                platform_summary,
                x="platform",
                y="Average_Trend",
                title="Average Trend Index"
            )

            fig.update_layout(
                PLOTLY_LAYOUT,
                height=420
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # Platform sentiment

    platform_sentiment = (
        filtered_df
        .groupby(
            [
                "platform",
                "sentiment"
            ]
        )
        .size()
        .reset_index(
            name="Posts"
        )
    )

    if len(platform_sentiment):

        fig = px.bar(
            platform_sentiment,
            x="platform",
            y="Posts",
            color="sentiment",
            barmode="group",
            title="Audience Sentiment Across Platforms"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 5 — BUSINESS INSIGHTS
# ============================================================

with tabs[4]:

    render_html(
        """
        <div class="section-title">
            💼 Business Intelligence
        </div>

        <div class="section-sub">
            Convert social-media signals into
            practical decisions for brands,
            marketers and businesses.
        </div>
        """
    )

    if len(filtered_df):

        sentiment_counts = (
            filtered_df[
                "sentiment"
            ]
            .value_counts()
        )

        emotion_counts = (
            filtered_df[
                "emotion"
            ]
            .value_counts()
        )

        topic_counts = (
            filtered_df[
                "topic"
            ]
            .value_counts()
        )

        platform_counts = (
            filtered_df[
                "platform"
            ]
            .value_counts()
        )


        dominant_sentiment = (
            sentiment_counts
            .index[0]
            if len(sentiment_counts)
            else "Unknown"
        )

        dominant_emotion = (
            emotion_counts
            .index[0]
            if len(emotion_counts)
            else "Unknown"
        )

        dominant_topic = (
            topic_counts
            .index[0]
            if len(topic_counts)
            else "Unknown"
        )

        strongest_platform = (
            platform_counts
            .index[0]
            if len(platform_counts)
            else "Unknown"
        )


        # ----------------------------------------------------
        # INSIGHT 1
        # ----------------------------------------------------

        render_html(
            f"""
            <div class="card">

                <div class="card-title">
                    🎯 Audience Mood
                </div>

                <div class="card-text">

                    The dominant sentiment is
                    <b>{html.escape(dominant_sentiment)}</b>,
                    while the most common emotion is
                    <b>{html.escape(dominant_emotion)}</b>.

                    This can help a business understand
                    the current audience response.

                </div>

            </div>
            """
        )


        # ----------------------------------------------------
        # INSIGHT 2
        # ----------------------------------------------------

        render_html(
            f"""
            <div class="card">

                <div class="card-title">
                    🔥 Conversation Opportunity
                </div>

                <div class="card-text">

                    The leading conversation theme is
                    <b>{html.escape(dominant_topic)}</b>.

                    Brands can use this signal to identify
                    content opportunities, campaigns,
                    products or messaging aligned with
                    audience interests.

                </div>

            </div>
            """
        )


        # ----------------------------------------------------
        # INSIGHT 3
        # ----------------------------------------------------

        render_html(
            f"""
            <div class="card">

                <div class="card-title">
                    📱 Strongest Platform
                </div>

                <div class="card-text">

                    <b>{html.escape(strongest_platform)}</b>
                    currently has the largest number
                    of conversations in the filtered dataset.

                    Businesses can compare this with
                    engagement before deciding where
                    to prioritize campaigns.

                </div>

            </div>
            """
        )


        # ----------------------------------------------------
        # TRENDING OPPORTUNITY
        # ----------------------------------------------------

        top_trend = (
            filtered_df
            .sort_values(
                "trend_score",
                ascending=False
            )
            .head(1)
        )

        if len(top_trend):

            post = top_trend.iloc[0]

            render_html(
                f"""
                <div class="info-box">

                    <b>🚀 Highest Trend Signal</b><br><br>

                    Platform:
                    <b>{html.escape(str(post["platform"]))}</b>
                    <br>

                    Sentiment:
                    <b>{html.escape(str(post["sentiment"]))}</b>
                    <br>

                    Topic:
                    <b>{html.escape(str(post["topic"]))}</b>
                    <br>

                    Trend Index:
                    <b>{post["trend_score_normalized"]:.1f}</b>

                </div>
                """
            )


        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        render_html(
            """
            <div class="section-title">
                💡 Recommended Business Actions
            </div>
            """
        )


        actions = [

            (
                "Content Strategy",
                "Create content around the strongest topics "
                "and hashtags."
            ),

            (
                "Audience Monitoring",
                "Track negative sentiment and sudden "
                "emotion changes."
            ),

            (
                "Platform Strategy",
                "Prioritize platforms with stronger "
                "engagement rather than relying only "
                "on audience size."
            ),

            (
                "Trend Detection",
                "Monitor high trend-score posts to identify "
                "emerging conversations early."
            ),

            (
                "Campaign Optimization",
                "Compare platform performance before "
                "allocating campaign resources."
            ),
        ]


        for title, description in actions:

            render_html(
                f"""
                <div class="card">

                    <div class="card-title">
                        {html.escape(title)}
                    </div>

                    <div class="card-text">
                        {html.escape(description)}
                    </div>

                </div>
                """
            )


# ============================================================
# TAB 6 — MODEL PERFORMANCE
# ============================================================

with tabs[5]:

    render_html(
        """
        <div class="section-title">
            🧪 Model Performance
        </div>

        <div class="section-sub">
            Evaluation metrics for the trained
            machine-learning models.
        </div>
        """
    )


    def evaluate_saved_model(
        test_path,
        model_path,
        vectorizer_path,
        task
    ):

        if not all(
            os.path.exists(path)
            for path in [
                test_path,
                model_path,
                vectorizer_path
            ]
        ):

            return None

        try:

            test_df = pd.read_csv(
                test_path
            )

            text_col = detect_column(
                test_df,
                [
                    "text",
                    "tweet",
                    "content",
                    "comment",
                    "post",
                    "sentence"
                ]
            )

            label_col = detect_column(
                test_df,
                [
                    "label",
                    "sentiment",
                    "emotion",
                    "target",
                    "class"
                ]
            )

            if (
                text_col is None
                or
                label_col is None
            ):

                return None

            test_df = (
                test_df[
                    [
                        text_col,
                        label_col
                    ]
                ]
                .dropna()
            )

            model = joblib.load(
                model_path
            )

            vectorizer = joblib.load(
                vectorizer_path
            )

            X = vectorizer.transform(
                test_df[text_col]
                .astype(str)
                .apply(clean_text)
            )

            predictions = (
                model.predict(X)
            )

            if task == "sentiment":

                y_true = (
                    test_df[label_col]
                    .apply(
                        normalize_sentiment
                    )
                )

                y_pred = (
                    pd.Series(
                        predictions
                    )
                    .apply(
                        normalize_sentiment
                    )
                )

                labels = (
                    SENTIMENT_ORDER
                )

            else:

                y_true = (
                    test_df[label_col]
                    .apply(
                        normalize_emotion
                    )
                )

                y_pred = (
                    pd.Series(
                        predictions
                    )
                    .apply(
                        normalize_emotion
                    )
                )

                labels = (
                    EMOTION_ORDER
                )

            valid = (
                y_true.isin(labels)
                &
                y_pred.isin(labels)
            )

            y_true = y_true[valid]
            y_pred = y_pred[valid]

            if len(y_true) == 0:
                return None

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            )

            f2 = fbeta_score(
                y_true,
                y_pred,
                labels=labels,
                beta=2,
                average="weighted",
                zero_division=0
            )

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels
            )

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "cm": cm,
                "labels": labels,
                "samples": len(y_true),
            }

        except Exception:

            return None


    sentiment_metrics = (
        evaluate_saved_model(
            SENTIMENT_TEST_PATH,
            SENTIMENT_MODEL_PATH,
            SENTIMENT_VECTOR_PATH,
            "sentiment"
        )
    )


    emotion_metrics = (
        evaluate_saved_model(
            EMOTION_TEST_PATH,
            EMOTION_MODEL_PATH,
            EMOTION_VECTOR_PATH,
            "emotion"
        )
    )


    if sentiment_metrics:

        render_html(
            """
            <div class="section-title">
                Sentiment Model
            </div>
            """
        )

        cols = st.columns(5)

        metric_names = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1"),
            ("F2 Score", "f2"),
        ]

        for col, (
            label,
            key
        ) in zip(
            cols,
            metric_names
        ):

            with col:

                render_html(
                    kpi_card(
                        label,
                        f"{sentiment_metrics[key] * 100:.1f}%",
                        "Test-set metric"
                    )
                )

        cm = sentiment_metrics["cm"]

        fig = px.imshow(
            cm,
            text_auto=True,
            x=sentiment_metrics[
                "labels"
            ],
            y=sentiment_metrics[
                "labels"
            ],
            title="Sentiment Confusion Matrix"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Sentiment evaluation dataset "
            "is not available."
        )


    if emotion_metrics:

        render_html(
            """
            <div class="section-title">
                Emotion Model
            </div>
            """
        )

        cols = st.columns(5)

        for col, (
            label,
            key
        ) in zip(
            cols,
            metric_names
        ):

            with col:

                render_html(
                    kpi_card(
                        label,
                        f"{emotion_metrics[key] * 100:.1f}%",
                        "Test-set metric"
                    )
                )

        cm = emotion_metrics["cm"]

        fig = px.imshow(
            cm,
            text_auto=True,
            x=emotion_metrics[
                "labels"
            ],
            y=emotion_metrics[
                "labels"
            ],
            title="Emotion Confusion Matrix"
        )

        fig.update_layout(
            PLOTLY_LAYOUT,
            height=480
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Emotion evaluation dataset "
            "is not available."
        )


# ============================================================
# TAB 7 — POST EXPLORER
# ============================================================

with tabs[6]:

    render_html(
        """
        <div class="section-title">
            🔎 Post Explorer
        </div>

        <div class="section-sub">
            Search and inspect individual posts
            and their AI-generated intelligence.
        </div>
        """
    )


    display_columns = [

        "post_id",
        "platform",
        "text",
        "sentiment",
        "emotion",
        "topic",
        "likes",
        "comments",
        "shares",
        "views",
        "engagement",
        "trend_score_normalized",
        "date"
    ]


    available_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]


    explorer_df = (
        filtered_df[
            available_columns
        ]
        .sort_values(
            "trend_score_normalized",
            ascending=False
        )
    )


    st.dataframe(
        explorer_df,
        use_container_width=True,
        hide_index=True
    )


    if len(filtered_df):

        st.markdown(
            "### 📌 Inspect a Post"
        )

        selected_index = st.selectbox(
            "Choose a post",
            filtered_df.index,
            format_func=lambda x:
            str(
                filtered_df.loc[
                    x,
                    "text"
                ]
            )[:100]
        )

        selected = (
            filtered_df
            .loc[
                selected_index
            ]
        )


        col1, col2 = st.columns(2)


        with col1:

            render_html(
                f"""
                <div class="card">

                    <div class="card-title">
                        📝 Post
                    </div>

                    <div class="card-text">
                        {html.escape(
                            str(selected["text"])
                        )}
                    </div>

                </div>
                """
            )


        with col2:

            render_html(
                f"""
                <div class="card">

                    <div class="card-title">
                        🧠 AI Analysis
                    </div>

                    <div class="card-text">

                        Platform:
                        <b>{html.escape(str(selected["platform"]))}</b>
                        <br>

                        Sentiment:
                        <b>{html.escape(str(selected["sentiment"]))}</b>
                        <br>

                        Emotion:
                        <b>{html.escape(str(selected["emotion"]))}</b>
                        <br>

                        Topic:
                        <b>{html.escape(str(selected["topic"]))}</b>
                        <br>

                        Trend Score:
                        <b>{selected["trend_score_normalized"]:.1f}</b>

                    </div>

                </div>
                """
            )


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def create_excel_report(
    report_df
):

    output = io.BytesIO()

    if report_df is None:

        report_df = pd.DataFrame()

    report_df = (
        report_df.copy()
    )

    if (
        report_df.empty
        and
        len(report_df.columns) == 0
    ):

        report_df = pd.DataFrame({
            "Message": [
                "No report data available."
            ]
        })

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


def create_pdf_report(
    report_df
):

    try:

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

    except Exception:

        return None


    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = (
        getSampleStyleSheet()
    )

    story = []

    story.append(
        Paragraph(
            "TrendSense AI Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "AI-powered social-media intelligence report.",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )


    if report_df is None:
        report_df = pd.DataFrame()


    if len(report_df):

        small_df = (
            report_df
            .head(50)
            .copy()
        )

        small_df = small_df.fillna("")

        headers = list(
            small_df.columns
        )

        data = [
            headers
        ]

        for _, row in small_df.iterrows():

            data.append(
                [
                    str(
                        value
                    )[:80]
                    for value
                    in row.tolist()
                ]
            )

        table = Table(
            data,
            repeatRows=1
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#6d28d9"
                        )
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                ]
            )
        )

        story.append(
            table
        )

    else:

        story.append(
            Paragraph(
                "No report data available.",
                styles["BodyText"]
            )
        )


    document.build(
        story
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# TAB 8 — REPORTS
# ============================================================

with tabs[7]:

    render_html(
        """
        <div class="section-title">
            📄 TrendSense Reports
        </div>

        <div class="section-sub">
            Export your analyzed dataset and
            business intelligence results.
        </div>
        """
    )


    report_columns = [

        "post_id",
        "platform",
        "date",
        "text",
        "sentiment",
        "emotion",
        "topic",
        "likes",
        "comments",
        "shares",
        "views",
        "engagement",
        "engagement_rate",
        "trend_score_normalized"
    ]


    report_columns = [
        column
        for column in report_columns
        if column in filtered_df.columns
    ]


    report_df = (
        filtered_df[
            report_columns
        ]
        .copy()
    )


    col1, col2 = st.columns(2)


    with col1:

        excel_bytes = (
            create_excel_report(
                report_df
            )
        )

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_bytes,
            file_name=(
                "TrendSense_Report.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )


    with col2:

        pdf_bytes = (
            create_pdf_report(
                report_df
            )
        )

        if pdf_bytes:

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=(
                    "TrendSense_Report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )


    csv_bytes = (
        report_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(
        label="⬇️ Download Analyzed CSV",
        data=csv_bytes,
        file_name=(
            "TrendSense_Analyzed_Data.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


    render_html(
        """
        <div class="info-box">

            <b>💡 Sponsor-ready workflow</b><br><br>

            Upload a real historical social-media dataset,
            let TrendSense automatically detect the platform
            and standardize the columns, then generate
            AI-powered sentiment, emotion, topic,
            hashtag, trend and business insights.

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">

        TrendSense AI • Social Intelligence Platform
        <br>
        Built for AI-powered trend discovery,
        analytics and decision intelligence.

    </div>
    """
)
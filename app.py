import os
import io
import re
import json
import html
import urllib.parse
import urllib.request
import urllib.error
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

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
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_PATH = os.path.join(
    DATA_DIR,
    "social_media.csv"
)

SENTIMENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "sentiment_model.pkl"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

TOPIC_MODEL_PATH = os.path.join(
    BASE_DIR,
    "topic_model.pkl"
)

TOPIC_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "topic_vectorizer.pkl"
)

EMOTION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "emotion_model.pkl"
)

EMOTION_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "emotion_vectorizer.pkl"
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
    "Positive",
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
    "Surprise",
]


TOPIC_NAMES = {
    1: "AI Automation & Productivity",
    2: "ChatGPT, Coding & Education",
    3: "AI, Jobs & Future Technology",
}


# ============================================================
# UI HELPERS
# ============================================================

def render_html(content):

    try:
        st.html(content)

    except Exception:
        st.markdown(
            content,
            unsafe_allow_html=True
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


def normalize_label(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def parse_metric(value):

    if pd.isna(value):
        return 0

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):
        return float(value)

    value = str(value).strip().upper()

    if not value:
        return 0

    multiplier = 1

    if value.endswith("K"):

        multiplier = 1_000
        value = value[:-1]

    elif value.endswith("M"):

        multiplier = 1_000_000
        value = value[:-1]

    elif value.endswith("B"):

        multiplier = 1_000_000_000
        value = value[:-1]

    value = value.replace(",", "")

    try:
        return float(value) * multiplier

    except Exception:
        return 0


def extract_hashtags(text):

    if pd.isna(text):
        return ""

    tags = re.findall(
        r"#([A-Za-z0-9_]+)",
        str(text)
    )

    return ", ".join(
        f"#{tag}" for tag in tags
    )


def clean_fallback(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    """
    Detect the language of social media text.
    Returns a readable language name.
    """

    if pd.isna(text):
        return "Unknown"

    text = str(text).strip()

    if len(text) < 3:
        return "Unknown"

    try:

        language_code = detect(text)

        language_map = {

            "en": "English",
            "hi": "Hindi",
            "mr": "Marathi",
            "gu": "Gujarati",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "ur": "Urdu",

            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",

            "ja": "Japanese",
            "ko": "Korean",
            "zh-cn": "Chinese",
            "zh-tw": "Chinese",
            "ar": "Arabic",
        }

        return language_map.get(
            language_code,
            "Other"
        )

    except Exception:

        return "Unknown"


# ============================================================
# KPI CARD
# ============================================================

def kpi_card(
    title,
    value,
    subtitle=""
):

    return f"""
    <div style="
        background:linear-gradient(135deg,#ffffff,#f7f5ff);
        border:1px solid #e9e5ff;
        border-radius:18px;
        padding:20px;
        box-shadow:0 8px 25px rgba(79,70,229,.07);
        min-height:125px;
    ">

        <div style="
            font-size:13px;
            color:#6b7280;
            font-weight:600;
            margin-bottom:8px;
        ">
            {html.escape(str(title))}
        </div>

        <div style="
            font-size:30px;
            font-weight:800;
            color:#111827;
        ">
            {html.escape(str(value))}
        </div>

        <div style="
            font-size:12px;
            color:#8b8fa3;
            margin-top:6px;
        ">
            {html.escape(str(subtitle))}
        </div>

    </div>
    """


# ============================================================
# PLATFORM DETECTION
# ============================================================

def detect_platform(df):

    columns = {
        str(c).lower()
        for c in df.columns
    }

    if {
        "video_id",
        "channel_title",
    } & columns:

        return "YouTube"

    if {
        "channel_id",
        "video_views",
    } & columns:

        return "YouTube"

    if {
        "shortcode",
        "owner_username",
    } & columns:

        return "Instagram"

    if {
        "save_count",
        "media_id",
    } & columns:

        return "Instagram"

    if {
        "tweet_id",
        "retweet_count",
    } & columns:

        return "Twitter/X"

    if {
        "favorite_count",
        "user_followers",
    } & columns:

        return "Twitter/X"

    if {
        "subreddit",
        "upvote_ratio",
    } & columns:

        return "Reddit"

    if "reddit_id" in columns:

        return "Reddit"

    return "Unknown"


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_column(
    df,
    candidates
):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    return None


TEXT_COLUMNS = [
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
    "title",
]


LIKE_COLUMNS = [
    "likes",
    "like",
    "likes_count",
    "like_count",
    "favorite_count",
    "favourites",
    "favorites",
    "reactions",
    "total_likes",
]


COMMENT_COLUMNS = [
    "comments",
    "comment_count",
    "comments_count",
    "num_comments",
    "replies",
    "reply_count",
]


SHARE_COLUMNS = [
    "shares",
    "share",
    "shares_count",
    "retweets",
    "retweet_count",
    "reposts",
    "repost_count",
]


VIEW_COLUMNS = [
    "views",
    "view_count",
    "views_count",
    "video_views",
    "play_count",
    "plays",
    "impressions",
]


DATE_COLUMNS = [
    "date",
    "created_at",
    "timestamp",
    "time",
    "datetime",
    "published_at",
    "published",
    "upload_date",
    "posted_at",
]


FOLLOWER_COLUMNS = [
    "author_followers",
    "followers",
    "follower_count",
    "user_followers",
    "subscribers",
    "subscriber_count",
]


LANGUAGE_COLUMNS = [
    "language",
    "lang",
]


URL_COLUMNS = [
    "source_url",
    "url",
    "link",
    "permalink",
]


TYPE_COLUMNS = [
    "post_type",
    "content_type",
    "media_type",
    "type",
]


# ============================================================
# UNIVERSAL DATA STANDARDIZATION
# ============================================================

def standardize_social_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # DETECT COLUMNS
    # --------------------------------------------------------

    text_col = detect_column(
        df,
        TEXT_COLUMNS
    )

    likes_col = detect_column(
        df,
        LIKE_COLUMNS
    )

    comments_col = detect_column(
        df,
        COMMENT_COLUMNS
    )

    shares_col = detect_column(
        df,
        SHARE_COLUMNS
    )

    views_col = detect_column(
        df,
        VIEW_COLUMNS
    )

    date_col = detect_column(
        df,
        DATE_COLUMNS
    )

    follower_col = detect_column(
        df,
        FOLLOWER_COLUMNS
    )

    language_col = detect_column(
        df,
        LANGUAGE_COLUMNS
    )

    url_col = detect_column(
        df,
        URL_COLUMNS
    )

    type_col = detect_column(
        df,
        TYPE_COLUMNS
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if text_col:

        df["text"] = (
            df[text_col]
            .fillna("")
            .astype(str)
        )

    else:

        df["text"] = ""

    # --------------------------------------------------------
    # PLATFORM
    # --------------------------------------------------------

    if "platform" not in df.columns:

        df["platform"] = detect_platform(df)

    else:

        df["platform"] = (
            df["platform"]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # POST ID
    # --------------------------------------------------------

    if "post_id" not in df.columns:

        df["post_id"] = np.arange(
            1,
            len(df) + 1
        )

    # --------------------------------------------------------
    # POST TYPE
    # --------------------------------------------------------

    if type_col:

        df["post_type"] = (
            df[type_col]
            .fillna("post")
            .astype(str)
        )

    else:

        df["post_type"] = "post"

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if date_col:

        df["date"] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

    else:

        df["date"] = pd.NaT

    # --------------------------------------------------------
    # LIKES
    # --------------------------------------------------------

    if likes_col:

        df["likes"] = (
            df[likes_col]
            .apply(parse_metric)
        )

    else:

        df["likes"] = 0

    # --------------------------------------------------------
    # COMMENTS
    # --------------------------------------------------------

    if comments_col:

        df["comments"] = (
            df[comments_col]
            .apply(parse_metric)
        )

    else:

        df["comments"] = 0

    # --------------------------------------------------------
    # SHARES
    # --------------------------------------------------------

    if shares_col:

        df["shares"] = (
            df[shares_col]
            .apply(parse_metric)
        )

    else:

        df["shares"] = 0

    # --------------------------------------------------------
    # VIEWS
    # --------------------------------------------------------

    if views_col:

        df["views"] = (
            df[views_col]
            .apply(parse_metric)
        )

    else:

        df["views"] = 0

    # --------------------------------------------------------
    # FOLLOWERS
    # --------------------------------------------------------

    if follower_col:

        df["author_followers"] = (
            df[follower_col]
            .apply(parse_metric)
        )

    else:

        df["author_followers"] = 0

    # --------------------------------------------------------
    # SOURCE URL
    # --------------------------------------------------------

    if url_col:

        df["source_url"] = (
            df[url_col]
            .fillna("")
            .astype(str)
        )

    else:

        df["source_url"] = ""

    # --------------------------------------------------------
    # HASHTAGS
    # --------------------------------------------------------

    if "hashtags" not in df.columns:

        df["hashtags"] = (
            df["text"]
            .apply(extract_hashtags)
        )

    else:

        df["hashtags"] = (
            df["hashtags"]
            .fillna("")
            .astype(str)
        )

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    df["clean_text"] = (
        df["text"]
        .apply(clean_text)
    )

    # --------------------------------------------------------
    # LANGUAGE DETECTION
    # --------------------------------------------------------

    if language_col:

        df["language"] = (
            df[language_col]
            .fillna("")
            .astype(str)
        )

        df["language"] = df.apply(
            lambda row:
            row["language"]
            if (
                row["language"].strip()
                and row["language"].lower()
                not in [
                    "unknown",
                    "nan",
                    "none",
                    ""
                ]
            )
            else detect_language(
                row["clean_text"]
            ),
            axis=1
        )

    else:

        df["language"] = (
            df["clean_text"]
            .apply(detect_language)
        )

    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    df["engagement"] = (
        df["likes"]
        + df["comments"]
        + df["shares"]
    )

    # --------------------------------------------------------
    # ENGAGEMENT RATE
    # --------------------------------------------------------

    denominator = (
        df["views"]
        .replace(
            0,
            np.nan
        )
    )

    df["engagement_rate"] = (
        df["engagement"]
        / denominator
        * 100
    )

    df["engagement_rate"] = (
        df["engagement_rate"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    sentiment_model = None
    vectorizer = None

    topic_model = None
    topic_vectorizer = None

    emotion_model = None
    emotion_vectorizer = None

    try:

        if os.path.exists(
            SENTIMENT_MODEL_PATH
        ):

            sentiment_model = joblib.load(
                SENTIMENT_MODEL_PATH
            )

    except Exception:
        pass

    try:

        if os.path.exists(
            TFIDF_PATH
        ):

            vectorizer = joblib.load(
                TFIDF_PATH
            )

    except Exception:
        pass

    try:

        if os.path.exists(
            TOPIC_MODEL_PATH
        ):

            topic_model = joblib.load(
                TOPIC_MODEL_PATH
            )

    except Exception:
        pass

    try:

        if os.path.exists(
            TOPIC_VECTORIZER_PATH
        ):

            topic_vectorizer = joblib.load(
                TOPIC_VECTORIZER_PATH
            )

    except Exception:
        pass

    try:

        if os.path.exists(
            EMOTION_MODEL_PATH
        ):

            emotion_model = joblib.load(
                EMOTION_MODEL_PATH
            )

    except Exception:
        pass

    try:

        if os.path.exists(
            EMOTION_VECTORIZER_PATH
        ):

            emotion_vectorizer = joblib.load(
                EMOTION_VECTORIZER_PATH
            )

    except Exception:
        pass

    return (
        sentiment_model,
        vectorizer,
        topic_model,
        topic_vectorizer,
        emotion_model,
        emotion_vectorizer,
    )


# ============================================================
# AI ANALYSIS
# ============================================================

def run_ai_analysis(df):

    df = df.copy()

    (
        sentiment_model,
        vectorizer,
        topic_model,
        topic_vectorizer,
        emotion_model,
        emotion_vectorizer,
    ) = load_models()

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    if (
        sentiment_model is not None
        and vectorizer is not None
    ):

        try:

            X = vectorizer.transform(
                texts
            )

            predictions = (
                sentiment_model
                .predict(X)
            )

            df["sentiment"] = [
                SENTIMENT_LABELS.get(
                    value,
                    str(value).title()
                )
                for value in predictions
            ]

            if hasattr(
                sentiment_model,
                "predict_proba"
            ):

                probabilities = (
                    sentiment_model
                    .predict_proba(X)
                )

                df["sentiment_confidence"] = (
                    probabilities.max(axis=1)
                )

            else:

                df["sentiment_confidence"] = 0.0

        except Exception:

            df["sentiment"] = "Neutral"
            df["sentiment_confidence"] = 0.0

    else:

        df["sentiment"] = "Neutral"
        df["sentiment_confidence"] = 0.0

    # --------------------------------------------------------
    # EMOTION
    # --------------------------------------------------------

    if (
        emotion_model is not None
        and emotion_vectorizer is not None
    ):

        try:

            X_emotion = (
                emotion_vectorizer
                .transform(texts)
            )

            predictions = (
                emotion_model
                .predict(X_emotion)
            )

            df["emotion"] = [
                EMOTION_LABELS.get(
                    value,
                    str(value).title()
                )
                for value in predictions
            ]

            if hasattr(
                emotion_model,
                "predict_proba"
            ):

                probabilities = (
                    emotion_model
                    .predict_proba(
                        X_emotion
                    )
                )

                df["emotion_confidence"] = (
                    probabilities.max(axis=1)
                )

            else:

                df["emotion_confidence"] = 0.0

        except Exception:

            df["emotion"] = "Joy"
            df["emotion_confidence"] = 0.0

    else:

        df["emotion"] = "Joy"
        df["emotion_confidence"] = 0.0

    # --------------------------------------------------------
    # TOPICS
    # --------------------------------------------------------

    if (
        topic_model is not None
        and topic_vectorizer is not None
    ):

        try:

            X_topic = (
                topic_vectorizer
                .transform(texts)
            )

            predictions = (
                topic_model
                .predict(X_topic)
            )

            df["topic_id"] = predictions

            df["topic"] = [

                TOPIC_NAMES.get(
                    int(value),
                    f"Topic {value}"
                )

                for value in predictions
            ]

            if hasattr(
                topic_model,
                "predict_proba"
            ):

                probabilities = (
                    topic_model
                    .predict_proba(
                        X_topic
                    )
                )

                df["topic_confidence"] = (
                    probabilities.max(axis=1)
                )

            else:

                df["topic_confidence"] = 0.0

        except Exception:

            df["topic_id"] = 0
            df["topic"] = "General"
            df["topic_confidence"] = 0.0

    else:

        df["topic_id"] = 0
        df["topic"] = "General"
        df["topic_confidence"] = 0.0

    # --------------------------------------------------------
    # TREND SCORE
    # --------------------------------------------------------

    engagement_score = np.log1p(
        df["engagement"]
    )

    views_score = np.log1p(
        df["views"]
    )

    sentiment_bonus = (
        df["sentiment"]
        .map({
            "Positive": 1.0,
            "Neutral": 0.5,
            "Negative": 0.2,
        })
        .fillna(0.5)
    )

    raw_score = (
        engagement_score * 0.55
        + views_score * 0.35
        + sentiment_bonus * 0.10
    )

    if (
        raw_score.max()
        > raw_score.min()
    ):

        df["trend_score_normalized"] = (
            (
                raw_score
                - raw_score.min()
            )
            /
            (
                raw_score.max()
                - raw_score.min()
            )
            * 100
        )

    else:

        df["trend_score_normalized"] = 0

    df["trend_score"] = (
        df["trend_score_normalized"]
        .round(2)
    )

    return df


# ============================================================
# YOUTUBE API
# ============================================================

YOUTUBE_BASE_URL = (
    "https://www.googleapis.com/youtube/v3"
)


def get_youtube_api_key():

    try:

        return st.secrets[
            "YOUTUBE_API_KEY"
        ]

    except Exception:

        return ""


def youtube_request(
    endpoint,
    params
):

    url = (
        f"{YOUTUBE_BASE_URL}/{endpoint}"
        f"?{urllib.parse.urlencode(params)}"
    )

    try:

        with urllib.request.urlopen(
            url,
            timeout=30
        ) as response:

            return json.loads(
                response.read()
                .decode("utf-8")
            )

    except urllib.error.HTTPError as error:

        try:

            body = (
                error.read()
                .decode("utf-8")
            )

            error_json = json.loads(
                body
            )

            message = (
                error_json
                .get("error", {})
                .get(
                    "message",
                    "YouTube API request failed."
                )
            )

            reason = (
                error_json
                .get("error", {})
                .get(
                    "errors",
                    [{}]
                )[0]
                .get(
                    "reason",
                    ""
                )
            )

            if reason:

                raise RuntimeError(
                    f"{message} ({reason})"
                )

            raise RuntimeError(
                message
            )

        except json.JSONDecodeError:

            raise RuntimeError(
                f"YouTube API error "
                f"HTTP {error.code}"
            )

    except urllib.error.URLError as error:

        raise RuntimeError(
            "Could not connect to YouTube API: "
            + str(error.reason)
        )


def collect_youtube_data(
    api_key,
    query,
    region_code="IN",
    max_videos=10,
    comments_per_video=10,
):

    if not api_key:

        raise RuntimeError(
            "YouTube API key is missing."
        )

    query = str(query).strip()

    if not query:

        raise RuntimeError(
            "Please enter a search topic."
        )

    max_videos = max(
        1,
        min(
            int(max_videos),
            20
        )
    )

    comments_per_video = max(
        0,
        min(
            int(comments_per_video),
            20
        )
    )

    # --------------------------------------------------------
    # SEARCH VIDEOS
    # --------------------------------------------------------

    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_videos,
        "order": "relevance",
        "key": api_key,
    }

    if region_code != "GLOBAL":

        search_params[
            "regionCode"
        ] = region_code

    search_data = youtube_request(
        "search",
        search_params
    )

    search_items = search_data.get(
        "items",
        []
    )

    if not search_items:

        raise RuntimeError(
            f"No YouTube videos found for "
            f"'{query}'."
        )

    video_ids = []

    for item in search_items:

        video_id = (
            item
            .get("id", {})
            .get("videoId")
        )

        if video_id:

            video_ids.append(
                video_id
            )

    if not video_ids:

        raise RuntimeError(
            "No valid YouTube video IDs returned."
        )

    # --------------------------------------------------------
    # VIDEO DETAILS
    # --------------------------------------------------------

    video_data = youtube_request(
        "videos",
        {
            "part": (
                "snippet,statistics"
            ),
            "id": ",".join(video_ids),
            "key": api_key,
        },
    )

    rows = []

    for video in video_data.get(
        "items",
        []
    ):

        video_id = video.get(
            "id",
            ""
        )

        snippet = video.get(
            "snippet",
            {}
        )

        statistics = video.get(
            "statistics",
            {}
        )

        title = snippet.get(
            "title",
            ""
        )

        description = snippet.get(
            "description",
            ""
        )

        text = (
            f"{title} {description}"
        ).strip()

        rows.append({

            "post_id": video_id,

            "platform": "YouTube",

            "post_type": "video",

            "text": text,

            "date": snippet.get(
                "publishedAt",
                ""
            ),

            "likes": parse_metric(
                statistics.get(
                    "likeCount",
                    0
                )
            ),

            "comments": parse_metric(
                statistics.get(
                    "commentCount",
                    0
                )
            ),

            "shares": 0,

            "views": parse_metric(
                statistics.get(
                    "viewCount",
                    0
                )
            ),

            "hashtags": extract_hashtags(
                text
            ),

            "author": snippet.get(
                "channelTitle",
                ""
            ),

            "author_followers": 0,

            "language": snippet.get(
                "defaultLanguage",
                "Unknown"
            ),

            "source_url": (
                "https://www.youtube.com/watch?v="
                + video_id
            ),
        })

    # --------------------------------------------------------
    # COMMENTS
    # --------------------------------------------------------

    for video_id in video_ids:

        if comments_per_video <= 0:
            break

        try:

            comment_data = youtube_request(
                "commentThreads",
                {
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": comments_per_video,
                    "order": "relevance",
                    "textFormat": "plainText",
                    "key": api_key,
                },
            )

        except RuntimeError as error:

            error_text = (
                str(error)
                .lower()
            )

            if (
                "disabled" in error_text
                or "forbidden" in error_text
                or "commentsdisabled"
                in error_text
            ):

                continue

            raise

        for item in comment_data.get(
            "items",
            []
        ):

            comment = (
                item
                .get(
                    "snippet",
                    {}
                )
                .get(
                    "topLevelComment",
                    {}
                )
            )

            snippet = comment.get(
                "snippet",
                {}
            )

            comment_id = comment.get(
                "id",
                ""
            )

            comment_text = snippet.get(
                "textDisplay",
                ""
            )

            if not comment_text:
                continue

            rows.append({

                "post_id": comment_id,

                "platform": "YouTube",

                "post_type": "comment",

                "text": comment_text,

                "date": snippet.get(
                    "publishedAt",
                    ""
                ),

                "likes": parse_metric(
                    snippet.get(
                        "likeCount",
                        0
                    )
                ),

                "comments": 0,

                "shares": 0,

                "views": 0,

                "hashtags": extract_hashtags(
                    comment_text
                ),

                "author": snippet.get(
                    "authorDisplayName",
                    ""
                ),

                "author_followers": 0,

                "language": "Unknown",

                "source_url": (
                    "https://www.youtube.com/watch?v="
                    + video_id
                ),
            })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise RuntimeError(
            "YouTube returned no usable records."
        )

    return result


# ============================================================
# EXCEL REPORT
# ============================================================

def create_excel_report(
    report_df
):

    output = io.BytesIO()

    if report_df is None:

        report_df = pd.DataFrame()

    report_df = report_df.copy()

    if (
        report_df.empty
        and len(report_df.columns) == 0
    ):

        report_df = pd.DataFrame({
            "Message": [
                "No report data available."
            ]
        })

    # --------------------------------------------------------
    # CONVERT DATETIME VALUES
    # --------------------------------------------------------

    for column in report_df.columns:

        def excel_safe_value(value):

            try:

                if isinstance(
                    value,
                    pd.Timestamp
                ):

                    return value.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                if hasattr(
                    value,
                    "tzinfo"
                ):

                    if (
                        value.tzinfo
                        is not None
                    ):

                        return (
                            value
                            .replace(
                                tzinfo=None
                            )
                            .strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        )

                return value

            except Exception:

                return str(value)

        report_df[column] = (
            report_df[column]
            .map(excel_safe_value)
        )

    # --------------------------------------------------------
    # FINAL DATE SAFETY
    # --------------------------------------------------------

    for column in report_df.columns:

        column_name = str(
            column
        ).lower()

        if any(
            word in column_name
            for word in [
                "date",
                "time",
                "timestamp",
                "created",
                "published"
            ]
        ):

            report_df[column] = (
                report_df[column]
                .astype(str)
            )

    # --------------------------------------------------------
    # CREATE EXCEL
    # --------------------------------------------------------

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


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(df):

    try:

        from reportlab.lib.pagesizes import A4

        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        from reportlab.lib import colors

        from reportlab.lib.styles import (
            getSampleStyleSheet
        )

    except Exception:

        return None

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

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
            f"Records analyzed: {len(df):,}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    if "sentiment" in df.columns:

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
        )

        table_data = [
            [
                "Sentiment",
                "Records"
            ]
        ]

        for label in SENTIMENT_ORDER:

            table_data.append([
                label,
                str(
                    sentiment_counts.get(
                        label,
                        0
                    )
                )
            ])

        table = Table(
            table_data
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(
            table
        )

    document.build(
        story
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if (
    "youtube_live_df"
    not in st.session_state
):

    st.session_state[
        "youtube_live_df"
    ] = None


if (
    "youtube_query"
    not in st.session_state
):

    st.session_state[
        "youtube_query"
    ] = ""


# ============================================================
# PREMIUM LIGHT UI
# ============================================================

render_html("""
<style>

.stApp {

    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(139,92,246,.08),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 10%,
            rgba(59,130,246,.07),
            transparent 28%
        ),

        #f8f9fc;
}

section[data-testid="stSidebar"] {

    background: #ffffff;

    border-right:
        1px solid #ececf3;
}

h1,
h2,
h3 {

    letter-spacing:
        -0.03em;
}

div[data-testid="stMetric"] {

    background: white;

    border:
        1px solid #ececf3;

    border-radius:
        16px;

    padding:
        14px;
}

.stButton > button {

    border-radius:
        12px;

    font-weight:
        700;
}

.stDownloadButton > button {

    border-radius:
        12px;

    font-weight:
        700;
}

</style>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 📊 TrendSense AI"
    )

    st.caption(
        "AI-powered social intelligence"
    )

    st.divider()

    # --------------------------------------------------------
    # DATA SOURCE
    # --------------------------------------------------------

    st.markdown(
        "### 📂 Data Source"
    )

    source = st.radio(
        "Choose data",
        [
            "Demo Dataset",
            "Upload CSV",
            "🔴 Live YouTube",
        ],
    )

    uploaded_file = None

    if source == "Upload CSV":

        uploaded_file = st.file_uploader(
            "Upload social media CSV",
            type=["csv"]
        )

    elif source == "🔴 Live YouTube":

        st.markdown(
            "#### 🔴 YouTube Collector"
        )

        youtube_query = st.text_input(
            "🔎 Search topic",
            value="artificial intelligence"
        )

        youtube_region = st.selectbox(
            "🌍 Region",
            [
                "India",
                "United States",
                "United Kingdom",
                "Global",
            ],
        )

        region_map = {

            "India": "IN",

            "United States": "US",

            "United Kingdom": "GB",

            "Global": "GLOBAL",
        }

        youtube_region_code = (
            region_map[
                youtube_region
            ]
        )

        youtube_video_count = st.slider(
            "📊 Number of videos",
            1,
            20,
            10,
        )

        youtube_comments_per_video = st.slider(
            "💬 Comments per video",
            0,
            20,
            10,
        )

        collect_button = st.button(
            "🚀 Collect YouTube Data",
            width="stretch",
        )

        if collect_button:

            api_key = (
                get_youtube_api_key()
            )

            if not api_key:

                st.error(
                    "YouTube API key not found. "
                    "Check .streamlit/secrets.toml."
                )

            else:

                try:

                    with st.spinner(
                        "🔴 Collecting current YouTube data..."
                    ):

                        live_df = (
                            collect_youtube_data(
                                api_key=api_key,
                                query=youtube_query,
                                region_code=(
                                    youtube_region_code
                                ),
                                max_videos=(
                                    youtube_video_count
                                ),
                                comments_per_video=(
                                    youtube_comments_per_video
                                ),
                            )
                        )

                    st.session_state[
                        "youtube_live_df"
                    ] = live_df

                    st.session_state[
                        "youtube_query"
                    ] = youtube_query

                    st.success(
                        f"Collected "
                        f"{len(live_df):,} records."
                    )

                except Exception as error:

                    st.error(
                        f"YouTube collection failed: "
                        f"{error}"
                    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Filters"
    )

    search_text = st.text_input(
        "Search posts",
        placeholder="Search keywords..."
    )

    sentiment_filter = st.multiselect(
        "Sentiment",
        SENTIMENT_ORDER,
        default=[]
    )

    emotion_filter = st.multiselect(
        "Emotion",
        EMOTION_ORDER,
        default=[]
    )

    minimum_likes = st.number_input(
        "Minimum likes",
        min_value=0,
        value=0,
        step=10,
    )


# ============================================================
# DATA LOADING
# ============================================================

if source == "🔴 Live YouTube":

    if (
        st.session_state[
            "youtube_live_df"
        ] is None
    ):

        st.info(
            "👆 Enter a YouTube topic and click "
            "'Collect YouTube Data' to start."
        )

        st.stop()

    raw_df = (
        st.session_state[
            "youtube_live_df"
        ].copy()
    )

    query_name = (
        st.session_state.get(
            "youtube_query",
            "YouTube"
        )
    )

    filename = "live_youtube.csv"

    data_source_label = (
        f"🔴 YouTube • {query_name}"
    )

elif (
    source == "Upload CSV"
    and uploaded_file is not None
):

    raw_df = pd.read_csv(
        uploaded_file
    )

    filename = uploaded_file.name

    data_source_label = filename

else:

    if not os.path.exists(
        DATA_PATH
    ):

        st.error(
            "Demo dataset not found."
        )

        st.stop()

    raw_df = pd.read_csv(
        DATA_PATH
    )

    filename = "social_media.csv"

    data_source_label = (
        "Demo Dataset"
    )


# ============================================================
# STANDARDIZE + AI
# ============================================================

df = standardize_social_data(
    raw_df
)

with st.spinner(
    "🧠 Running TrendSense AI analysis..."
):

    df = run_ai_analysis(
        df
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


if search_text:

    mask = (
        filtered_df["text"]
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    )

    filtered_df = (
        filtered_df[mask]
    )


if sentiment_filter:

    filtered_df = (
        filtered_df[
            filtered_df[
                "sentiment"
            ].isin(
                sentiment_filter
            )
        ]
    )


if emotion_filter:

    filtered_df = (
        filtered_df[
            filtered_df[
                "emotion"
            ].isin(
                emotion_filter
            )
        ]
    )


filtered_df = (
    filtered_df[
        filtered_df["likes"]
        >= minimum_likes
    ]
)


# ============================================================
# LANGUAGE SUMMARY
# ============================================================

language_counts = (
    filtered_df["language"]
    .value_counts()
    .reset_index()
)

language_counts.columns = [
    "Language",
    "Count"
]


# ============================================================
# HERO
# ============================================================

render_html(
    f"""
<div style="
    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #f5f3ff 45%,
            #eef6ff 100%
        );

    border:
        1px solid #e8e5f5;

    border-radius:
        26px;

    padding:
        34px;

    margin-bottom:
        25px;
">

    <div style="
        display:inline-block;
        padding:7px 12px;
        background:#ede9fe;
        color:#6d28d9;
        border-radius:999px;
        font-size:12px;
        font-weight:800;
        margin-bottom:14px;
    ">
        AI-POWERED SOCIAL INTELLIGENCE
    </div>

    <h1 style="
        font-size:44px;
        margin:0;
        color:#111827;
    ">
        TrendSense AI
    </h1>

    <p style="
        font-size:17px;
        color:#5b6474;
        margin-top:10px;
        max-width:800px;
    ">
        Transform social media data into actionable
        sentiment, emotion, topic and trend intelligence.
    </p>

    <div style="
        margin-top:16px;
        font-size:13px;
        color:#6b7280;
    ">

        📂 Source:
        <b>
            {html.escape(data_source_label)}
        </b>

        &nbsp;&nbsp;•&nbsp;&nbsp;

        🧠 NLP analyzed

        &nbsp;&nbsp;•&nbsp;&nbsp;

        📊 Business-ready insights

    </div>

</div>
"""
)


# ============================================================
# PIPELINE
# ============================================================
# ============================================================
# TRENDsense INTELLIGENCE PIPELINE
# ============================================================
# ============================================================
# TRENDSENSE INTELLIGENCE PIPELINE
# ============================================================
# ============================================================
# TRENDSENSE INTELLIGENCE PIPELINE
# ============================================================

st.subheader("🔄 TrendSense Intelligence Pipeline")

pipeline = [
    ("📥", "Data"),
    ("🔍", "Platform"),
    ("💭", "Sentiment"),
    ("❤️", "Emotion"),
    ("🧠", "Topics"),
    ("🔥", "Trends"),
    ("💼", "Business Insights"),
]

cols = st.columns(7)

for col, (icon, label) in zip(cols, pipeline):
    with col:
        st.markdown(f"### {icon}")
        st.caption(label)

# ============================================================
# DATA STATUS
# ============================================================

st.markdown(
    f"**Dataset:** `{filename}`"
)

st.caption(
    f"{len(filtered_df):,} records shown "
    f"out of {len(df):,} total records."
)


# ============================================================
# KPI
# ============================================================

total_records = len(
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

total_views = (
    filtered_df["views"]
    .sum()
)

avg_engagement = (
    filtered_df["engagement"]
    .mean()
    if len(filtered_df)
    else 0
)

positive_percentage = (

    (
        filtered_df["sentiment"]
        .eq("Positive")
        .mean()
        * 100
    )

    if len(filtered_df)

    else 0
)


kpis = st.columns(5)


with kpis[0]:

    render_html(
        kpi_card(
            "Records",
            compact_number(
                total_records
            ),
            "Analyzed posts/comments"
        )
    )


with kpis[1]:

    render_html(
        kpi_card(
            "Likes",
            compact_number(
                total_likes
            ),
            "Total likes"
        )
    )


with kpis[2]:

    render_html(
        kpi_card(
            "Comments",
            compact_number(
                total_comments
            ),
            "Total comments"
        )
    )


with kpis[3]:

    render_html(
        kpi_card(
            "Views",
            compact_number(
                total_views
            ),
            "Total views"
        )
    )


with kpis[4]:

    render_html(
        kpi_card(
            "Positive",
            f"{positive_percentage:.1f}%",
            "Positive sentiment"
        )
    )


st.markdown("")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([

    "📊 Overview",

    "💭 Sentiment & Emotion",

    "🔥 Topics & Trends",

    "🌐 Platform Comparison",

    "💼 Business Insights",

    "🧪 Model Performance",

    "🔎 Post Explorer",

    "📄 Reports",

])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader(
        "Social Intelligence Overview"
    )

    # --------------------------------------------------------
    # SENTIMENT + EMOTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

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
            title="Sentiment Distribution",
            text="Count",
        )

        fig.update_layout(
            template="plotly_white",
            height=380,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    with col2:

        emotion_counts = (
            filtered_df["emotion"]
            .value_counts()
            .reindex(
                EMOTION_ORDER,
                fill_value=0
            )
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
            title="Emotion Distribution",
            text="Count",
        )

        fig.update_layout(
            template="plotly_white",
            height=380,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # --------------------------------------------------------
    # ACTIVITY OVER TIME
    # --------------------------------------------------------

    if (
        "date" in filtered_df.columns
        and filtered_df[
            "date"
        ].notna().any()
    ):

        daily = (
            filtered_df
            .dropna(
                subset=["date"]
            )
            .assign(
                day=lambda x:
                x["date"].dt.date
            )
            .groupby("day")
            .size()
            .reset_index(
                name="Posts"
            )
        )

        fig = px.line(
            daily,
            x="day",
            y="Posts",
            markers=True,
            title="Activity Over Time",
        )

        fig.update_layout(
            template="plotly_white",
            height=360,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # --------------------------------------------------------
    # LANGUAGE DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        "### 🌐 Language Distribution"
    )

    if not language_counts.empty:

        fig = px.bar(
            language_counts,
            x="Language",
            y="Count",
            text="Count",
            title="Detected Languages"
        )

        fig.update_layout(
            template="plotly_white",
            height=380
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "No language data available."
        )


# ============================================================
# TAB 2 — SENTIMENT & EMOTION
# ============================================================

with tabs[1]:

    st.subheader(
        "Sentiment & Emotion Intelligence"
    )

    c1, c2 = st.columns(2)


    with c1:

        sentiment = (
            filtered_df["sentiment"]
            .value_counts()
            .reindex(
                SENTIMENT_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        sentiment.columns = [
            "Sentiment",
            "Count"
        ]

        fig = px.pie(
            sentiment,
            names="Sentiment",
            values="Count",
            hole=0.45,
            title="Sentiment Mix",
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    with c2:

        emotion = (
            filtered_df["emotion"]
            .value_counts()
            .reindex(
                EMOTION_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        emotion.columns = [
            "Emotion",
            "Count"
        ]

        fig = px.pie(
            emotion,
            names="Emotion",
            values="Count",
            hole=0.45,
            title="Emotion Mix",
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    st.markdown(
        "### Sentiment by Platform"
    )

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
            name="Count"
        )
    )

    if not platform_sentiment.empty:

        fig = px.bar(
            platform_sentiment,
            x="platform",
            y="Count",
            color="sentiment",
            barmode="group",
            title="Sentiment Across Platforms",
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# TAB 3 — TOPICS & TRENDS
# ============================================================

with tabs[2]:

    st.subheader(
        "Topics & Trending Content"
    )

    topic_counts = (
        filtered_df["topic"]
        .value_counts()
        .reset_index()
    )

    topic_counts.columns = [
        "Topic",
        "Count"
    ]

    fig = px.bar(
        topic_counts,
        x="Count",
        y="Topic",
        orientation="h",
        text="Count",
        title="Top Topics",
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.markdown(
        "### 🔥 Trending Posts"
    )

    trending = (
        filtered_df
        .sort_values(
            "trend_score",
            ascending=False
        )
        .head(10)
        .copy()
    )

    display_cols = [

        "platform",

        "post_type",

        "text",

        "sentiment",

        "emotion",

        "topic",

        "trend_score",

        "likes",

        "comments",

        "views",
    ]

    display_cols = [
        col
        for col in display_cols
        if col in trending.columns
    ]

    st.dataframe(
        trending[
            display_cols
        ],
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # HASHTAGS
    # --------------------------------------------------------

    st.markdown(
        "### #️⃣ Trending Hashtags"
    )

    all_hashtags = []

    for value in filtered_df[
        "hashtags"
    ].dropna():

        tags = re.findall(
            r"#?[A-Za-z0-9_]+",
            str(value)
        )

        for tag in tags:

            tag = tag.strip()

            if not tag:
                continue

            if not tag.startswith("#"):

                tag = "#" + tag

            all_hashtags.append(
                tag.lower()
            )

    if all_hashtags:

        hashtag_counts = (
            pd.Series(
                all_hashtags
            )
            .value_counts()
            .head(20)
            .reset_index()
        )

        hashtag_counts.columns = [
            "Hashtag",
            "Count"
        ]

        fig = px.bar(
            hashtag_counts,
            x="Count",
            y="Hashtag",
            orientation="h",
            title="Most Frequent Hashtags",
        )

        fig.update_layout(
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "No hashtags available in this dataset."
        )


# ============================================================
# TAB 4 — PLATFORM COMPARISON
# ============================================================

with tabs[3]:

    st.subheader(
        "🌐 Platform Comparison"
    )

    platform_summary = (
        filtered_df
        .groupby("platform")
        .agg(
            Posts=(
                "post_id",
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

            Views=(
                "views",
                "sum"
            ),

            Avg_Engagement=(
                "engagement",
                "mean"
            ),
        )
        .reset_index()
    )

    st.dataframe(
        platform_summary,
        width="stretch",
        hide_index=True,
    )


    if not platform_summary.empty:

        fig = px.bar(
            platform_summary,
            x="platform",
            y="Likes",
            title="Likes by Platform",
            text="Likes",
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


        fig = px.bar(
            platform_summary,
            x="platform",
            y="Avg_Engagement",
            title="Average Engagement by Platform",
            text="Avg_Engagement",
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# TAB 5 — BUSINESS INSIGHTS
# ============================================================

with tabs[4]:

    st.subheader(
        "💼 Business Intelligence"
    )

    if len(filtered_df):

        top_post = (
            filtered_df
            .sort_values(
                "trend_score",
                ascending=False
            )
            .iloc[0]
        )

        dominant_sentiment = (
            filtered_df[
                "sentiment"
            ]
            .value_counts()
            .idxmax()
        )

        dominant_emotion = (
            filtered_df[
                "emotion"
            ]
            .value_counts()
            .idxmax()
        )

        dominant_topic = (
            filtered_df[
                "topic"
            ]
            .value_counts()
            .idxmax()
        )

        insights = [

            (
                "🔥 Top Trend",
                f"The strongest trending content "
                f"belongs to: {dominant_topic}."
            ),

            (
                "💭 Audience Sentiment",
                f"The dominant audience sentiment "
                f"is {dominant_sentiment}."
            ),

            (
                "❤️ Audience Emotion",
                f"The dominant detected emotion "
                f"is {dominant_emotion}."
            ),

            (
                "📈 Best Content",
                f"The highest trend-score content "
                f"has a score of "
                f"{top_post['trend_score']:.1f}."
            ),
        ]


        for title, message in insights:

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border:1px solid #ececf3;
                    border-radius:16px;
                    padding:18px;
                    margin-bottom:12px;
                ">

                    <b>{title}</b>

                    <div style="
                        margin-top:7px;
                        color:#596273;
                    ">

                        {html.escape(message)}

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        st.markdown(
            "### 🎯 Recommended Business Actions"
        )

        actions = []

        if dominant_sentiment == "Positive":

            actions.append(
                "Use positive audience reactions "
                "in marketing campaigns."
            )

        elif dominant_sentiment == "Negative":

            actions.append(
                "Investigate negative themes and "
                "prioritize customer feedback."
            )

        else:

            actions.append(
                "Create content designed to move "
                "neutral audiences toward engagement."
            )


        actions.extend([

            (
                f"Prioritize the topic "
                f"'{dominant_topic}' "
                "for future content."
            ),

            (
                "Monitor rapidly increasing "
                "engagement to identify emerging trends."
            ),

            (
                "Compare platforms before allocating "
                "marketing resources."
            ),
        ])


        for action in actions:

            st.markdown(
                f"- {action}"
            )

    else:

        st.info(
            "No records match the current filters."
        )


# ============================================================
# TAB 6 — MODEL PERFORMANCE
# ============================================================

with tabs[5]:

    st.subheader(
        "🧪 Model Performance"
    )

    st.caption(
        "Confidence scores below indicate the model's "
        "prediction confidence for the analyzed records."
    )

    c1, c2, c3 = st.columns(3)


    with c1:

        avg_sentiment_confidence = (

            filtered_df[
                "sentiment_confidence"
            ].mean()

            if len(filtered_df)

            else 0
        )

        st.metric(
            "Sentiment Confidence",
            f"{avg_sentiment_confidence * 100:.1f}%"
        )


    with c2:

        avg_emotion_confidence = (

            filtered_df[
                "emotion_confidence"
            ].mean()

            if len(filtered_df)

            else 0
        )

        st.metric(
            "Emotion Confidence",
            f"{avg_emotion_confidence * 100:.1f}%"
        )


    with c3:

        avg_topic_confidence = (

            filtered_df[
                "topic_confidence"
            ].mean()

            if len(filtered_df)

            else 0
        )

        st.metric(
            "Topic Confidence",
            f"{avg_topic_confidence * 100:.1f}%"
        )


    confidence_df = pd.DataFrame({

        "Model": [
            "Sentiment",
            "Emotion",
            "Topic",
        ],

        "Confidence": [

            avg_sentiment_confidence * 100,

            avg_emotion_confidence * 100,

            avg_topic_confidence * 100,
        ],
    })


    fig = px.bar(
        confidence_df,
        x="Model",
        y="Confidence",
        text="Confidence",
        title="Average Model Confidence",
    )

    fig.update_yaxes(
        range=[0, 100]
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# TAB 7 — POST EXPLORER
# ============================================================

with tabs[6]:

    st.subheader(
        "🔎 Post Explorer"
    )

    explorer_cols = [

        "post_id",

        "platform",

        "post_type",

        "text",

        "sentiment",

        "emotion",

        "topic",

        "trend_score",

        "likes",

        "comments",

        "shares",

        "views",

        "author",

        "source_url",
    ]

    explorer_cols = [

        col

        for col in explorer_cols

        if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[
            explorer_cols
        ],
        width="stretch",
        hide_index=True,
    )


# ============================================================
# TAB 8 — REPORTS
# ============================================================

with tabs[7]:

    st.subheader(
        "📄 Reports & Export"
    )

    report_cols = [

        "post_id",

        "platform",

        "post_type",

        "text",

        "date",

        "language",

        "likes",

        "comments",

        "shares",

        "views",

        "sentiment",

        "sentiment_confidence",

        "emotion",

        "emotion_confidence",

        "topic",

        "topic_confidence",

        "trend_score",

        "source_url",
    ]

    report_cols = [

        col

        for col in report_cols

        if col in filtered_df.columns
    ]

    report_df = (
        filtered_df[
            report_cols
        ].copy()
    )

    c1, c2, c3 = st.columns(3)


    with c1:

        csv_data = (
            report_df
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            file_name="trendsense_report.csv",
            mime="text/csv",
            width="stretch",
        )


    with c2:

        excel_data = (
            create_excel_report(
                report_df
            )
        )

        st.download_button(
            "📊 Download Excel",
            excel_data,
            file_name="trendsense_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            width="stretch",
        )


    with c3:

        pdf_data = (
            create_pdf_report(
                report_df
            )
        )

        if pdf_data:

            st.download_button(
                "📄 Download PDF",
                pdf_data,
                file_name="trendsense_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

        else:

            st.warning(
                "PDF generation unavailable."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        color:#8b8fa3;
        font-size:12px;
    ">

        <b>TrendSense AI</b> •

        AI-powered social intelligence platform

        <br>

        CSV + Live YouTube • NLP • Sentiment •
        Emotion • Topics • Trends • Business Insights

    </div>
    """,
    unsafe_allow_html=True,
)

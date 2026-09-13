import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")


lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Cleans social media text for NLP processing.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Split into words
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def extract_hashtags(text):
    """
    Extract hashtags from a social media post.
    """

    hashtags = re.findall(r"#\w+", str(text))

    return [tag.lower() for tag in hashtags]

import re
from concurrent.futures import ThreadPoolExecutor

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0  # makes detection consistent
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False


# Unicode ranges for Indian-language scripts (+ Urdu)
SCRIPT_RANGES = {
    "Hindi": (0x0900, 0x097F),       # Devanagari (Hindi/Marathi/Nepali)
    "Bengali": (0x0980, 0x09FF),
    "Punjabi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Urdu": (0x0600, 0x06FF),
}

LANG_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "ne": "Nepali", "bn": "Bengali",
    "pa": "Punjabi", "gu": "Gujarati", "or": "Odia", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "ar": "Arabic", "id": "Indonesian", "it": "Italian", "ru": "Russian",
}


def detect_language_label(text):
    """Returns 'English' or the name of the detected language."""
    text = str(text or "")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "English"

    # 1) Script-based detection (fast and reliable for Indic scripts)
    for name, (lo, hi) in SCRIPT_RANGES.items():
        count = sum(1 for c in letters if lo <= ord(c) <= hi)
        if count / len(letters) >= 0.3:
            if name == "Hindi" and HAS_LANGDETECT:
                try:
                    code = detect_langs(text)[0].lang
                    if code in ("mr", "ne"):
                        return LANG_NAMES[code]
                except Exception:
                    pass
            return name

    # 2) Latin-script text in other languages (Spanish, French, etc.)
    if HAS_LANGDETECT and len(text.strip()) >= 25:
        try:
            best = detect_langs(text)[0]
            if best.lang != "en" and best.prob >= 0.90:
                return LANG_NAMES.get(best.lang, best.lang)
        except Exception:
            pass

    return "English"


def _translate_one(text):
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(
            str(text)[:4500]
        )
        return translated if translated else text
    except Exception:
        return text  # if translation fails, keep original


def translate_to_english(texts):
    """Translate a list of strings to English (parallel)."""
    if not HAS_TRANSLATOR or not texts:
        return list(texts)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_translate_one, texts))
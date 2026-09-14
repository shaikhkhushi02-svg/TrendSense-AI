import json
import urllib.parse
import urllib.request
import urllib.error
import pandas as pd


YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3"


def youtube_request(endpoint, params):
    """
    Make a GET request to the YouTube Data API.
    """
    url = f"{YOUTUBE_BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        try:
            error_body = error.read().decode("utf-8")
            error_json = json.loads(error_body)

            message = (
                error_json
                .get("error", {})
                .get("message", "YouTube API request failed.")
            )

            reason = (
                error_json
                .get("error", {})
                .get("errors", [{}])[0]
                .get("reason", "")
            )

            if reason:
                raise RuntimeError(
                    f"{message} ({reason})"
                )

            raise RuntimeError(message)

        except json.JSONDecodeError:
            raise RuntimeError(
                f"YouTube API request failed with HTTP {error.code}."
            )

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to YouTube API: {error.reason}"
        )


def collect_youtube_data(
    api_key,
    query,
    region_code="IN",
    max_videos=10,
    comments_per_video=10
):
    """
    Collect current public YouTube video data and
    optional top-level comments.

    Returns:
        pandas.DataFrame
    """

    if not api_key:
        raise RuntimeError(
            "YouTube API key is missing."
        )

    query = str(query).strip()

    if not query:
        raise RuntimeError(
            "Please enter a YouTube search topic."
        )

    max_videos = max(
        1,
        min(int(max_videos), 50)
    )

    comments_per_video = max(
        0,
        min(int(comments_per_video), 50)
    )

    # =========================================================
    # 1. SEARCH YOUTUBE VIDEOS
    # =========================================================

    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_videos,
        "order": "relevance",
        "key": api_key,
    }

    if region_code and region_code != "GLOBAL":
        search_params["regionCode"] = region_code

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
            f"No YouTube videos found for '{query}'."
        )

    video_ids = [
        item.get("id", {}).get("videoId")
        for item in search_items
        if item.get("id", {}).get("videoId")
    ]

    if not video_ids:
        raise RuntimeError(
            "YouTube returned no valid video IDs."
        )

    # =========================================================
    # 2. GET VIDEO DETAILS + STATISTICS
    # =========================================================

    video_params = {
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": api_key,
    }

    video_data = youtube_request(
        "videos",
        video_params
    )

    video_items = video_data.get(
        "items",
        []
    )

    rows = []

    # =========================================================
    # 3. CREATE VIDEO ROWS
    # =========================================================

    for video in video_items:

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

        published_at = snippet.get(
            "publishedAt",
            ""
        )

        channel_title = snippet.get(
            "channelTitle",
            ""
        )

        likes = statistics.get(
            "likeCount",
            0
        )

        comments = statistics.get(
            "commentCount",
            0
        )

        views = statistics.get(
            "viewCount",
            0
        )

        source_url = (
            f"https://www.youtube.com/watch?v={video_id}"
        )

        rows.append({
            "post_id": video_id,
            "platform": "YouTube",
            "post_type": "video",
            "text": text,
            "date": published_at,
            "likes": likes,
            "comments": comments,
            "shares": 0,
            "views": views,
            "hashtags": "",
            "author": channel_title,
            "author_followers": 0,
            "language": snippet.get(
                "defaultLanguage",
                "Unknown"
            ),
            "source_url": source_url,
        })

    # =========================================================
    # 4. COLLECT COMMENTS
    # =========================================================

    if comments_per_video > 0:

        for video_id in video_ids:

            try:

                comment_params = {
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": comments_per_video,
                    "order": "relevance",
                    "textFormat": "plainText",
                    "key": api_key,
                }

                comment_data = youtube_request(
                    "commentThreads",
                    comment_params
                )

                comment_items = comment_data.get(
                    "items",
                    []
                )

                video_url = (
                    f"https://www.youtube.com/watch?v={video_id}"
                )

                for item in comment_items:

                    snippet = (
                        item
                        .get("snippet", {})
                        .get("topLevelComment", {})
                        .get("snippet", {})
                    )

                    comment_id = (
                        item
                        .get("snippet", {})
                        .get("topLevelComment", {})
                        .get("id", "")
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
                        "likes": snippet.get(
                            "likeCount",
                            0
                        ),
                        "comments": 0,
                        "shares": 0,
                        "views": 0,
                        "hashtags": "",
                        "author": snippet.get(
                            "authorDisplayName",
                            ""
                        ),
                        "author_followers": 0,
                        "language": "Unknown",
                        "source_url": video_url,
                    })

            except RuntimeError as error:

                # Some videos have comments disabled.
                # Continue collecting the other videos.
                error_text = str(error).lower()

                if (
                    "disabled" in error_text
                    or "commentsdisabled" in error_text
                    or "forbidden" in error_text
                ):
                    continue

                raise

    # =========================================================
    # 5. FINAL DATAFRAME
    # =========================================================

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "YouTube returned no usable data."
        )

    return df
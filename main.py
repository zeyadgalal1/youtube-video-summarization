from transformers import AutoTokenizer, pipeline
from urllib.parse import urlparse, parse_qs
import requests

MODEL_NAME = "facebook/bart-large-cnn"
MAX_CHUNK_TOKENS = 900


def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=1024
    )

    summarizer = pipeline(
        "summarization",
        model=MODEL_NAME,
        tokenizer=tokenizer
    )

    return tokenizer, summarizer


tokenizer, summarizer = load_summarizer()


def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from a URL.
    """

    parsed = urlparse(url)

    host = (parsed.hostname or "").lower()

    if host.startswith("www."):
        host = host[4:]

    # youtu.be/VIDEO_ID
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]

        if video_id:
            return video_id

    # youtube.com/watch?v=VIDEO_ID
    if host in (
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ):

        qs = parse_qs(parsed.query)

        if qs.get("v"):
            return qs["v"][0]

        # youtube.com/shorts/VIDEO_ID
        # youtube.com/embed/VIDEO_ID
        # youtube.com/live/VIDEO_ID
        parts = parsed.path.strip("/").split("/")

        if len(parts) >= 2 and parts[0] in (
            "shorts",
            "embed",
            "live",
            "v"
        ):
            return parts[1]

    raise ValueError(f"No video ID found in URL: {url}")


def get_transcript(url: str) -> str:
    """
    Retrieve the transcript from a YouTube video
    using a hosted transcript service.
    """

    response = requests.get(
        "https://api.freetranscriptapi.com/v1/transcript",
        params={
            "video_url": url,
            "lang": "en"
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    transcript = data.get("transcript", [])

    if not transcript:
        raise ValueError(
            "No transcript was found for this video."
        )

    return "\n".join(
        item["text"]
        for item in transcript
        if item.get("text")
    )

def chunk_text_by_tokens(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:

    """
    Split text into chunks that fit within
    the model's token limit.
    """

    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    for start in range(
        0,
        len(input_ids),
        max_tokens
    ):

        chunk_ids = input_ids[
            start:start + max_tokens
        ]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        chunks.append(chunk_text)

    return chunks


def summarize_long_text(
    text: str,
    final_compression: bool = True
) -> str:

    """
    Summarize a transcript of any length.
    """

    chunks = chunk_text_by_tokens(text)

    chunk_summaries = []

    for chunk in chunks:

        result = summarizer(
            chunk,
            max_length=150,
            min_length=30
        )

        summary_text = result[0]["summary_text"]

        chunk_summaries.append(summary_text)

    combined = " ".join(chunk_summaries)

    # Compress all chunk summaries into one final summary
    if final_compression and len(chunks) > 1:

        final = summarizer(
            combined,
            max_length=150,
            min_length=30
        )

        return final[0]["summary_text"]

    return combined


def summarize_youtube_video(url: str) -> str:
    """
    Complete pipeline:
    YouTube URL → Transcript → Summary
    """

    text = get_transcript(url)

    if not text.strip():
        raise ValueError(
            "No transcript was found for this video."
        )

    summary = summarize_long_text(
        text,
        final_compression=True
    )

    return summary
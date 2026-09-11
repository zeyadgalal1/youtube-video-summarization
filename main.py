from transformers import AutoTokenizer, pipeline
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME , model_max_length=1024)
summarizer = pipeline("summarization", model=MODEL_NAME, tokenizer=tokenizer)
print("Model loaded successfully.")

from transformers import pipeline
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

MAX_CHUNK_TOKENS = 900

def extract_video_id(url: str) -> str:

    """
    Extract the YouTube video ID from a URL.
    Raises ValueError if no 'v' parameter is found.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        qs = parse_qs(parsed.query)
        if qs.get("v"):
            return qs["v"][0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]

    raise ValueError(f"No video id found in URL: {url}")

def chunk_text_by_tokens(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """
    Split text into chunks that each fit within max_tokens,
    measured using the model's own tokenizer.
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for start in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[start:start + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


def summarize_long_text(text: str, final_compression: bool = True) -> str:
    """
    Summarize a full transcript of any length by chunking, summarizing
    each chunk, then optionally compressing the combined summaries into
    one final summary.
    """
    chunks = chunk_text_by_tokens(text)
    print(f"Split transcript into {len(chunks)} chunk(s).")
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        result = summarizer(
            chunk,
            max_length=150,
            min_length=30,
        )
        summary_text = result[0]["summary_text"]
        chunk_summaries.append(summary_text)
        print(f"Chunk {i+1}/{len(chunks)} summarized.")
        print(f"Summary {i+1}:\n{summary_text}\n")

    combined = " ".join(chunk_summaries)

    if final_compression and len(chunks) > 1:
        final = summarizer(
            combined,
            max_length=150,
            min_length=30,
        )
        return final[0]["summary_text"]
    return combined

url = "https://youtu.be/vLijZb9BpkE?si=z36cP0wJ8meNRI7B"
video_id = extract_video_id(url)
api = YouTubeTranscriptApi()
fetched = api.fetch(video_id, languages=["ar", "en"])
text = "\n".join(snippet.text for snippet in fetched)
print(text)
print(f"Transcript length: {len(text)} characters")

summary = summarize_long_text(text, final_compression=True)
print("\n--- Final Summary ---")
print(summary)
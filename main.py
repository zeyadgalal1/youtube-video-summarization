
import time
import requests

from transformers import AutoTokenizer, pipeline
from urllib.parse import urlparse, parse_qs


MODEL_NAME = "facebook/bart-large-cnn"
MAX_CHUNK_TOKENS = 900


# ==========================================
# 1. LOAD MODEL
# ==========================================

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


model_start_time = time.perf_counter()

tokenizer, summarizer = load_summarizer()

print(
    f"Model loading time: "
    f"{time.perf_counter() - model_start_time:.2f} seconds"
)


# ==========================================
# 2. EXTRACT YOUTUBE VIDEO ID
# ==========================================

def extract_video_id(url: str) -> str:

    parsed = urlparse(url)

    host = (parsed.hostname or "").lower()

    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":

        video_id = parsed.path.lstrip("/").split("/")[0]

        if video_id:
            return video_id

    if host in (
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ):

        qs = parse_qs(parsed.query)

        if qs.get("v"):
            return qs["v"][0]

        parts = parsed.path.strip("/").split("/")

        if (
            len(parts) >= 2
            and parts[0] in ("shorts", "embed", "live", "v")
        ):
            return parts[1]

    raise ValueError(f"No video ID found in URL: {url}")


# ==========================================
# 3. GET ENGLISH TRANSCRIPT
# ==========================================

def get_transcript(url: str) -> str:

    start_time = time.perf_counter()

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
        raise ValueError("No transcript was found for this video.")

    text = "\n".join(
        item["text"]
        for item in transcript
        if item.get("text")
    )

    print(
        f"Transcript retrieval time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    return text


# ==========================================
# 4. SPLIT TEXT INTO CHUNKS
# ==========================================

def chunk_text_by_tokens(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:

    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    for start in range(0, len(input_ids), max_tokens):

        chunk_ids = input_ids[start:start + max_tokens]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        chunks.append(chunk_text)

    return chunks


# ==========================================
# 5. SUMMARIZE LONG TEXT
# ==========================================

def summarize_long_text(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    final_compression: bool = True
) -> str:

    total_start_time = time.perf_counter()

    chunks = chunk_text_by_tokens(
        text,
        max_tokens=750
    )

    print(f"Number of chunks: {len(chunks)}")

    chunk_summaries = []

    for i, chunk in enumerate(chunks, start=1):

        chunk_start_time = time.perf_counter()

        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            num_beams=1,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )

        chunk_summaries.append(
            result[0]["summary_text"]
        )

        print(
            f"Chunk {i}/{len(chunks)} time: "
            f"{time.perf_counter() - chunk_start_time:.2f} seconds"
        )

    combined = " ".join(chunk_summaries)

    combined_tokens = tokenizer.encode(
        combined,
        add_special_tokens=False
    )

    # Second compression if necessary

    if len(combined_tokens) > 750:

        smaller_chunks = []

        for start in range(0, len(combined_tokens), 750):

            chunk_ids = combined_tokens[start:start + 750]

            smaller_chunk = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True
            )

            smaller_chunks.append(smaller_chunk)

        reduced_summaries = []

        for i, chunk in enumerate(smaller_chunks, start=1):

            reduction_start_time = time.perf_counter()

            result = summarizer(
                chunk,
                max_length=120,
                min_length=30,
                num_beams=4,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=False,
                truncation=True
            )

            reduced_summaries.append(
                result[0]["summary_text"]
            )

            print(
                f"Reduction {i}/{len(smaller_chunks)} time: "
                f"{time.perf_counter() - reduction_start_time:.2f} seconds"
            )

        combined = " ".join(reduced_summaries)

    # Final compression

    if final_compression:

        final_start_time = time.perf_counter()

        final = summarizer(
            combined,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )

        print(
            f"Final compression time: "
            f"{time.perf_counter() - final_start_time:.2f} seconds"
        )

        summary = final[0]["summary_text"]

    else:

        summary = combined

    print(
        f"Total summarization time: "
        f"{time.perf_counter() - total_start_time:.2f} seconds"
    )

    return summary


# ==========================================
# 6. SUMMARIZE YOUTUBE VIDEO
# ==========================================

def summarize_youtube_video(
    url: str,
    max_length: int = 150,
    min_length: int = 40
) -> str:

    start_time = time.perf_counter()

    text = get_transcript(url)

    if not text.strip():
        raise ValueError("No transcript was found for this video.")

    summary = summarize_long_text(
        text,
        max_length=max_length,
        min_length=min_length,
        final_compression=True
    )

    print("\n========== PERFORMANCE REPORT ==========")

    print(
        f"Total processing time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("=========================================\n")

    return summary
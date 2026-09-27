# YouTube Video Summarization 🎥🤖

An AI-powered YouTube video summarization system that extracts a video's transcript and generates a concise summary using the **BART Large CNN** transformer model from Hugging Face.

## 🚀 Project Overview

This project automatically summarizes YouTube videos by:

1. Extracting the YouTube video ID from the URL.
2. Fetching the video's transcript using the YouTube Transcript API.
3. Splitting long transcripts into smaller token-based chunks.
4. Summarizing each chunk using **BART Large CNN**.
5. Combining the generated summaries.
6. Applying a final summarization step to produce a concise final summary.

## 🧠 Architecture

```text
YouTube Video URL
        │
        ▼
Extract Video ID
        │
        ▼
YouTube Transcript API
        │
        ▼
Video Transcript
        │
        ▼
Token-Based Chunking
        │
        ▼
BART Large CNN
        │
        ▼
Chunk Summaries
        │
        ▼
Final Compression
        │
        ▼
Final Video Summary
```

## 🛠️ Technologies Used

* **Python**
* **Hugging Face Transformers**
* **BART Large CNN**
* **PyTorch**
* **YouTube Transcript API**
* **Hugging Face Tokenizer**

## 🤖 Model

The project uses:

```text
facebook/bart-large-cnn
```

BART Large CNN is a transformer-based sequence-to-sequence model that is fine-tuned for abstractive text summarization.

## ⚙️ How It Works

### 1. Extract YouTube Video ID

The system extracts the video ID from different YouTube URL formats, including:

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/shorts/VIDEO_ID
https://www.youtube.com/embed/VIDEO_ID
https://www.youtube.com/live/VIDEO_ID
```

### 2. Fetch the Transcript

The project uses `youtube-transcript-api` to retrieve the available transcript.

It attempts to support:

* English (`en`)
* Arabic (`ar`)

### 3. Token-Based Chunking

Long transcripts cannot always be passed directly to the model because transformer models have a maximum input length.

The project therefore uses the BART tokenizer to divide the transcript into chunks.

```python
MAX_CHUNK_TOKENS = 900
```

Each chunk is processed separately.

### 4. Summarize Each Chunk

Each transcript chunk is passed to BART:

```python
result = summarizer(
    chunk,
    max_length=150,
    min_length=30
)
```

The generated summaries are then stored.

### 5. Final Compression

If the transcript contains multiple chunks, their summaries are combined and passed through BART again.

This produces one final concise summary of the complete video.

## 📁 Project Structure

```text
youtube-video-summarization/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── summarizer.py
│
├── examples/
│   └── example_output.txt
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/zeyadgalal1/youtube-video-summarization.git
```

Move into the project directory:

```bash
cd youtube-video-summarization
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the project:

```bash
python src/main.py
```

Enter a YouTube video URL when prompted:

```text
Enter YouTube URL:
```

The system will:

```text
Fetching transcript...
Split transcript into X chunk(s).
Chunk 1/X summarized.
Chunk 2/X summarized.
...
--- Final Summary ---
```

## 📝 Example

### Input

```text
https://youtu.be/vLijZb9BpkE
```

### Processing

```text
YouTube URL
     ↓
Video ID
     ↓
Transcript
     ↓
900-token chunks
     ↓
BART summarization
     ↓
Combined summaries
     ↓
Final summary
```

### Output

```text
--- Final Summary ---

The system generates a concise summary
containing the main information from the video.
```

## ✨ Features

* 🎥 YouTube video URL processing
* 📝 Automatic transcript extraction
* 🌍 English and Arabic transcript support
* 🔤 Token-based transcript chunking
* 🤖 BART Large CNN summarization
* 📚 Long transcript handling
* 🧠 Multi-stage summarization
* 📄 Final concise video summary

## 🔮 Future Improvements

* Add a web interface using **Streamlit**
* Add speech-to-text for videos without transcripts
* Support more languages
* Add timestamps to important sections
* Allow users to download summaries
* Build a REST API
* Add a graphical user interface
* Add multiple summarization models for comparison


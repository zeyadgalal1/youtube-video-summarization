# YouTube Video Summarization 🎥🤖

An AI-powered YouTube Video Summarization application that extracts video transcripts and generates concise summaries using the **BART Large CNN** Transformer model from Hugging Face.

The application provides a user-friendly web interface built with **Streamlit** and is deployed on Streamlit Community Cloud.

## 🚀 Live Demo

**Try the application:** https://zeyadgalal1-youtube-video-summarization-app-ggprys.streamlit.app/

## 📸 Application Demo

<!-- Add a screenshot of your Streamlit application here -->

## 🚀 Project Overview

This project automatically summarizes YouTube videos through the following pipeline:

1. Extracts the YouTube video ID from the provided URL.
2. Fetches the available video transcript using the YouTube Transcript API.
3. Splits long transcripts into smaller token-based chunks.
4. Summarizes each chunk using BART Large CNN.
5. Combines the generated summaries.
6. Applies a final summarization step to generate one concise summary.

## 🧠 Architecture

```text
YouTube Video URL
        |
        v
Extract Video ID
        |
        v
YouTube Transcript API
        |
        v
Video Transcript
        |
        v
Token-Based Chunking
        |
        v
BART Large CNN
        |
        v
Chunk Summaries
        |
        v
Final Compression
        |
        v
Final Video Summary
```

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – Web application interface
* **Hugging Face Transformers** – NLP model integration
* **BART Large CNN** – Text summarization
* **PyTorch** – Deep learning framework
* **YouTube Transcript API** – Transcript extraction
* **Hugging Face Tokenizer** – Tokenization and chunking

## 🤖 Model

The project uses:

```text
facebook/bart-large-cnn
```

BART Large CNN is a Transformer-based sequence-to-sequence model fine-tuned for abstractive text summarization.

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

The application uses `youtube-transcript-api` to retrieve the available transcript.

### 3. Token-Based Chunking

Long transcripts cannot always be passed directly to the model because Transformer models have maximum input-length limitations.

The BART tokenizer is used to divide long transcripts into smaller chunks.

```python
MAX_CHUNK_TOKENS = 900
```

Each chunk is processed separately.

### 4. Summarize Each Chunk

Each transcript chunk is passed to the summarization pipeline:

```python
result = summarizer(
    chunk,
    max_length=150,
    min_length=30
)
```

The generated summaries are stored and combined.

### 5. Final Compression

When the transcript contains multiple chunks, their summaries are combined and passed through another summarization step to generate one concise final summary.

## ✨ Features

* 🎥 YouTube video URL processing
* 📝 Automatic transcript extraction
* 🔤 Token-based transcript chunking
* 🤖 BART Large CNN summarization
* 📚 Long transcript handling
* 🧠 Multi-stage summarization
* 📄 Concise final video summaries
* 🌐 Interactive Streamlit web interface
* ☁️ Cloud deployment

## 📁 Project Structure

Update this section to match your actual repository structure.

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

Navigate to the project directory:

```bash
cd youtube-video-summarization
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

If your Streamlit entry-point has a different filename, replace `app.py` with its actual name.

Open the local URL displayed in your terminal to access the application.

## 🔮 Future Improvements

* Add speech-to-text support for videos without available transcripts.
* Add timestamps to important sections.
* Allow users to download generated summaries.
* Build a REST API.
* Experiment with different summarization models.
* Improve summarization quality and processing efficiency.

## 👨‍💻 Author

**Zeyad Galal**

GitHub: [@zeyadgalal1](https://github.com/zeyadgalal1)

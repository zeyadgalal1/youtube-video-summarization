import streamlit as st

st.set_page_config(
    page_title="YouTube AI Summarizer",
    page_icon="🎥"
)

st.title("🎥 YouTube AI Summarizer")

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("Test Summarization"):

    if not youtube_url.strip():
        st.warning("Enter a YouTube URL first.")

    else:

        try:
            from main import summarize_youtube_video

            st.info("Starting summarization...")

            summary = summarize_youtube_video(
                youtube_url,
                max_length=80,
                min_length=20
            )

            st.success("✅ Summarization worked!")

            st.write(summary)

        except Exception as e:

            st.error("❌ Summarization failed.")

            st.exception(e)
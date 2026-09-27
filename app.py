import streamlit as st

from main import summarize_youtube_video


st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥"
)


st.title("🎥 YouTube Video Summarizer")

st.write(
    "Enter a YouTube video URL and get an AI-generated summary."
)


youtube_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


if st.button("Summarize Video"):

    if not youtube_url:

        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        try:

            with st.spinner(
                "Getting transcript and generating summary..."
            ):

                summary = summarize_youtube_video(
                    youtube_url
                )

            st.subheader("📝 Summary")

            st.write(summary)

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )
import streamlit as st
from main import summarize_youtube_video


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="YouTube AI Summarizer",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Summary */
    .summary-box {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        line-height: 1.7;
        font-size: 1.05rem;
    }

    .summary-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Language Notice
# -----------------------------

st.info("🇬🇧 English videos only")


# -----------------------------
# YouTube URL
# -----------------------------

st.markdown("### 🔗 YouTube Video")

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="Paste your YouTube video link here...",
    label_visibility="collapsed"
)


# -----------------------------
# Summary Settings
# -----------------------------

st.markdown("### ⚙️ Summary Settings")

summary_length = st.select_slider(
    "Summary Length",
    options=["Short", "Medium", "Long"],
    value="Medium"
)


length_description = {
    "Short": "Quick overview with the key points.",
    "Medium": "Balanced summary with the main ideas.",
    "Long": "More detailed summary with additional context."
}


st.caption(length_description[summary_length])


length_settings = {
    "Short": {
        "max_length": 80,
        "min_length": 20
    },
    "Medium": {
        "max_length": 150,
        "min_length": 40
    },
    "Long": {
        "max_length": 220,
        "min_length": 60
    }
}


selected_settings = length_settings[summary_length]


# -----------------------------
# Summarize Button
# -----------------------------

st.markdown("<br>", unsafe_allow_html=True)

summarize_button = st.button(
    "✨ Summarize Video",
    type="primary",
    use_container_width=True
)


# -----------------------------
# Summarization
# -----------------------------

if summarize_button:

    if not youtube_url.strip():

        st.warning(
            "Please paste a YouTube video URL first."
        )

    else:

        progress_bar = st.progress(0)
        status = st.empty()

        try:

            status.info(
                "🔗 Connecting to the video..."
            )

            progress_bar.progress(20)

            status.info(
                "📄 Retrieving transcript..."
            )

            progress_bar.progress(40)

            status.info(
                "🤖 Generating AI summary..."
            )

            progress_bar.progress(70)

            summary = summarize_youtube_video(
                youtube_url,
                max_length=selected_settings["max_length"],
                min_length=selected_settings["min_length"]
            )

            progress_bar.progress(100)

            status.success(
                "✅ Summary generated successfully!"
            )

            st.markdown(
                '<div class="summary-title">📝 AI Summary</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="summary-box">
                    {summary}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Summary Length",
                    f"{len(summary.split())} words"
                )

            with col2:

                st.metric(
                    "Selected Length",
                    summary_length
                )

        except Exception as e:

            progress_bar.empty()

            status.error(
                "❌ Something went wrong."
            )

            st.error(str(e))


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    """
    <div class="custom-footer">
        Built with Streamlit, Hugging Face Transformers & BART
    </div>
    """,
    unsafe_allow_html=True
)
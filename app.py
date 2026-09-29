import streamlit as st

try:
    from main import summarize_youtube_video
except Exception as e:
    st.error("Application startup error:")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube AI Summarizer",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #777;
        margin-bottom: 2.5rem;
    }

    /* Feature cards */
    .feature-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        text-align: center;
        height: 100%;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .feature-text {
        color: #777;
        font-size: 0.9rem;
    }

    /* Summary box */
    .summary-header {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎥 YouTube AI Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Turn long YouTube videos into clear, concise summaries using AI.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Features
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast</div>
            <div class="feature-text">
                Automatically process video transcripts.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Powered</div>
            <div class="feature-text">
                Uses BART to generate meaningful summaries.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Simple</div>
            <div class="feature-text">
                Paste a URL and get your summary.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------
# URL input
# --------------------------------------------------

st.markdown("### 🔗 YouTube Video")

youtube_url = st.text_input(
    "Paste your YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    label_visibility="collapsed"
)


# --------------------------------------------------
# Summary settings
# --------------------------------------------------

st.markdown("### ⚙️ Summary Settings")

summary_length = st.select_slider(
    "Summary Length",
    options=["Short", "Medium", "Long"],
    value="Medium"
)

st.caption("Short = quick overview  •  Medium = balanced  •  Long = more detail")



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


# --------------------------------------------------
# Summarize button
# --------------------------------------------------

summarize_button = st.button(
    "✨ Summarize Video",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Processing
# --------------------------------------------------

if summarize_button:

    if not youtube_url.strip():

        st.warning(
            "Please enter a YouTube video URL first."
        )

    else:

        progress_bar = st.progress(0)

        status = st.empty()

        try:

            status.info("🔎 Reading the YouTube video...")
            progress_bar.progress(20)

            status.info("📄 Retrieving transcript...")
            progress_bar.progress(40)

            status.info("🤖 Generating AI summary...")
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

            # --------------------------------------------------
            # Summary
            # --------------------------------------------------

            st.markdown(
                '<div class="summary-header">📝 Summary</div>',
                unsafe_allow_html=True
            )

            st.write(summary)

            # --------------------------------------------------
            # Summary information
            # --------------------------------------------------

            st.divider()

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


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Built with Streamlit, Hugging Face Transformers & BART
    </div>
    """,
    unsafe_allow_html=True
)
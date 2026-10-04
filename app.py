import streamlit as st
import config
from modules.extractor import extract_text_from_pdf, extract_text_from_youtube
from modules.vector_store import VectorStoreManager
from modules.rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Knowledge Distiller",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark UI/UX Styling
st.markdown("""
    <style>
    /* Dark Theme Backgrounds */
    .stApp {
        background-color: #0F111A;
        color: #E2E8F0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161926;
        border-right: 1px solid #2D3748;
    }

    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6C5CE7 0%, #a29bfe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.4);
    }

    /* Metric Display Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #A29BFE;
        font-weight: 700;
    }
    
    /* Expanders for Flashcards */
    .streamlit-expanderHeader {
        background-color: #1A1D2D !important;
        border-radius: 6px !important;
        color: #E2E8F0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State & Managers
if "v_store" not in st.session_state:
    st.session_state["v_store"] = VectorStoreManager()

if "rag" not in st.session_state:
    st.session_state["rag"] = RAGEngine()

v_store = st.session_state["v_store"]
rag = st.session_state["rag"]

# Header Title
st.title("⚡ Local Knowledge Distiller")
st.caption("AI/ML Powered Multimodal Video & PDF Knowledge Extraction System")

# Sidebar Configuration
st.sidebar.header("📥 Source Material")
source_type = st.sidebar.radio("Choose Input Format", ["PDF Document", "YouTube Video"])

# Document Ingestion Handler
if source_type == "PDF Document":
    uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_file and st.sidebar.button("Process Document"):
        with st.spinner("Parsing text and indexing chunks..."):
            text, accuracy = extract_text_from_pdf(uploaded_file)
            if text:
                v_store.add_document(uploaded_file.name, text)
                st.session_state["active_text"] = text
                st.session_state["model_accuracy"] = accuracy
                st.session_state["source_mode"] = "pdfplumber Structural Engine"
                st.sidebar.success("PDF Indexed Successfully!")
            else:
                st.sidebar.error("Could not extract text from this PDF.")

elif source_type == "YouTube Video":
    yt_url = st.sidebar.text_input("Enter YouTube Video URL")
    if yt_url and st.sidebar.button("Process Video"):
        with st.spinner("Extracting audio/transcript & indexing..."):
            try:
                text, accuracy, mode = extract_text_from_youtube(yt_url)
                v_store.add_document(yt_url, text)
                st.session_state["active_text"] = text
                st.session_state["model_accuracy"] = accuracy
                st.session_state["source_mode"] = mode
                st.sidebar.success("Video Indexed Successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to process video: {str(e)}")

# Real-Time Operational Performance Metrics Dashboard
if "model_accuracy" in st.session_state:
    st.markdown("---")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Extraction Engine", st.session_state["source_mode"])
    m_col2.metric("ML Model Accuracy", f"{st.session_state['model_accuracy']}%")
    m_col3.metric(
        "Confidence Status",
        "High Confidence" if st.session_state["model_accuracy"] > 80 else "Moderate Noise"
    )
    st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📝 Executive Summary", "🎴 Flashcards Deck", "💬 Interactive Q&A"])

if "active_text" in st.session_state:
    active_text = st.session_state["active_text"]

    with tab1:
        st.subheader("High-Yield Executive Summary")
        if st.button("Generate Summary"):
            with st.spinner("Synthesizing core takeaways..."):
                summary = rag.generate_summary(active_text)
                st.markdown(summary)

    with tab2:
        st.subheader("Study Flashcards")
        if st.button("Generate Flashcards Deck"):
            with st.spinner("Extracting key concepts..."):
                cards = rag.generate_flashcards(active_text)
                for idx, card in enumerate(cards):
                    with st.expander(f"📌 Card {idx+1}: {card.get('question', 'Concept Question')}"):
                        st.write(f"**Answer:** {card.get('answer', 'Detailed explanation')}")

    with tab3:
        st.subheader("Grounded Natural Language Q&A")
        user_query = st.text_input("Ask a question about your uploaded document or video:")
        if user_query:
            with st.spinner("Searching vector database..."):
                context, match_accuracy = v_store.query_context_with_accuracy(user_query)
                answer = rag.answer_question(user_query, context)
                
                # Display dynamic semantic match accuracy badge
                st.caption(f"🎯 **Semantic Search Relevance Match Score:** `{match_accuracy}%`")
                st.markdown("### Answer")
                st.write(answer)
else:
    st.info("👈 Please select a source material from the sidebar and click process to get started.")
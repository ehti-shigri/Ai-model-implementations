import os
import streamlit as st
from src.rag_engine import DocuLensIndex, render_page_image
st.set_page_config(page_title="DocuLens — Multimodal PDF QA", page_icon="🔍", layout="wide")

st.title("🔍 DocuLens")
st.caption("Hybrid Multimodal RAG (BM25 + Gemini Embeddings + FAISS + Gemini Vision)")

# --- Sidebar: API Key & Settings ---
with st.sidebar:
    st.header("Settings")

    # Auto-load key from Streamlit Secrets or environment; allow manual override
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    except Exception:
        default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        help="Get a free key at aistudio.google.com",
    )
    top_k = st.slider("Pages to inspect (Top-K)", min_value=1, max_value=5, value=3)
    dpi = st.select_slider("Page Render DPI", options=[100, 150, 200], value=150)

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Extracts text per page with PyMuPDF\n"
        "2. Indexes with BM25 + FAISS\n"
        "3. Sends Top-K page text & images to Gemini VLM"
    )

# --- PDF Upload & Session Caching ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if not api_key:
    st.warning("Enter your Gemini API key in the sidebar (or configure `GEMINI_API_KEY` in Streamlit Secrets) to start.")
    st.stop()

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_sig = f"{uploaded_file.name}_{len(pdf_bytes)}"

    # Only re-index when a new PDF is uploaded
    if st.session_state.get("file_sig") != file_sig:
        with st.spinner(f"Indexing '{uploaded_file.name}' (BM25 + Gemini Embeddings)..."):
            st.session_state["index"] = DocuLensIndex(pdf_bytes, api_key)
            st.session_state["file_sig"] = file_sig
            st.session_state["history"] = []

    doc_index: DocuLensIndex = st.session_state["index"]
    st.success(f"Ready: **{uploaded_file.name}** ({len(doc_index.pages)} pages indexed)")

    # --- Chat History ---
    for item in st.session_state.get("history", []):
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(f"**Answer:** {item['answer']}")
            st.caption(
                f"Cited pages: {item['evidence_page_number']} | "
                f"Retrieved candidates: {item['retrieved_pages']}"
            )
            with st.expander("View reasoning"):
                st.write(item["reasoning"])
            cols = st.columns(len(item["evidence_page_number"]))
            for col, p_num in zip(cols, item["evidence_page_number"]):
                img = render_page_image(pdf_bytes, p_num, dpi=120)
                col.image(img, caption=f"Page {p_num}", use_container_width=True)

    # --- Question Input ---
    question = st.chat_input("Ask a question about your PDF (tables, charts, or text)...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving pages and analyzing visuals..."):
                try:
                    result = doc_index.ask(question, top_k=top_k, dpi=dpi)
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.caption(
                        f"Cited pages: {result['evidence_page_number']} | "
                        f"Retrieved candidates: {result['retrieved_pages']}"
                    )
                    with st.expander("View reasoning"):
                        st.write(result["reasoning"])

                    cols = st.columns(len(result["evidence_page_number"]))
                    for col, p_num in zip(cols, result["evidence_page_number"]):
                        img = render_page_image(pdf_bytes, p_num, dpi=120)
                        col.image(img, caption=f"Page {p_num}", use_container_width=True)

                    st.session_state["history"].append({"question": question, **result})
                except Exception as exc:
                    st.error(str(exc))
else:
    st.info("Upload any PDF above to start asking questions.")

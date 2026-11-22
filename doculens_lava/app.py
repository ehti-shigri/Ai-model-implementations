import os
import streamlit as st
from src.rag_engine import DocuLensIndex, render_page_image

# --- Optimal Production Defaults (Hidden from End Users) ---
DEFAULT_TOP_K = 3
DEFAULT_RENDER_DPI = 150

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuLens | Multimodal Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Hide Sidebar Completely & Apply Clean SaaS Styling ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    .hero-wrap {
        text-align: center;
        padding: 1.2rem 0 1.5rem 0;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero-sub {
        font-size: 1.02rem;
        color: #94a3b8;
        margin-bottom: 0.85rem;
    }
    .doc-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: rgba(59, 130, 246, 0.12);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.25);
        margin: 0 0.25rem;
    }
    .cite-pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-right: 0.35rem;
    }
    .cand-pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        background-color: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Silently Load Backend API Key ---
try:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")).strip()
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

if not api_key:
    st.error("Service configuration error: `GEMINI_API_KEY` is not set in server secrets.")
    st.stop()

# --- Centered Landing View (When No PDF Is Active) ---
if "index" not in st.session_state:
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">📄 DocuLens</div>
            <div class="hero-sub">Multimodal Document Intelligence — Ask questions across tables, charts, and dense PDFs</div>
            <span class="doc-badge">Hybrid BM25 + FAISS</span>
            <span class="doc-badge">Multilingual Latin & CJK</span>
            <span class="doc-badge">Verifiable Page Citations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.8, 1])
    with center_col:
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Upload a PDF document to begin",
                type=["pdf"],
                help="Supports financial reports, technical manuals, charts, and scanned layouts.",
            )

        if uploaded_file is not None:
            pdf_bytes = uploaded_file.getvalue()
            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                st.session_state["index"] = DocuLensIndex(pdf_bytes, api_key)
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["pdf_name"] = uploaded_file.name
                st.session_state["history"] = []
                st.session_state["inspector_page"] = 1
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 1. Hybrid Retrieval")
            st.caption("Combines exact keyword search with dense semantic embeddings to pinpoint relevant pages.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 2. Visual & Text Reasoning")
            st.caption("Reads charts, multi-column tables, and diagrams directly alongside extracted page text.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 3. Instant Source Audit")
            st.caption("Every answer cites exact page numbers and opens the source page side by side.")
    st.stop()

# --- Active Workspace View (After PDF Upload) ---
doc_index: DocuLensIndex = st.session_state["index"]
pdf_bytes: bytes = st.session_state["pdf_bytes"]
pdf_name: str = st.session_state["pdf_name"]
total_pages = len(doc_index.pages)

# Top Header Bar with Document Info & Reset Action
head_left, head_right = st.columns([3.5, 1])
with head_left:
    st.markdown(f"### 📄 {pdf_name}")
    st.caption(f"{total_pages} pages indexed  •  Hybrid Multimodal RAG")
with head_right:
    st.markdown("<div style='padding-top: 0.6rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Upload New PDF", use_container_width=True):
        for key in ["index", "pdf_bytes", "pdf_name", "history", "inspector_page"]:
            st.session_state.pop(key, None)
        st.rerun()

st.divider()

# --- Split-Screen Workspace: Left = Document Q&A | Right = Source Page Inspector ---
chat_col, viewer_col = st.columns([1.3, 1.0], gap="large")
queued_question = None

with chat_col:
    st.subheader("Document Q&A")

    if not st.session_state["history"]:
        st.caption("Try a quick prompt or ask your own question below:")
        q_cols = st.columns(3)
        prompts = [
            "Summarize the key findings and main purpose of this document.",
            "Extract all key numerical metrics, tables, or financial figures.",
            "List the primary conclusions or recommendations with page citations.",
        ]
        labels = ["📋 Summary", "📊 Key Metrics", "✅ Conclusions"]
        for col, label, p_text in zip(q_cols, labels, prompts):
            if col.button(label, use_container_width=True):
                queued_question = p_text

    chat_box = st.container(height=520, border=True)
    with chat_box:
        if not st.session_state["history"] and not queued_question:
            st.markdown(
                "<div style='text-align:center; color:#888; padding-top:190px;'>"
                "Ask anything about tables, charts, or text inside your document."
                "</div>",
                unsafe_allow_html=True,
            )

        for turn_idx, item in enumerate(st.session_state["history"]):
            with st.chat_message("user"):
                st.write(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item["answer"])

                cited_html = "".join(
                    f"<span class='cite-pill'>✓ Cited Page {p}</span>"
                    for p in item["evidence_page_number"]
                )
                cand_html = f"<span class='cand-pill'>Reviewed: {item['retrieved_pages']}</span>"
                st.markdown(f"{cited_html} {cand_html}", unsafe_allow_html=True)

                col_r, col_v = st.columns([3, 1.3])
                with col_r:
                    with st.expander("View reasoning"):
                        st.write(item["reasoning"])
                with col_v:
                    if st.button(
                        f"View Page {item['evidence_page_number'][0]}",
                        key=f"inspect_btn_{turn_idx}",
                        use_container_width=True,
                    ):
                        st.session_state["inspector_page"] = item["evidence_page_number"][0]
                        st.rerun()

    user_input = st.chat_input("Ask a question about your PDF...")
    active_question = queued_question or user_input

    if active_question:
        with chat_box:
            with st.chat_message("user"):
                st.write(active_question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing pages..."):
                    try:
                        result = doc_index.ask(
                            active_question,
                            top_k=DEFAULT_TOP_K,
                            dpi=DEFAULT_RENDER_DPI,
                        )
                        st.session_state["history"].append({"question": active_question, **result})
                        if result["evidence_page_number"]:
                            st.session_state["inspector_page"] = result["evidence_page_number"][0]
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

# --- Right Pane: Live Source Page Inspector ---
with viewer_col:
    st.subheader("Source Page Inspector")

    current_page = int(st.session_state.get("inspector_page", 1))
    current_page = max(1, min(current_page, total_pages))

    nav_prev, nav_sel, nav_next = st.columns([1, 2.2, 1])
    with nav_prev:
        if st.button("◀ Prev", disabled=(current_page <= 1), use_container_width=True):
            st.session_state["inspector_page"] = current_page - 1
            st.rerun()
    with nav_sel:
        selected_page = st.selectbox(
            "Jump to page",
            options=list(range(1, total_pages + 1)),
            index=current_page - 1,
            format_func=lambda p: f"Page {p} of {total_pages}",
            label_visibility="collapsed",
        )
        if selected_page != current_page:
            st.session_state["inspector_page"] = selected_page
            st.rerun()
    with nav_next:
        if st.button("Next ▶", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state["inspector_page"] = current_page + 1
            st.rerun()

    if st.session_state["history"]:
        latest_cites = st.session_state["history"][-1]["evidence_page_number"]
        if len(latest_cites) > 1:
            st.caption("Cited pages in latest answer:")
            cite_cols = st.columns(len(latest_cites))
            for c_col, p_num in zip(cite_cols, latest_cites):
                if c_col.button(f"Page {p_num}", key=f"jump_cite_{p_num}", use_container_width=True):
                    st.session_state["inspector_page"] = p_num
                    st.rerun()

    with st.container(border=True):
        page_img = render_page_image(pdf_bytes, current_page, dpi=140)
        st.image(
            page_img,
            caption=f"{pdf_name} — Page {current_page} / {total_pages}",
            use_container_width=True,
        )
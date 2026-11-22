import os
import streamlit as st
from src.rag_engine import DocuLensIndex, render_page_image

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuLens | Enterprise Multimodal Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Subtle Professional Styling ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .doc-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: rgba(59, 130, 246, 0.12);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.25);
        margin-right: 0.4rem;
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

# --- Load API Key Safely (Secrets -> Environment -> Manual Input) ---
try:
    secret_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    secret_key = os.environ.get("GEMINI_API_KEY", "")

# --- Sidebar: Workspace Controls ---
with st.sidebar:
    st.markdown("### 📄 DocuLens")
    st.caption("Multimodal Hybrid RAG Workspace")
    st.divider()

    # Document Uploader in Sidebar so main canvas stays focused
    uploaded_file = st.file_uploader(
        "Upload Document (PDF)",
        type=["pdf"],
        help="Supports digital reports, financial tables, charts, and scanned layouts.",
    )

    st.divider()
    st.markdown("**Retrieval & Vision Controls**")
    top_k = st.slider(
        "Candidate Pages (Top-K)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of top-ranked pages sent to Gemini Vision.",
    )
    dpi = st.select_slider(
        "Visual Render Resolution",
        options=[100, 150, 200],
        value=150,
        help="Use 200 DPI for dense tables, small chemical codes, or fine print.",
    )

    st.divider()
    # Keep API key clean if already configured in Streamlit Secrets
    if secret_key:
        st.success("API Key: Connected via Secrets", icon="✅")
        with st.expander("Override API Key"):
            manual_key = st.text_input("Custom Gemini API Key", type="password")
        api_key = manual_key.strip() if manual_key else secret_key
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get a free key at aistudio.google.com",
        ).strip()

    if st.session_state.get("history"):
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["inspector_page"] = 1
            st.rerun()

# --- Top Executive Header ---
st.markdown(
    """
    <div style="margin-bottom: 0.75rem;">
        <h2 style="margin-bottom: 0.2rem;">DocuLens — Document Intelligence</h2>
        <span class="doc-badge">Hybrid BM25 + FAISS RRF</span>
        <span class="doc-badge">Multilingual Latin & CJK</span>
        <span class="doc-badge">Gemini 3.5 Flash-Lite Vision</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.warning("Configure your `GEMINI_API_KEY` in the sidebar (or Streamlit Cloud Secrets) to activate the workspace.")
    st.stop()

# --- Landing State (Before PDF Upload) ---
if uploaded_file is None:
    st.info("👈 Upload a PDF document in the left sidebar to initialize the multimodal search index.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.container(border=True):
            st.markdown("#### 1. Hybrid Retrieval")
            st.caption("Fuses lexical BM25 keyword matching with dense semantic embeddings via Reciprocal Rank Fusion.")
    with col_b:
        with st.container(border=True):
            st.markdown("#### 2. Visual Reasoning")
            st.caption("Inspects both extracted text and rendered page images to read charts, diagrams, and complex tables.")
    with col_c:
        with st.container(border=True):
            st.markdown("#### 3. Verifiable Citations")
            st.caption("Every answer is grounded in explicit page numbers with side-by-side visual source inspection.")
    st.stop()

# --- Indexing & Session State Management ---
pdf_bytes = uploaded_file.getvalue()
file_sig = f"{uploaded_file.name}_{len(pdf_bytes)}"

if st.session_state.get("file_sig") != file_sig:
    with st.spinner(f"Building hybrid BM25 + FAISS index for '{uploaded_file.name}'..."):
        st.session_state["index"] = DocuLensIndex(pdf_bytes, api_key)
        st.session_state["file_sig"] = file_sig
        st.session_state["history"] = []
        st.session_state["inspector_page"] = 1

doc_index: DocuLensIndex = st.session_state["index"]
total_pages = len(doc_index.pages)

# --- Document Telemetry Bar ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Document", uploaded_file.name[:24] + ("..." if len(uploaded_file.name) > 24 else ""))
m2.metric("Indexed Pages", f"{total_pages} pages")
m3.metric("Search Context", f"Top-{top_k} RRF")
m4.metric("Render Quality", f"{dpi} DPI")

st.divider()

# --- Split-Screen Workspace: Left = Q&A Chat | Right = Live Evidence Viewer ---
chat_col, viewer_col = st.columns([1.35, 1.0], gap="large")

# Determine if a quick prompt button was clicked
queued_question = None

with chat_col:
    st.subheader("Document Q&A")

    # Quick-start audit prompts when conversation is empty
    if not st.session_state["history"]:
        st.caption("Select a quick audit prompt or ask your own question below:")
        q_cols = st.columns(3)
        prompts = [
            "Summarize the key findings and main purpose of this document.",
            "Extract all key numerical metrics, tables, or financial figures.",
            "List the primary conclusions or recommendations with page citations.",
        ]
        for idx, (col, p_text) in enumerate(zip(q_cols, prompts)):
            labels = ["📋 Executive Summary", "📊 Extract Key Metrics", "✅ Main Conclusions"]
            if col.button(labels[idx], use_container_width=True):
                queued_question = p_text

    # Render Conversation History
    chat_box = st.container(height=520, border=True)
    with chat_box:
        if not st.session_state["history"] and not queued_question:
            st.markdown(
                "<div style='text-align:center; color:#888; padding-top:180px;'>"
                "Ask any question about tables, charts, or text inside your PDF."
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
                cand_html = (
                    f"<span class='cand-pill'>Searched: {item['retrieved_pages']}</span>"
                )
                st.markdown(f"{cited_html} {cand_html}", unsafe_allow_html=True)

                col_r, col_v = st.columns([3, 1.3])
                with col_r:
                    with st.expander("View reasoning trace"):
                        st.write(item["reasoning"])
                with col_v:
                    if st.button(
                        f"Inspect Page {item['evidence_page_number'][0]}",
                        key=f"inspect_btn_{turn_idx}",
                        use_container_width=True,
                    ):
                        st.session_state["inspector_page"] = item["evidence_page_number"][0]
                        st.rerun()

    # Chat Input
    user_input = st.chat_input("Ask a question about this document...")
    active_question = queued_question or user_input

    if active_question:
        with chat_box:
            with st.chat_message("user"):
                st.write(active_question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving top pages and analyzing visuals..."):
                    try:
                        result = doc_index.ask(active_question, top_k=top_k, dpi=dpi)
                        st.session_state["history"].append({"question": active_question, **result})
                        # Automatically focus the right-hand viewer on the primary cited page
                        if result["evidence_page_number"]:
                            st.session_state["inspector_page"] = result["evidence_page_number"][0]
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Inference error: {exc}")

# --- Right Pane: Live Document & Citation Inspector ---
with viewer_col:
    st.subheader("Source Page Inspector")

    # Sync page selector with session state
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

    # Quick-jump pills if the latest answer cited multiple pages
    if st.session_state["history"]:
        latest_cites = st.session_state["history"][-1]["evidence_page_number"]
        if len(latest_cites) > 1:
            st.caption("Cited pages in latest answer:")
            cite_cols = st.columns(len(latest_cites))
            for c_col, p_num in zip(cite_cols, latest_cites):
                if c_col.button(f"Page {p_num}", key=f"jump_cite_{p_num}", use_container_width=True):
                    st.session_state["inspector_page"] = p_num
                    st.rerun()

    # Render selected page image inside bordered container
    with st.container(border=True):
        page_img = render_page_image(pdf_bytes, current_page, dpi=140)
        st.image(
            page_img,
            caption=f"{uploaded_file.name} — Page {current_page} / {total_pages}",
            use_container_width=True,
        )
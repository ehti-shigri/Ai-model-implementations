import json
import re
import time
import faiss
import numpy as np
import pymupdf
from PIL import Image
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
VLM_MODEL = "gemini-3.5-flash-lite"


def tokenize(text: str) -> list[str]:
    """Simple multilingual tokenizer: splits Latin/number words and individual CJK characters."""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff\u3040-\u30ff]", text.lower())


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract 1-based page numbers and text from an uploaded PDF."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append({
            "page_number": i,
            "text": text if text else f"[Scanned or image-only page {i}]",
        })
    doc.close()
    return pages


def render_page_image(pdf_bytes: bytes, page_number: int, dpi: int = 150) -> Image.Image:
    """Render a single 1-based PDF page into an RGB PIL image."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[int(page_number) - 1]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


class DocuLensIndex:
    """Lightweight Hybrid RAG index: BM25 (keyword) + Gemini Embeddings & FAISS (semantic)."""

    def __init__(self, pdf_bytes: bytes, api_key: str):
        self.pdf_bytes = pdf_bytes
        self.client = genai.Client(api_key=api_key)
        self.pages = extract_pdf_pages(pdf_bytes)

        # 1. Build BM25 keyword index
        corpus_tokens = [tokenize(p["text"]) or ["empty"] for p in self.pages]
        self.bm25 = BM25Okapi(corpus_tokens)

        # 2. Build FAISS dense vector index using gemini-embedding-001 (batched by 50 pages)
        embeddings = self._embed_texts(
            [p["text"][:4000] for p in self.pages],
            task_type="RETRIEVAL_DOCUMENT",
        )
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    def _embed_texts(self, texts: list[str], task_type: str, batch_size: int = 50) -> np.ndarray:
        """Embed a list of strings in safe batches and return a float32 NumPy matrix."""
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            res = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                ),
            )
            vectors.extend([emb.values for emb in res.embeddings])
        return np.array(vectors, dtype="float32")

    def retrieve(self, question: str, top_k: int = 3) -> list[int]:
        """Return Top-K 1-based page numbers using Reciprocal Rank Fusion (BM25 + FAISS)."""
        n_pages = len(self.pages)
        top_k = min(top_k, n_pages)

        # Keyword ranking (BM25)
        bm25_scores = self.bm25.get_scores(tokenize(question))
        bm25_order = np.argsort(bm25_scores)[::-1]

        # Semantic ranking (FAISS cosine similarity)
        q_vec = self._embed_texts([question], task_type="RETRIEVAL_QUERY")
        faiss.normalize_L2(q_vec)
        _, faiss_order = self.faiss_index.search(q_vec, n_pages)
        faiss_order = faiss_order[0]

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(bm25_order):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(faiss_order):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)

        ranked_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [self.pages[idx]["page_number"] for idx in ranked_indices]

    def ask(self, question: str, top_k: int = 3, dpi: int = 150) -> dict:
        """Retrieve top pages and send both text + rendered images to Gemini VLM."""
        candidate_pages = self.retrieve(question, top_k=top_k)

        prompt = (
            "You are DocuLens, a precise multimodal document QA assistant. "
            "Use BOTH the extracted text (for exact spelling, numbers, and codes) "
            "and the page images (for charts, tables, diagrams, and layout).\n"
            f"Question: {question}\n\n"
            "Rules:\n"
            "1. Answer in the EXACT same language as the question.\n"
            "2. In 'reasoning', write 2-3 concise lines citing exact values, labels, or calculations.\n"
            "3. In 'answer', give a clear, direct, accurate answer grounded ONLY in the document.\n"
            "4. In 'evidence_page_number', return a JSON array of ONLY the integer page numbers you used.\n"
            "5. If the answer is not in the provided pages, say so honestly.\n"
            "6. Return valid JSON with keys: 'reasoning', 'answer', 'evidence_page_number'."
        )

        contents = [prompt]
        for p_num in candidate_pages:
            page_text = self.pages[p_num - 1]["text"][:2500]
            page_img = render_page_image(self.pdf_bytes, p_num, dpi=dpi)
            contents.append(f"--- Page {p_num} Extracted Text ---\n{page_text}")
            contents.append(page_img)

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=VLM_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                result = json.loads(response.text.strip())
                raw_pages = result.get("evidence_page_number", candidate_pages[:1])
                valid_pages = sorted(
                    {
                        int(p)
                        for p in raw_pages
                        if str(p).strip().isdigit() and 1 <= int(p) <= len(self.pages)
                    }
                )
                return {
                    "answer": str(result.get("answer", "")).strip(),
                    "reasoning": str(result.get("reasoning", "")).strip(),
                    "evidence_page_number": valid_pages or candidate_pages[:1],
                    "retrieved_pages": candidate_pages,
                }
            except Exception as err:
                if attempt == 2:
                    raise RuntimeError(f"Gemini API error: {err}") from err
                time.sleep(3 * (attempt + 1))
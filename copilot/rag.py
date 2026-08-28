"""
RAG index over maintenance manuals.

- Loads .md/.txt/.pdf from copilot/manuals/
- Chunks them, embeds with a local sentence-transformers model (no API key)
- Stores vectors in a persistent Chroma collection
- retrieve(query, k) returns the top-k relevant chunks with source metadata

Embedding model is local/offline by design; the first run downloads the weights.
"""
import os
import glob
import re

_HERE = os.path.dirname(__file__)
MANUALS_DIR = os.path.join(_HERE, "manuals")
CHROMA_DIR = os.path.join(_HERE, ".chroma")
COLLECTION = "maintenance_manuals"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

_client = None
_collection = None
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _embed(texts):
    return _get_embedder().encode(texts, normalize_embeddings=True).tolist()


def _read_file(path):
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            print(f"[rag] skip PDF {path}: {e}")
            return ""
    with open(path, "r", errors="ignore") as f:
        return f.read()


def _chunk(text, target=900, overlap=150):
    """Split on paragraph boundaries, packing up to ~target chars with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= target:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"})
    return _collection


def build_index(force=False):
    """Index all manuals. Idempotent unless force=True (which clears first)."""
    col = _get_collection()
    if force and col.count() > 0:
        import chromadb
        _client.delete_collection(COLLECTION)
        globals()["_collection"] = None
        col = _get_collection()
    if col.count() > 0:
        print(f"[rag] index already has {col.count()} chunks; skip (use force=True to rebuild)")
        return col.count()

    paths = []
    for ext in ("*.md", "*.txt", "*.pdf"):
        paths.extend(glob.glob(os.path.join(MANUALS_DIR, ext)))
    if not paths:
        print(f"[rag] no manuals found in {MANUALS_DIR}")
        return 0

    ids, docs, metas = [], [], []
    for path in sorted(paths):
        text = _read_file(path)
        if not text.strip():
            continue
        src = os.path.basename(path)
        for i, ch in enumerate(_chunk(text)):
            ids.append(f"{src}::chunk{i}")
            docs.append(ch)
            metas.append({"source": src, "chunk": i})
    if not docs:
        print("[rag] no chunks produced")
        return 0

    embs = _embed(docs)
    col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    print(f"[rag] indexed {len(docs)} chunks from {len(paths)} file(s)")
    return len(docs)


def retrieve(query, k=4):
    """Return top-k chunks: list of {text, source, chunk, score}."""
    col = _get_collection()
    if col.count() == 0:
        build_index()
    q_emb = _embed([query])
    res = col.query(query_embeddings=q_emb, n_results=k,
                    include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "source": meta.get("source"),
                    "chunk": meta.get("chunk"), "score": round(1 - dist, 3)})
    return out


if __name__ == "__main__":
    n = build_index(force=True)
    print(f"built index with {n} chunks")
    if n:
        for r in retrieve("what should I do when RUL is critical?", k=3):
            print(f"\n[{r['source']} #{r['chunk']} score={r['score']}]\n{r['text'][:180]}...")
            
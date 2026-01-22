import json
import math
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = BASE_DIR

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def chunk_text(text, chunk_size=600):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_sources():
    sources = []
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in {".md", ".txt"}:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            chunks = chunk_text(text)
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]
            chunks = lines
        else:
            continue
        for idx, chunk in enumerate(chunks):
            sources.append(
                {
                    "source": name,
                    "chunk_id": idx,
                    "text": chunk,
                    "type": ext.replace(".", ""),
                }
            )
    return sources


def build_index(chunks):
    doc_freq = {}
    doc_tokens = []
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        unique = set(tokens)
        for token in unique:
            doc_freq[token] = doc_freq.get(token, 0) + 1
        doc_tokens.append(tokens)

    total_docs = len(chunks)
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((total_docs + 1) / (df + 1)) + 1.0

    vectors = []
    for tokens in doc_tokens:
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        vec = {token: count * idf[token] for token, count in tf.items()}
        vectors.append(vec)

    return idf, vectors


def cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for token, weight in vec_a.items():
        if token in vec_b:
            dot += weight * vec_b[token]
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_query_vector(query, idf):
    tokens = tokenize(query)
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return {token: tf[token] * idf.get(token, 0.0) for token in tf}


class SearchServer(SimpleHTTPRequestHandler):
    sources = load_sources()
    idf, vectors = build_index(sources)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            response = self.run_search(query)
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def run_search(self, query, top_k=5):
        if not query.strip():
            return {"query": query, "results": []}
        q_vec = build_query_vector(query, self.idf)
        scored = []
        for idx, vec in enumerate(self.vectors):
            score = cosine_similarity(q_vec, vec)
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True, key=lambda item: item[0])
        results = []
        for score, idx in scored[:top_k]:
            chunk = self.sources[idx]
            snippet = chunk["text"][:240]
            results.append(
                {
                    "score": round(score, 4),
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "type": chunk["type"],
                    "snippet": snippet,
                }
            )
        return {"query": query, "results": results}


def run():
    os.chdir(STATIC_DIR)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), SearchServer)
    print("MVP server running on http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    run()

"""
Enterprise Intelligent Search MVP
Works offline with TF-IDF, upgrades to OpenAI if available
"""

# SSL fix - MUST be at the very top before any imports
import ssl
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

import json
import math
import re
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = BASE_DIR

# Try to import numpy (optional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Try to load OpenAI with timeout
OPENAI_AVAILABLE = False
openai_client = None

print("Checking OpenAI connection (3s timeout)...")
try:
    import httpx
    from openai import OpenAI
    # Create client with SSL verification disabled and short timeout
    http_client = httpx.Client(verify=False, timeout=3.0)
    openai_client = OpenAI(http_client=http_client)
    # Quick test with timeout
    openai_client.models.list()
    OPENAI_AVAILABLE = True
    print("✓ OpenAI connected (embeddings + GPT-4o)")
except Exception as e:
    print(f"⚠ OpenAI not available: {str(e)[:50]}")
    print("  → Using TF-IDF (works offline)")

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with",
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


SOURCE_LABELS = {
    "team_directory.csv": ("Team Directory", "👤"),
    "customer_contracts.csv": ("Customer Contracts", "📄"),
    "support_tickets.csv": ("Support Tickets", "🎫"),
    "employee_handbook.md": ("Employee Handbook", "📘"),
    "engineering_runbook.md": ("Engineering Runbook", "🔧"),
    "security_policies.md": ("Security Policies", "🔒"),
    "product_faq.md": ("Product FAQ", "❓"),
    "ops_sop.md": ("Operations SOP", "📋"),
}

CSV_HEADERS = {
    "team_directory.csv": ["name", "team", "role", "location", "manager"],
    "customer_contracts.csv": ["account_id", "segment", "term_months", "renewal_date", "notes"],
    "support_tickets.csv": ["ticket_id", "category", "summary", "owner", "status"],
}


def format_csv_row(filename, row_text):
    """Convert CSV row to formatted display"""
    headers = CSV_HEADERS.get(filename)
    if not headers:
        return row_text
    
    values = [v.strip() for v in row_text.split(",")]
    if len(values) != len(headers):
        return row_text
    
    # Skip header row
    if values == headers:
        return None
    
    parts = []
    for header, value in zip(headers, values):
        label = header.replace("_", " ").title()
        parts.append(f"**{label}:** {value}")
    
    return " | ".join(parts)


def load_sources():
    sources = []
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        label, icon = SOURCE_LABELS.get(name, (name, "📁"))
        
        if ext in {".md", ".txt"}:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                sources.append({
                    "source": name,
                    "source_label": label,
                    "icon": icon,
                    "chunk_id": idx,
                    "text": chunk,
                    "display_text": chunk,
                    "type": "document",
                    "is_structured": False,
                })
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]
            for idx, line in enumerate(lines):
                formatted = format_csv_row(name, line)
                if formatted is None:  # Skip header
                    continue
                sources.append({
                    "source": name,
                    "source_label": label,
                    "icon": icon,
                    "chunk_id": idx,
                    "text": line,  # Raw for search
                    "display_text": formatted,  # Formatted for display
                    "type": "record",
                    "is_structured": True,
                })
    return sources


def get_openai_embeddings(texts):
    """Get embeddings from OpenAI"""
    if not OPENAI_AVAILABLE:
        return None
    try:
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def build_tfidf_index(chunks):
    """Build TF-IDF index (always available, works offline)"""
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


def build_embedding_index(chunks):
    """Build embedding index using OpenAI (optional)"""
    if not OPENAI_AVAILABLE or not NUMPY_AVAILABLE:
        return None
    
    print(f"Building OpenAI embedding index for {len(chunks)} chunks...")
    texts = [c["text"][:8000] for c in chunks]
    
    # Batch in groups of 100
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        embeddings = get_openai_embeddings(batch)
        if embeddings is None:
            print("  ⚠ Embedding failed, using TF-IDF only")
            return None
        all_embeddings.extend(embeddings)
    
    print("✓ Embedding index built!")
    return np.array(all_embeddings)


def cosine_tfidf(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_embedding(vec_a, vec_b):
    if not NUMPY_AVAILABLE:
        return 0.0
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_query_vector(query, idf):
    tokens = tokenize(query)
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return {token: tf[token] * idf.get(token, 0.0) for token in tf}


def route_intent(query):
    query_lower = query.lower()
    if re.search(r"(acct|inc|req)-\d+", query_lower):
        return "exact_match"
    elif any(x in query_lower for x in ["who", "employee", "team", "manager", "director"]):
        return "people"
    elif any(x in query_lower for x in ["ticket", "issue", "bug", "incident"]):
        return "tickets"
    elif any(x in query_lower for x in ["contract", "account", "smb", "enterprise"]):
        return "contracts"
    else:
        return "semantic"


def generate_answer(query, context_chunks, sources_meta):
    """Generate answer using GPT-4o (optional)"""
    if not OPENAI_AVAILABLE or openai_client is None:
        return None
    
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(context_chunks, sources_meta)):
        context_parts.append(f"[{i+1}] (Source: {meta['source']})\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer based ONLY on the context below. Be concise. Cite sources using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an enterprise search assistant. Answer questions based on provided context with citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


class SearchServer(SimpleHTTPRequestHandler):
    sources = None
    idf = None
    tfidf_vectors = None
    embeddings = None
    structured_indices = None
    embeddings_available = False

    @classmethod
    def initialize(cls):
        if cls.sources is None:
            print("\nLoading data sources...")
            cls.sources = load_sources()
            # Track structured data (CSV) indices
            cls.structured_indices = [i for i, s in enumerate(cls.sources) if s["is_structured"]]
            
            # Always build TF-IDF (works offline)
            print("Building TF-IDF index...")
            cls.idf, cls.tfidf_vectors = build_tfidf_index(cls.sources)
            
            # Try OpenAI embeddings (optional enhancement)
            cls.embeddings = build_embedding_index(cls.sources)
            cls.embeddings_available = cls.embeddings is not None
            
            print(f"\n✓ Indexed {len(cls.sources)} chunks ({len(cls.structured_indices)} structured records)")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            use_llm = params.get("llm", ["false"])[0].lower() == "true"
            response = self.run_search(query, use_llm=use_llm)
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

    def run_search(self, query, use_llm=False):
        if not query.strip():
            return {"query": query, "results": [], "intent": None, "answer": None,
                    "embeddings_available": self.embeddings_available, "llm_available": OPENAI_AVAILABLE}

        intent = route_intent(query)

        # Exact ID match
        id_match = re.search(r"(ACCT|INC|REQ)-\d+", query.upper())
        if id_match:
            exact_id = id_match.group()
            results = []
            for idx in self.structured_indices:
                chunk = self.sources[idx]
                if exact_id in chunk["text"].upper():
                    results.append({
                        "score": 1.0,
                        "source": chunk["source_label"],
                        "icon": chunk["icon"],
                        "chunk_id": chunk["chunk_id"],
                        "type": chunk["type"],
                        "snippet": chunk["display_text"][:400],
                    })
            
            answer = None
            if use_llm and results:
                answer = generate_answer(query, [r["snippet"] for r in results[:5]],
                                        [{"source": r["source"]} for r in results[:5]])
            
            return {"query": query, "intent": "exact_match", "results": results, "answer": answer,
                    "embeddings_available": self.embeddings_available, "llm_available": OPENAI_AVAILABLE}

        # Determine search scope
        search_indices = self.structured_indices if intent in ["people", "tickets", "contracts"] else range(len(self.sources))
        
        scored = []
        
        # Use OpenAI embeddings if available, otherwise TF-IDF
        if self.embeddings_available and self.embeddings is not None and NUMPY_AVAILABLE:
            query_emb = get_openai_embeddings([query])
            if query_emb:
                query_emb = query_emb[0]
                for idx in search_indices:
                    score = cosine_embedding(query_emb, self.embeddings[idx])
                    if score > 0.2:
                        scored.append((score, idx))
            else:
                # Fallback to TF-IDF
                q_vec = build_query_vector(query, self.idf)
                for idx in search_indices:
                    score = cosine_tfidf(q_vec, self.tfidf_vectors[idx])
                    if score > 0:
                        scored.append((score, idx))
        else:
            # TF-IDF (default, works offline)
            q_vec = build_query_vector(query, self.idf)
            for idx in search_indices:
                score = cosine_tfidf(q_vec, self.tfidf_vectors[idx])
                if score > 0:
                    scored.append((score, idx))

        scored.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, idx in scored[:20]:
            chunk = self.sources[idx]
            results.append({
                "score": round(float(score), 4),
                "source": chunk["source_label"],
                "icon": chunk["icon"],
                "chunk_id": chunk["chunk_id"],
                "type": chunk["type"],
                "snippet": chunk["display_text"][:400],
            })

        # Fallback to substring search
        if not results:
            needle = query.strip().lower()
            for idx in search_indices:
                chunk = self.sources[idx]
                if needle in chunk["text"].lower():
                    results.append({
                        "score": 0.5,
                        "source": chunk["source_label"],
                        "icon": chunk["icon"],
                        "chunk_id": chunk["chunk_id"],
                        "type": chunk["type"],
                        "snippet": chunk["display_text"][:400],
                    })
                    if len(results) >= 20:
                        break

        answer = None
        if use_llm and results:
            answer = generate_answer(query, [r["snippet"] for r in results[:5]],
                                    [{"source": r["source"]} for r in results[:5]])

        return {"query": query, "intent": intent, "results": results, "answer": answer,
                "embeddings_available": self.embeddings_available, "llm_available": OPENAI_AVAILABLE}


def run():
    SearchServer.initialize()
    os.chdir(STATIC_DIR)
    
    # Allow port reuse
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", 8000), SearchServer)
    
    print("\n" + "=" * 50)
    print("MVP server running on http://127.0.0.1:8000")
    print("=" * 50)
    print(f"Search:  {'✓ OpenAI Semantic' if SearchServer.embeddings_available else '✓ TF-IDF (offline)'}")
    print(f"Answers: {'✓ GPT-4o' if OPENAI_AVAILABLE else '✗ Disabled (no API)'}")
    print("=" * 50 + "\n")
    server.serve_forever()


if __name__ == "__main__":
    run()

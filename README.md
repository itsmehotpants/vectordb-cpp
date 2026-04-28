# VectorDB — Vector Database Engine from Scratch

> A production-grade vector database built in C++ with HNSW, KD-Tree, and Brute Force ANN search, a REST API backend, and a full RAG pipeline using local LLMs via Ollama.

---

## What This Is

Most developers use vector databases as a black box. This project builds one from scratch.

It implements the core algorithms that power production databases like Pinecone, Weaviate, and Chroma — and layers a full RAG (Retrieval-Augmented Generation) pipeline on top using a local LLM. The result: you can insert text documents, ask questions in natural language, and get answers grounded in your own data — no cloud, no API keys.

---

## Tech Stack

- **Language:** C++17
- **HTTP Server:** [cpp-httplib](https://github.com/yhirose/cpp-httplib) (single-header)
- **Local LLM runtime:** [Ollama](https://ollama.com)
- **Embedding model:** `nomic-embed-text` (768-dimensional vectors)
- **LLM:** `llama3.2`
- **Frontend:** Vanilla HTML/CSS/JS

---

## Features

- Three ANN search algorithms implemented from scratch: **Brute Force**, **KD-Tree**, and **HNSW**
- Pluggable distance metrics: **Cosine**, **Euclidean**, **Manhattan**
- Real-time **benchmarking** — compare all three algorithms side-by-side with microsecond-level timing
- **HNSW graph visualizer** — inspect layers, nodes, and edges in the browser
- **Document Q&A (RAG pipeline)** — insert any text, ask questions, get LLM answers backed by retrieved context
- Thread-safe inserts and searches via `std::mutex`
- Full REST API — 10 endpoints, all returning JSON
- Interactive web UI — no framework needed

---

## Architecture

```
main.cpp (1100+ lines)
├── Distance Metrics        cosine, euclidean, manhattan
├── BruteForce              O(N·d) linear scan
├── KDTree                  O(log N) axis-aligned pruning
├── HNSW                    O(log N) multilayer graph, approximate
├── VectorDB                wraps all 3 — 16D demo vectors, benchmarking
├── DocumentDB              HNSW only — real 768D Ollama embeddings
├── OllamaClient            HTTP calls to /api/embeddings and /api/generate
├── chunkText()             splits docs into 250-word overlapping windows
└── main()                  httplib::Server — 10 REST endpoints
```

Two databases run in parallel:

- **VectorDB** — 16-dimensional demo vectors across all 3 algorithms (for comparison and benchmarking)
- **DocumentDB** — 768-dimensional real text embeddings, HNSW only (for actual RAG)

---

## How the Algorithms Work

### Brute Force
Computes distance from the query vector to every single vector in the database. Exact results, but O(N) — doubles in cost as data doubles. Used as a correctness baseline.

### KD-Tree
Builds a binary tree by splitting space along alternating dimensions. Prunes entire subtrees during search when they can't contain a closer neighbor. Works well at low dimensions but degrades at 768D due to the curse of dimensionality.

### HNSW (Hierarchical Navigable Small World)
The production algorithm. Builds a multi-layer graph:

- **Upper layers** — sparse, long-range connections (like highways)
- **Lower layers** — denser, short-range connections (local streets)
- **Search** — enter at the top, greedily zoom toward the query, drop to the next layer, repeat

Maintains O(log N) performance even at 768 dimensions because navigation is graph-based, not geometry-based. This is what Pinecone, Weaviate, and Chroma use internally.

---

## RAG Pipeline

**Ingestion:**
```
Document → split into 250-word chunks → embed each chunk (nomic-embed-text) → store in HNSW index
```

**Query:**
```
Question → embed question → HNSW finds top 3 chunks → chunks passed as context to llama3.2 → answer
```

The LLM only answers from the retrieved context — no hallucination on your own documents.

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/insert` | Insert a demo vector |
| `GET` | `/search` | K-NN search with algo + metric params |
| `GET` | `/benchmark` | Run all 3 algos and compare latency |
| `GET` | `/hnsw-info` | Full HNSW graph structure (layers, edges) |
| `DELETE` | `/delete` | Remove a vector by ID |
| `POST` | `/doc/insert` | Embed + store a text document |
| `POST` | `/doc/ask` | Full RAG: question → retrieve → LLM answer |

Example search request:
```
GET /search?v=0.1,0.4,0.9,...&k=5&metric=cosine&algo=hnsw
```

---

## Getting Started

### Prerequisites

- C++17 compiler (GCC / Clang / MSVC)
- [Ollama](https://ollama.com) installed and running locally
- Pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Build

```bash
g++ -std=c++17 -O2 -o vectordb main.cpp
```

### Run

```bash
./vectordb
```

Server starts at `http://localhost:8080`. Open `index.html` in your browser.

---

## Demo

### Benchmark — HNSW vs KD-Tree vs Brute Force
Insert vectors and hit `/benchmark` to compare all three algorithms. Output shows latency in microseconds.

### Document Q&A
1. Insert any text via the **Documents** tab in the UI (or POST to `/doc/insert`)
2. Ask a question in natural language
3. The system retrieves the most relevant chunks and generates a grounded answer via `llama3.2`

---

## Key C++ Concepts Used

- `std::priority_queue` — max-heap for HNSW beam search
- `std::unordered_map<int, Node>` — O(1) node lookup in the graph
- `std::mutex` + `std::lock_guard` — thread-safe concurrent access (RAII)
- `std::chrono::high_resolution_clock` — microsecond benchmarking
- `std::function<float(...)>` — strategy pattern for pluggable distance metrics
- Lambda captures on HTTP route handlers

---

## What I Learned

Building this forced a deep understanding of:

- Why KD-Trees fail at high dimensions (curse of dimensionality)
- How HNSW's multilayer graph achieves sub-linear search without geometry tricks
- The full RAG pipeline — from chunking and embedding to context-grounded generation
- Thread safety in a multi-client HTTP server
- The difference between cosine, Euclidean, and Manhattan distance for semantic similarity tasks

---

## References

- [Efficient and robust approximate nearest neighbor search using HNSW (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320)
- [Ollama](https://ollama.com)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)

---

## License

MIT

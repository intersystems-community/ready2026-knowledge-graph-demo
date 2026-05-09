# Mining Knowledge from Support Tickets — READY 2026 Demo

> **From READY 2026 Tech Exchange #42:** *AI-Powered Support: Mining Knowledge from Ticket Data*

This repository contains the full demo from TE #42 — how semantic clustering turns a support ticket backlog into KB articles, and how IRIS GraphRAG enables hybrid vector + graph retrieval at scale.

Uses **PlanetCare**, a fictional EMR system, so all data is safe to share and adapt.

---

## What's Here

| Notebook | What it shows | Requires IRIS? |
|----------|--------------|----------------|
| `planetcare_clustering_demo.ipynb` | HDBSCAN semantic clustering → KB article generation | No |
| `planetcare_system_demo.ipynb` | IRIS vector search + knowledge graph + KBAC agents | Yes |

The clustering notebook is the one that generated the most discussion at READY — it shows how 295 questionnaire tickets in 5 languages cluster automatically into 2 groups, and how the canonical resolution pattern becomes a KB article.

---

## Quick Start

### Clustering notebook (no IRIS needed)

```bash
git clone https://github.com/intersystems-community/ready2026-knowledge-graph-demo
cd ready2026-knowledge-graph-demo
pip install -r requirements.txt
jupyter notebook notebooks/planetcare_clustering_demo.ipynb
```

### Full system demo (with IRIS)

**Step 1 — Start IRIS Community**
```bash
cd docker
docker compose up -d
# Wait ~60 seconds for IRIS to start
```

**Step 2 — Initialize schema and load data**
```bash
export OPENAI_API_KEY=sk-...
python setup/setup_iris.py
```

**Step 3 — Open notebooks**
```bash
jupyter notebook notebooks/
```

---

## What You'll See

### Clustering Demo

1. **The problem**: 295 questionnaire tickets, 5 languages, 20+ hospitals — zero discoverability across language boundaries
2. **HDBSCAN clustering**: embed with `all-MiniLM-L6-v2`, cluster by density — 2 groups emerge automatically
3. **Why it works**: Spanish, French, and English tickets about the same issue land in the same cluster because the model encodes *intent*, not language
4. **Resolution pattern**: 247 tickets → canonical 5-step fix, extracted without labeling
5. **KB article**: GPT-4o-mini writes a structured article from the cluster — one that didn't exist before the pipeline ran

### System Demo

1. **IRIS VECTOR_COSINE**: query with ada-002, search 181 tickets, get 0.83+ similarity
2. **Graph walk**: from any ticket, traverse `AFFECTS`, `EXHIBITS`, `HAS_SYMPTOM`, `FIXED_BY` edges
3. **Entity expansion**: vector search finds 9 tickets → graph expansion finds 115+ that share the same module/error
4. **KBAC enforcement**: role-gated tool calls, 12 real denials, every allow/deny stored as a permanent audit event
5. **PHI routing**: 26% of tickets contain PHI → routed to local model, never sent to cloud LLM
6. **KB synthesis**: anchor a known fix to similar unresolved tickets, generate a KB article

---

## Data

All data is synthetic or anonymized:

| File | Description |
|------|-------------|
| `data/planetcare_tickets.json` | 183 synthetic PlanetCare EMR tickets (GPT-4o-mini generated) |
| `data/planetcare/questionnaire_clusters_anon.csv` | 295 anonymized questionnaire tickets with cluster labels |
| `data/planetcare/questionnaire_kb_articles_anon.json` | 3 generated KB articles from the questionnaire cluster |

No real patient data. No real hospital names. Ticket IDs replaced with `PC-XXXXX` and `PC-QXXXX`.

---

## Architecture

```
Support tickets (any EMR)
    ↓ sentence-transformers (all-MiniLM-L6-v2)
HDBSCAN clustering  ─────────────────────────────→  KB Article
    ↓ OpenAI ada-002                                  (GPT-4o-mini)
IRIS VECTOR_COSINE search                              ↑
    ↓ iris-vector-graph                           Resolution
Graph_KG entity walk  (AFFECTS / EXHIBITS / FIXED_BY)  pattern
    ↓
Entity expansion (find tickets sharing same module/error)
    ↓
KBAC enforcement (PHIReader / KGWriter roles)
    ↓ Presidio
PHI routing (local vs cloud LLM)
```

---

## Connection to iris-vector-graph

This demo uses [iris-vector-graph](https://github.com/intersystems/iris-vector-graph) for:
- `IRISGraphEngine` — graph node/edge creation
- `IRISGraphOperators` — `kg_GRAPH_WALK`, `kg_NEIGHBORHOOD_EXPANSION`
- IRIS `VECTOR_COSINE` via the native irispython driver

See the [iris-vector-graph examples](https://github.com/intersystems/iris-vector-graph/tree/main/examples) for more patterns.

---

## Requirements

- Python 3.10+
- OpenAI API key (for ada-002 embedding + GPT-4o-mini synthesis)
- Docker (for IRIS Community — only needed for the system demo notebook)

The clustering notebook runs entirely in Python without IRIS.

---

## Adapting to Your Data

The pipeline is EMR-agnostic. To run it on your own tickets:

1. Export your tickets to JSON with fields: `id`, `summary`, `description`, `classification`, `status`, `resolution`
2. Replace `data/planetcare_tickets.json` with your export
3. Run `setup/setup_iris.py` to ingest
4. Open the notebooks — everything else stays the same

---

## Presented at

**InterSystems READY 2026** — Tech Exchange #42  
*AI-Powered Support: Mining Knowledge from Ticket Data*  
Thomas Dyar, Sr. Manager AI Platform & Ecosystems, InterSystems

---

## License

MIT

# Mining Knowledge from Support Tickets — READY 2026 Demo

> **From READY 2026 Tech Exchange #42:** *AI-Powered Support: Mining Knowledge from Ticket Data*

This repository demonstrates how semantic clustering turns a support ticket backlog into KB articles, and how IRIS GraphRAG enables hybrid vector + graph retrieval at scale.

Uses **PlanetCare**, a fictional EMR system — all data is synthetic, safe to share and adapt.

---

## Quick Start

### Notebook 1 — Clustering (no IRIS, no API key needed)

```bash
git clone https://github.com/intersystems-community/ready2026-knowledge-graph-demo
cd ready2026-knowledge-graph-demo
pip install -r requirements.txt
jupyter notebook notebooks/planetcare_clustering_demo.ipynb
```

Runs entirely in Python. Uses `all-MiniLM-L6-v2` locally for embeddings.  
An OpenAI API key is only needed for KB article synthesis (optional — clustering works without it).

### Notebook 2 — Full System Demo (IRIS + iris-vector-graph)

**Step 1 — Copy and configure**
```bash
cp .env.example .env
# Edit .env — set IRIS_PORT if needed, add OPENAI_API_KEY or OPENROUTER_API_KEY
```

**Step 2 — Start IRIS Community**
```bash
cd docker && docker compose up -d
# Wait ~60s, then check: docker ps | grep planetcare-iris
```

**Step 3 — Initialize Graph_KG and load data**
```bash
python setup/setup_iris.py
# Loads 276 tickets as Graph_KG nodes with embeddings (IVG)
# Creates AFFECTS/EXHIBITS/FIXED_BY entity edges (requires LLM)
# No API key needed for embedding (local MiniLM by default)
```

**Step 4 — Open notebooks**
```bash
jupyter notebook notebooks/
```

---

## What You'll See

### Clustering Demo (`planetcare_clustering_demo.ipynb`)

1. **The data quality problem** — 276 PlanetCare tickets, 58% with no resolution text. The norm, not the exception.
2. **HDBSCAN clustering** — embed problem descriptions with `all-MiniLM-L6-v2`, clusters emerge automatically. No labels required.
3. **Why it works** — the model encodes *semantic intent*, not keywords. Similar problems group together across different modules and phrasings.
4. **Resolution-anchored discovery** — the 42% with solutions become anchors. Each anchor suggests fixes for similar unresolved tickets in its cluster.
5. **KB article generation** — LLM synthesizes a structured article from the cluster's anchor tickets.
6. **Wiki augmentation** — article appended to `data/planetcare_wiki/billing.md` (pre-existing doc with documented gaps). Provenance recorded in Graph_KG.

### System Demo (`planetcare_system_demo.ipynb`)

1. **`engine.kg_KNN_VEC(query_json, k=8)`** — K-nearest neighbors by embedding similarity
2. **`engine.kg_VECTOR_GRAPH_SEARCH(query_json)`** — hybrid: vector + graph expansion
3. **`ops.kg_GRAPH_WALK(node, depth=2)`** — traverse AFFECTS/EXHIBITS/FIXED_BY edges
4. **`ops.kg_NEIGHBORHOOD_EXPANSION(entities)`** — find tickets sharing the same modules/errors
5. **`engine.execute_cypher(query)`** — direct graph queries
6. **KB synthesis** — LLM writes from cluster, result goes to wiki + Graph_KG audit trail

---

## Repo Contents

```
notebooks/
  planetcare_clustering_demo.ipynb   ← Start here (no IRIS needed)
  planetcare_system_demo.ipynb       ← iris-vector-graph API demo

data/
  planetcare_demo_tickets.json       ← 276 synthetic PlanetCare tickets
  planetcare/
    questionnaire_clusters_anon.csv  ← 295 anonymized questionnaire tickets
    questionnaire_kb_articles_anon.json
  planetcare_wiki/
    billing.md                       ← Pre-existing KB, has documented gaps
    laboratory.md                    ← Pre-existing KB, has documented gaps
    pharmacy.md                      ← Mostly empty — gaps noted

setup/
  setup_iris.py    ← Initializes Graph_KG via IRISGraphEngine
  embedder.py      ← Local / OpenAI / OpenRouter / custom embedding abstraction

docker/
  docker-compose.yml   ← IRIS Community container

.env.example         ← All configuration options
requirements.txt
```

---

## Data

All data is synthetic — no real patient data, no real hospital names.

| File | Description |
|------|-------------|
| `data/planetcare_demo_tickets.json` | 276 synthetic PlanetCare tickets · 7 categories · 42% resolved / 58% open |
| `data/planetcare/questionnaire_clusters_anon.csv` | 295 anonymized questionnaire tickets with pre-computed cluster labels |
| `data/planetcare_wiki/*.md` | Pre-existing KB articles with documented knowledge gaps |

---

## Embedding and LLM Configuration

Defaults require **no API key**. Set providers in `.env` or as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_PROVIDER` | `local` | `local` / `openai` / `openrouter` / `custom` |
| `LLM_PROVIDER` | `openai` | `openai` / `openrouter` / `custom` |
| `OPENAI_API_KEY` | — | Required if using OpenAI for embedding or LLM |
| `OPENROUTER_API_KEY` | — | Required if using OpenRouter |

The clustering notebook works without any API key. KB article synthesis needs a language model.

See `.env.example` for all options including local Ollama setup.

---

## PlanetCare Wiki

`data/planetcare_wiki/` is the fictional PlanetCare KB.  
Pre-existing articles document what's known. Each has a "Knowledge Gaps" section.

Running the clustering notebook **augments** these articles with AI-generated sections — clearly marked with source ticket count, agent ID, and a review warning.

Provenance is recorded in Graph_KG:
```cypher
MATCH (agent)-[:AUTHORED_KB]->(kb:KBArticle)<-[:SOURCED_KB]-(ticket:PCTicket)
RETURN agent.agent_id, kb.article_id, kb.category, ticket.ticket_id
LIMIT 20
```

---

## iris-vector-graph

This demo is built on [iris-vector-graph](https://github.com/intersystems/iris-vector-graph):

```python
from iris_vector_graph import IRISGraphEngine
from iris_vector_graph.operators import IRISGraphOperators

engine = IRISGraphEngine(conn, embedding_dimension=384)
ops = IRISGraphOperators(conn)

# Vector search
results = engine.kg_KNN_VEC(query_json, k=8)

# Hybrid vector + graph
results = engine.kg_VECTOR_GRAPH_SEARCH(query_json, k=10, expansion_depth=1)

# Graph walk
walk = ops.kg_GRAPH_WALK("pc_ticket:PC-00001", max_depth=2)

# Neighborhood expansion
neighbors = ops.kg_NEIGHBORHOOD_EXPANSION(entity_nodes, expansion_depth=1)
```

---

## Adapting to Your Data

The pipeline works on any ticket system:

1. Export tickets to JSON: `[{ticket_id, Summary, Problem, Solution, Classification, Status}, ...]`
2. Replace `data/planetcare_demo_tickets.json`
3. Run `python setup/setup_iris.py` to ingest into Graph_KG
4. Open the notebooks — clustering and retrieval work on any EMR data

---

## Presented at

**InterSystems READY 2026** — Tech Exchange #42  
*AI-Powered Support: Mining Knowledge from Ticket Data*  
Thomas Dyar, Sr. Manager AI Platform & Ecosystems, InterSystems

---

## License

MIT

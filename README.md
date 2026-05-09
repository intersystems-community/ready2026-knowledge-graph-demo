# Mining Knowledge from Support Tickets — READY 2026 Demo

> **From READY 2026 Tech Exchange #42:** *AI-Powered Support: Mining Knowledge from Ticket Data*

This repository demonstrates how semantic clustering turns a support ticket backlog into KB articles, and how IRIS GraphRAG enables hybrid vector + graph retrieval at scale.

Uses **PlanetCare**, a fictional EMR system — all data is synthetic, safe to share and adapt.

---

## What's Here

| Notebook | What it shows |
|----------|--------------|
| `planetcare_clustering_demo.ipynb` | MDS gap → HDBSCAN clustering → resolution anchoring → KB article → wiki augmentation + Graph_KG audit |
| `planetcare_system_demo.ipynb` | iris-vector-graph API: `kg_KNN_VEC`, `kg_VECTOR_GRAPH_SEARCH`, `kg_GRAPH_WALK`, Cypher |

Both notebooks require IRIS Community and an OpenAI API key (or compatible alternative).

---

## Quick Start

**Prerequisites:**
- [IRIS Community](https://containers.intersystems.com) — free, runs in Docker
- OpenAI API key — for KB article synthesis and graph entity extraction
- Python 3.10+, Docker

```bash
git clone https://github.com/intersystems-community/ready2026-knowledge-graph-demo
cd ready2026-knowledge-graph-demo

# Start IRIS Community
cd docker && docker compose up -d && cd ..
# Wait ~60s for IRIS to start

# Set your API key
export OPENAI_API_KEY=sk-...

# Install dependencies and initialize IRIS + Graph_KG
pip install -r requirements.txt
python setup/setup_iris.py

# Open notebooks
jupyter notebook notebooks/
```

> **Using OpenRouter, Ollama, or another provider?**  
> Copy `.env.example` to `.env` and set `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`.  
> Any OpenAI-compatible endpoint works — see `.env.example` for all options.

---

## What You'll See

### Clustering Demo (`planetcare_clustering_demo.ipynb`)

1. **The data quality problem** — 276 PlanetCare tickets, 58% with no resolution text. The norm, not the exception.
2. **HDBSCAN clustering** — embed problem descriptions with `all-MiniLM-L6-v2`, clusters emerge automatically without any labeling.
3. **Why it works** — the model encodes *semantic intent*, not keywords. Similar problems group together across different phrasings.
4. **Resolution-anchored discovery** — the 42% with solutions become anchors. Each anchor suggests fixes for similar unresolved tickets in its cluster.
5. **KB article generation** — GPT-4o-mini synthesizes a structured article from the cluster's anchor tickets.
6. **Wiki augmentation** — article appended to `data/planetcare_wiki/billing.md` (pre-existing doc with documented gaps). AUTHORED_KB and SOURCED_KB edges recorded in Graph_KG.

### System Demo (`planetcare_system_demo.ipynb`)

Uses iris-vector-graph directly:

```python
engine.kg_KNN_VEC(query_json, k=8)              # K-nearest by embedding similarity
engine.kg_VECTOR_GRAPH_SEARCH(query_json)        # hybrid: vector + graph expansion
ops.kg_GRAPH_WALK("pc_ticket:PC-00001", depth=2) # traverse entity edges
ops.kg_NEIGHBORHOOD_EXPANSION(entity_nodes)      # find tickets sharing same entities
engine.execute_cypher("MATCH (t:PCTicket)...")   # direct Cypher queries
```

---

## Repo Contents

```
notebooks/
  planetcare_clustering_demo.ipynb   ← The main demo
  planetcare_system_demo.ipynb       ← iris-vector-graph API showcase

data/
  planetcare_demo_tickets.json       ← 276 synthetic PlanetCare tickets
  planetcare/
    questionnaire_clusters_anon.csv  ← 295 anonymized questionnaire tickets
    questionnaire_kb_articles_anon.json
  planetcare_wiki/
    billing.md                       ← Pre-existing KB, documented gaps
    laboratory.md                    ← Pre-existing KB, documented gaps
    pharmacy.md                      ← Mostly empty — gaps noted

setup/
  setup_iris.py    ← Initializes Graph_KG via IRISGraphEngine, loads tickets, builds embeddings
  embedder.py      ← Embedding provider abstraction (local / OpenAI / OpenRouter / custom)

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
| `data/planetcare/questionnaire_clusters_anon.csv` | 295 anonymized questionnaire tickets with cluster labels |
| `data/planetcare_wiki/*.md` | Pre-existing KB articles with documented knowledge gaps |

---

## PlanetCare Wiki

`data/planetcare_wiki/` is the fictional PlanetCare KB.  
Pre-existing articles document what's known, with explicit "Knowledge Gaps" sections.

Running the clustering notebook **augments** these articles — AI-generated sections are appended, clearly marked with source ticket count, agent ID, timestamp, and a human-review warning.

Provenance is recorded in Graph_KG:
```cypher
MATCH (agent)-[:AUTHORED_KB]->(kb:KBArticle)<-[:SOURCED_KB]-(ticket:PCTicket)
RETURN agent.agent_id, kb.article_id, kb.category, ticket.ticket_id
LIMIT 20
```

---

## iris-vector-graph

This demo is built on [iris-vector-graph](https://github.com/intersystems/iris-vector-graph).  
See the [iris-vector-graph examples](https://github.com/intersystems/iris-vector-graph/tree/main/examples) for more API patterns.

**What `setup_iris.py` does:**
- `engine.initialize_schema()` — sets up Graph_KG tables and indexes
- `engine.create_node()` — creates `pc_ticket:PC-#####` nodes
- `engine.store_embeddings()` — stores vectors in `kg_NodeEmbeddings` for `kg_KNN_VEC`
- `engine.create_edge()` — builds AFFECTS/EXHIBITS/FIXED_BY/SOURCED_KB/AUTHORED_KB edges

---

## Adapting to Your Data

The pipeline works on any ticket system:

1. Export tickets to JSON: `[{ticket_id, Summary, Problem, Solution, Classification, Status}, ...]`
2. Replace `data/planetcare_demo_tickets.json`
3. Run `python setup/setup_iris.py`
4. Open the notebooks — clustering and retrieval work on any EMR data

---

## Presented at

**InterSystems READY 2026** — Tech Exchange #42  
*AI-Powered Support: Mining Knowledge from Ticket Data*  
Thomas Dyar, Sr. Manager AI Platform & Ecosystems, InterSystems

---

## License

MIT

#!/usr/bin/env python3
"""
setup_iris.py — Load PlanetCare demo data into IRIS using iris-vector-graph.

Run once after IRIS starts:
    python setup/setup_iris.py

Embedding provider (default: local MiniLM, no API key needed):
    EMBED_PROVIDER=local      # sentence-transformers all-MiniLM-L6-v2 (default)
    EMBED_PROVIDER=openai     # requires OPENAI_API_KEY
    EMBED_PROVIDER=openrouter # requires OPENROUTER_API_KEY
    (see .env.example for all options)

What this does:
    1. Initializes Graph_KG schema via IRISGraphEngine.initialize_schema()
    2. Creates pc_ticket:PC-##### nodes in Graph_KG
    3. Stores 384-dim embeddings in kg_NodeEmbeddings (IVG's vector store)
    4. Creates AFFECTS / EXHIBITS / FIXED_BY edges using GPT entity extraction
    5. Loads planetcare_demo_tickets into PC.Tickets for SQL queries in notebooks
"""
import os, sys, json, time, re
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup.embedder import get_embedder, get_llm_client

IRIS_HOST = os.getenv("IRIS_HOST", "localhost")
IRIS_PORT = int(os.getenv("IRIS_PORT", "11983"))
IRIS_USER = os.getenv("IRIS_USERNAME", "_SYSTEM")
IRIS_PASS = os.getenv("IRIS_PASSWORD", "SYS")

sys.stdout.write(str("PlanetCare Demo — IRIS Setup")
sys.stdout.write(str("="*50)
sys.stdout.write(str(f"IRIS: {IRIS_HOST}:{IRIS_PORT}")

import iris
try:
    conn = iris.connect(IRIS_HOST, IRIS_PORT, "USER", IRIS_USER, IRIS_PASS)
    print("IRIS connection: OK")
except Exception as e:
    print(f"IRIS connection failed: {e}")
    print("Make sure IRIS is running: cd docker && docker compose up -d")
    sys.exit(1)

from iris_vector_graph import IRISGraphEngine
from iris_vector_graph.operators import IRISGraphOperators

embedder = get_embedder()
llm_client, llm_model = get_llm_client()

engine = IRISGraphEngine(conn, embedding_dimension=embedder.dim)
cur = conn.cursor()

sys.stdout.write(str(f"\nEmbedder: {embedder.name} ({embedder.dim}-dim)")
sys.stdout.write(str(f"LLM: {llm_model or 'not configured — entity extraction will be skipped'}")

sys.stdout.write(str("\nInitializing Graph_KG schema...")
engine.initialize_schema()
sys.stdout.write(str("  Schema ready")

tickets_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "planetcare_demo_tickets.json"
)
tickets = json.load(open(tickets_path))
sys.stdout.write(str(f"\nLoaded {len(tickets)} PlanetCare tickets from {os.path.basename(tickets_path)}")

sys.stdout.write(str("\nCreating Graph_KG nodes...")
nodes_created = 0
for t in tickets:
    tid = t.get("ticket_id", "")
    node_id = f"pc_ticket:{tid}"
    try:
        engine.create_node(node_id, labels=["PCTicket", "Ticket"], properties={
            "ticket_id": tid,
            "text": f"{t.get('Summary','')} {t.get('Problem','')[:200]}",
            "category": t.get("Classification", ""),
            "hospital": t.get("Hospital", ""),
            "status": t.get("Status", "Open"),
            "has_solution": str(t.get("has_solution", False)),
            "system": "PlanetCare",
        })
        nodes_created += 1
    except Exception:
        pass
conn.commit()
sys.stdout.write(str(f"  {nodes_created} ticket nodes created in Graph_KG")

sys.stdout.write(str(f"\nStoring embeddings in kg_NodeEmbeddings ({embedder.dim}-dim)...")
BATCH = 32
stored = 0
for i in range(0, len(tickets), BATCH):
    batch = tickets[i:i+BATCH]
    texts = [f"{t.get('Summary','')} {t.get('Problem','')[:400]}" for t in batch]
    vectors = embedder.embed(texts)
    items = []
    for t, vec in zip(batch, vectors):
        items.append({
            "node_id": f"pc_ticket:{t['ticket_id']}",
            "embedding": vec,
            "metadata": {
                "category": t.get("Classification", ""),
                "status": t.get("Status", "Open"),
                "has_solution": t.get("has_solution", False),
            }
        })
    engine.store_embeddings(items)
    stored += len(items)
    print(f"  {stored}/{len(tickets)}", end="\r", flush=True)
sys.stdout.write(str(f"\n  Stored {stored} embeddings")

if llm_client:
    print("\nExtracting entities and building graph edges (GPT)...")
    edges_created = 0
    for i in range(0, len(tickets), 10):
        batch = tickets[i:i+10]
        ticket_texts = [
            f"{t['ticket_id']}: {t.get('Summary','')} | {t.get('Problem','')[:200]}"
            for t in batch
        ]
        prompt = (
            "For each PlanetCare ticket, extract: modules (PC-Finance, PC-Lab, PC-Pharmacy, "
            "PC-HL7Gateway, PC-PrintService, PC-Forms, PC-WaitList, PC-Orders etc), "
            "errors (specific error messages), symptoms (what user reported). "
            "Return JSON array: [{ticket_id, modules:[], errors:[], symptoms:[], root_cause:str}]"
        )
        try:
            resp = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": f"{prompt}\n\nTickets:\n" + "\n".join(ticket_texts)}],
                response_format={"type": "json_object"},
                max_tokens=2000, temperature=0.2,
            )
            data = json.loads(resp.choices[0].message.content)
            ents = data if isinstance(data, list) else (data.get("tickets") or list(data.values())[0] if data else [])

            for ent in ents:
                tid = ent.get("ticket_id", "")
                ticket_node = f"pc_ticket:{tid}"
                for mod in (ent.get("modules") or [])[:3]:
                    if mod:
                        mod_id = f"PC_MODULE:{mod.upper()[:30]}"
                        engine.create_node(mod_id, labels=["PCModule"], properties={"text": mod, "system": "PlanetCare"})
                        engine.create_edge(ticket_node, "AFFECTS", mod_id)
                        edges_created += 1
                for err in (ent.get("errors") or [])[:2]:
                    if err and len(str(err)) > 5:
                        err_id = f"PC_ERROR:{str(err)[:40].upper().replace(' ','_')}"
                        engine.create_node(err_id, labels=["PCError"], properties={"text": str(err)[:100], "system": "PlanetCare"})
                        engine.create_edge(ticket_node, "EXHIBITS", err_id)
                        edges_created += 1
                rc = ent.get("root_cause", "")
                if rc and len(rc) > 10:
                    t_orig = next((t for t in batch if t.get("ticket_id") == tid), {})
                    if t_orig.get("has_solution"):
                        rc_id = f"PC_RES:{tid}"
                        engine.create_node(rc_id, labels=["PCResolution"], properties={"text": str(rc)[:300], "system": "PlanetCare"})
                        engine.create_edge(ticket_node, "FIXED_BY", rc_id)
                        edges_created += 1
            conn.commit()
        except Exception as e:
            pass
        print(f"  Batch {i//10+1}/{len(tickets)//10+1}: {edges_created} edges", end="\r", flush=True)
    print(f"\n  {edges_created} graph edges created (AFFECTS/EXHIBITS/FIXED_BY)")
else:
    print("\nSkipping entity extraction (no LLM configured).")
    print("Graph walk will still work — ticket nodes and embeddings are in place.")
    print("Set OPENAI_API_KEY or OPENROUTER_API_KEY and re-run for entity edges.")

sys.stdout.write(str("\nCreating PC.Tickets SQL table for notebook queries...")
for ddl in [
    "DROP TABLE IF EXISTS PC.Tickets",
    """CREATE TABLE PC.Tickets (
        ticket_id VARCHAR(20) PRIMARY KEY,
        summary VARCHAR(500),
        problem VARCHAR(3000),
        solution VARCHAR(2000),
        classification VARCHAR(100),
        hospital VARCHAR(200),
        status VARCHAR(20),
        has_solution TINYINT
    )"""
]:
    try:
        cur.execute(ddl)
        conn.commit()
    except Exception as e:
        if "already" not in str(e).lower():
            print(f"  DDL note: {e}")

inserted = 0
for t in tickets:
    try:
        cur.execute(
            "INSERT INTO PC.Tickets (ticket_id,summary,problem,solution,classification,hospital,status,has_solution) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [t.get("ticket_id",""), t.get("Summary","")[:500],
             t.get("Problem","")[:3000], t.get("Solution","")[:2000],
             t.get("Classification",""), t.get("Hospital",""),
             t.get("Status","Open"), 1 if t.get("has_solution") else 0]
        )
        inserted += 1
    except Exception as e:
        if "UNIQUE" not in str(e) and "-119" not in str(e):
            pass
conn.commit()
sys.stdout.write(str(f"  {inserted} tickets in PC.Tickets")

stats = engine.graph_stats()
sys.stdout.write(str(f"""
Setup complete!
  Graph_KG nodes:      {stats.get('node_count', 0):,}
  Embeddings:          {stats.get('embedding_count', 0):,}  ({embedder.dim}-dim)
  Graph edges:         {stats.get('edge_count', 0):,}
  PC.Tickets (SQL):    {inserted:,}

Try in the notebook:
  engine.kg_KNN_VEC(query_json, k=8)          # vector search
  engine.kg_VECTOR_GRAPH_SEARCH(query_json)    # hybrid search
  ops.kg_GRAPH_WALK('pc_ticket:PC-00001')      # graph traversal
  engine.execute_cypher("MATCH (t:PCTicket)...")

Next steps:
  jupyter notebook notebooks/
  → planetcare_clustering_demo.ipynb  (no IRIS needed)
  → planetcare_system_demo.ipynb      (uses IVG on port {IRIS_PORT})
""")

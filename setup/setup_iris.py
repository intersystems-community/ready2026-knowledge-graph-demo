#!/usr/bin/env python3
"""
setup_iris.py — Initialize IRIS schema and load PlanetCare demo data.

Run once after IRIS starts:
    python setup/setup_iris.py

Embedding provider (set via env vars, see .env.example):
    EMBED_PROVIDER=local      # default — sentence-transformers, no API key
    EMBED_PROVIDER=openai     # requires OPENAI_API_KEY
    EMBED_PROVIDER=openrouter # requires OPENROUTER_API_KEY
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup.embedder import get_embedder, get_llm_client

IRIS_HOST = os.getenv("IRIS_HOST", "localhost")
IRIS_PORT = int(os.getenv("IRIS_PORT", "11983"))
IRIS_USER = os.getenv("IRIS_USERNAME", "_SYSTEM")
IRIS_PASS = os.getenv("IRIS_PASSWORD", "SYS")

print("PlanetCare Demo Setup")
print("="*50)
print(f"IRIS: {IRIS_HOST}:{IRIS_PORT}")

import iris
from openai import OpenAI

try:
    conn = iris.connect(IRIS_HOST, IRIS_PORT, "USER", IRIS_USER, IRIS_PASS)
    cur = conn.cursor()
    print("IRIS connection: OK")
except Exception as e:
    print(f"IRIS connection failed: {e}")
    print("Make sure IRIS is running: cd docker && docker compose up -d")
    sys.exit(1)

embedder = get_embedder()
llm_client, llm_model = get_llm_client()

# Create tables
print("\nCreating schema...")
for ddl in [
    "DROP TABLE IF EXISTS PC.TicketVectors",
    "DROP TABLE IF EXISTS PC.Tickets",
    """CREATE TABLE PC.Tickets (
        ticket_id VARCHAR(20) PRIMARY KEY,
        summary VARCHAR(500),
        description VARCHAR(5000),
        classification VARCHAR(100),
        hospital VARCHAR(200),
        status VARCHAR(20),
        resolution VARCHAR(2000),
        priority VARCHAR(5),
        pc_module VARCHAR(100),
        version VARCHAR(50)
    )""",
    f"""CREATE TABLE PC.TicketVectors (
        id VARCHAR(20) PRIMARY KEY,
        embedding VECTOR(FLOAT, {embedder.dim}),
        document VARCHAR(3000),
        m_classification VARCHAR(100),
        m_status VARCHAR(20)
    )""",
]:
    try:
        cur.execute(ddl)
        conn.commit()
    except Exception as e:
        if "already" not in str(e).lower():
            print(f"  DDL note: {e}")

print("  Tables created")

# Load tickets
tickets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "planetcare_tickets.json")
tickets = json.load(open(tickets_path))

def normalize_priority(p):
    p = str(p or "").strip().upper()
    if p in ("P1", "CRITICAL", "URGENT", "HIGH"): return "P1"
    if p in ("P3", "LOW", "MINOR"): return "P3"
    return "P2"

inserted = 0
for t in tickets:
    try:
        cur.execute(
            "INSERT INTO PC.Tickets (ticket_id,summary,description,classification,hospital,status,resolution,priority,pc_module,version) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [t.get("ticket_id",""), t.get("summary","")[:500], t.get("description","")[:5000],
             t.get("classification",""), t.get("hospital",""), t.get("status","Open"),
             t.get("resolution","")[:2000], normalize_priority(t.get("priority","")),
             t.get("module","PC-Core"), t.get("version","PlanetCare 4.2")]
        )
        inserted += 1
    except Exception as e:
        if "UNIQUE" not in str(e) and "-119" not in str(e):
            pass
conn.commit()
print(f"  Loaded {inserted} tickets into PC.Tickets")

# Embed
if embedder:
    print(f"\nEmbedding tickets with {embedder.name} (dim={embedder.dim})...")
    cur.execute("SELECT ticket_id, summary, description, classification, status FROM PC.Tickets")
    rows = cur.fetchall()
    embedded = 0
    for i in range(0, len(rows), 20):
        batch = rows[i:i+20]
        texts = [f"{r[1]} {(r[2] or '')[:400]}" for r in batch]
        vectors = embedder.embed(texts)
        for (tid, s, d, cat, status), vec_vals in zip(batch, vectors):
            vec = ",".join(str(round(x, 6)) for x in vec_vals)
            try:
                cur.execute(
                    "INSERT INTO PC.TicketVectors (id,embedding,document,m_classification,m_status) VALUES (?,TO_VECTOR(?),?,?,?)",
                    [tid, vec, f"{s} {(d or '')[:300]}"[:3000], cat, status]
                )
                embedded += 1
            except: pass
        conn.commit()
        print(f"  {embedded}/{len(rows)}", end="\r", flush=True)
    print(f"\n  Embedded {embedded} tickets ({embedder.dim}-dim)")
else:
    print("\nSkipping embeddings. Vector search will not work.")
    print("Run with EMBED_PROVIDER=local (default) or set an API key.")

# KBAC roles for demo agents
print("\nSetting up demo agent roles...")
ts = time.strftime("%Y-%m-%dT%H:%M:%S")
roles = [
    ("user:pc.discovery.agent",   "KGWriter"),
    ("user:pc.phi.router.agent",  "PHIReader"),
    ("user:pc.phi.router.agent",  "KGReader"),
    ("user:pc.coordinator.agent", "PHIReader"),
    ("user:pc.coordinator.agent", "KGReader"),
    ("user:pc.mds.checker.agent", "KGReader"),
    ("user:pc.mds.checker.agent", "KGWriter"),
]
try:
    cur.execute("""CREATE TABLE IF NOT EXISTS PC.Roles (
        role_id VARCHAR(100) PRIMARY KEY,
        agent_id VARCHAR(100),
        role_name VARCHAR(50),
        granted_at VARCHAR(30)
    )""")
    for agent, role in roles:
        rid = f"{agent}:{role}"
        try:
            cur.execute("INSERT INTO PC.Roles (role_id,agent_id,role_name,granted_at) VALUES (?,?,?,?)",
                [rid, agent, role, ts])
        except: pass
    conn.commit()
    print(f"  {len(roles)} role assignments created")
except Exception as e:
    print(f"  Roles note: {e}")

cur.execute("SELECT COUNT(*) FROM PC.Tickets"); t = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM PC.TicketVectors"); v = cur.fetchone()[0]

print(f"""
Setup complete!
  PC.Tickets:       {t:,}
  PC.TicketVectors: {v:,}

Next steps:
  jupyter notebook notebooks/
  → Open planetcare_clustering_demo.ipynb  (no IRIS required)
  → Open planetcare_system_demo.ipynb      (uses IRIS on port {IRIS_PORT})
""")

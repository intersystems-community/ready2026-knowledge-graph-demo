#!/usr/bin/env python3
"""
setup_iris.py — Initialize IRIS schema and load PlanetCare demo data.

Run once after IRIS starts:
    python setup/setup_iris.py

Then open the notebooks.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IRIS_HOST = os.getenv("IRIS_HOST", "localhost")
IRIS_PORT = int(os.getenv("IRIS_PORT", "11983"))
IRIS_USER = os.getenv("IRIS_USERNAME", "_SYSTEM")
IRIS_PASS = os.getenv("IRIS_PASSWORD", "SYS")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

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

if not OPENAI_KEY:
    print("WARNING: OPENAI_API_KEY not set — vector embedding will be skipped")
    print("Set it with: export OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

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
    """CREATE TABLE PC.TicketVectors (
        id VARCHAR(20) PRIMARY KEY,
        embedding VECTOR(FLOAT, 1536),
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
if client:
    print("\nEmbedding tickets with ada-002 (this takes ~2 minutes)...")
    cur.execute("SELECT ticket_id, summary, description, classification, status FROM PC.Tickets")
    rows = cur.fetchall()
    embedded = 0
    for i in range(0, len(rows), 20):
        batch = rows[i:i+20]
        texts = [f"{r[1]} {(r[2] or '')[:400]}" for r in batch]
        resp = client.embeddings.create(model="text-embedding-ada-002", input=texts)
        for (tid, s, d, cat, status), e in zip(batch, resp.data):
            vec = ",".join(str(round(x, 6)) for x in e.embedding)
            try:
                cur.execute(
                    "INSERT INTO PC.TicketVectors (id,embedding,document,m_classification,m_status) VALUES (?,TO_VECTOR(?),?,?,?)",
                    [tid, vec, f"{s} {(d or '')[:300]}"[:3000], cat, status]
                )
                embedded += 1
            except: pass
        conn.commit()
        print(f"  {embedded}/{len(rows)}", end="\r", flush=True)
    print(f"\n  Embedded {embedded} tickets")
else:
    print("\nSkipping embeddings (no API key). Vector search will not work.")
    print("Set OPENAI_API_KEY and re-run to enable.")

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

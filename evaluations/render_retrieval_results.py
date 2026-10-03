import json
from pathlib import Path
from html import escape


current_dir = Path(__file__).parent
input_path = current_dir / "results" / "dense_baseline.json"
output_path = current_dir / "results" / "dense_baseline_dashboard.html"


with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)


metadata = data["metadata"]
queries = data["queries"]


def esc(value):
    return escape(str(value), quote=True)


def build_result_card(result):
    rank = result["rank"]
    point_id = result["point_id"]
    score = result["score"]
    document = result["document"]
    chunk_text = result["text"]

    return f"""
    <article class="result-card">
        <div class="result-head">
            <span class="rank">Rank {rank}</span>
            <span class="score">Score {score:.4f}</span>
        </div>

        <div class="meta-row">
            <span class="meta-label">Document</span>
            <span>{esc(document)}</span>
        </div>

        <div class="meta-row">
            <span class="meta-label">Point ID</span>
            <span class="mono">{esc(point_id)}</span>
        </div>

        <details {"open" if rank <= 3 else ""}>
            <summary>Chunk text</summary>
            <div class="chunk">{esc(chunk_text)}</div>
        </details>
    </article>
    """


def build_query_panel(query, active=False):
    query_id = query["query_id"]
    topic = query.get("topic", "")
    query_text = query["query"]
    results = query["results"]

    active_class = " active" if active else ""

    top1 = results[0]["score"] if results else 0.0
    avg_score = sum(result["score"] for result in results) / len(results) if results else 0.0
    min_score = min((result["score"] for result in results), default=0.0)
    max_score = max((result["score"] for result in results), default=0.0)

    result_cards = "".join(build_result_card(result) for result in results)

    return f"""
    <section class="query-panel{active_class}" id="{esc(query_id)}">
        <div class="query-header">
            <div>
                <div class="eyebrow">{esc(query_id)} · {esc(topic)}</div>
                <h2>{esc(query_text)}</h2>
            </div>

            <div class="score-stats">
                <div><span>Top-1</span><strong>{top1:.4f}</strong></div>
                <div><span>Average</span><strong>{avg_score:.4f}</strong></div>
                <div><span>Range</span><strong>{min_score:.4f}–{max_score:.4f}</strong></div>
            </div>
        </div>

        <div class="metrics-box">
            <div class="metrics-title">Retrieval metrics</div>

            <div class="metric-grid">
                <div class="metric pending"><span>Precision@5</span><strong>Pending</strong></div>
                <div class="metric pending"><span>MRR</span><strong>Pending</strong></div>
                <div class="metric pending"><span>NDCG@5</span><strong>Pending</strong></div>
                <div class="metric pending"><span>NDCG@10</span><strong>Pending</strong></div>
            </div>

            <p>
                These metrics will be populated after relevance judgments are added.
                The score statistics above are embedding similarity scores, not retrieval-quality metrics.
            </p>
        </div>

        <div class="results-titlebar">
            <h3>Retrieved chunks</h3>

            <div>
                <button class="utility" onclick="toggleAll('{esc(query_id)}', true)">
                    Expand all
                </button>

                <button class="utility" onclick="toggleAll('{esc(query_id)}', false)">
                    Collapse all
                </button>
            </div>
        </div>

        <div class="results-list">
            {result_cards}
        </div>
    </section>
    """


nav_items = []
query_panels = []

for index, query in enumerate(queries):
    query_id = query["query_id"]
    topic = query.get("topic", "")
    active_class = " active" if index == 0 else ""

    nav_items.append(
        f"""
        <button class="query-tab{active_class}" data-query="{esc(query_id)}">
            <span class="qid">{esc(query_id)}</span>
            <span class="topic">{esc(topic)}</span>
        </button>
        """
    )

    query_panels.append(
        build_query_panel(
            query=query,
            active=(index == 0)
        )
    )


total_results = sum(len(query["results"]) for query in queries)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Retrieval Evaluation Dashboard</title>

<style>
:root {{
  --bg:#f4f7fb;
  --panel:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --border:#dfe6ee;
  --accent:#2563eb;
  --accent-soft:#eff6ff;
  --good:#047857;
  --warn:#9a5b00;
  --radius:16px;
  --shadow:0 10px 28px rgba(15,23,42,.06);
}}

* {{
  box-sizing:border-box;
}}

body {{
  margin:0;
  font-family:Inter,Segoe UI,Arial,sans-serif;
  background:var(--bg);
  color:var(--text);
}}

button {{
  font:inherit;
}}

.app {{
  min-height:100vh;
  display:grid;
  grid-template-columns:270px minmax(0,1fr);
}}

.sidebar {{
  position:sticky;
  top:0;
  height:100vh;
  padding:24px 18px;
  background:#0f172a;
  color:white;
  overflow:auto;
}}

.brand h1 {{
  margin:0 0 6px;
  font-size:1.15rem;
}}

.brand p {{
  margin:0;
  color:#94a3b8;
  font-size:.86rem;
  line-height:1.4;
}}

.side-label {{
  margin:28px 8px 10px;
  color:#94a3b8;
  font-size:.72rem;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.1em;
}}

.query-tab {{
  width:100%;
  display:block;
  text-align:left;
  margin:0 0 8px;
  padding:11px 12px;
  border:1px solid transparent;
  border-radius:11px;
  background:transparent;
  color:#cbd5e1;
  cursor:pointer;
}}

.query-tab:hover {{
  background:rgba(255,255,255,.06);
  color:white;
}}

.query-tab.active {{
  background:rgba(37,99,235,.22);
  border-color:rgba(96,165,250,.45);
  color:white;
}}

.qid {{
  display:block;
  font-weight:800;
}}

.topic {{
  display:block;
  margin-top:3px;
  color:#94a3b8;
  font-size:.76rem;
  overflow-wrap:anywhere;
}}

.main {{
  min-width:0;
  padding:30px;
}}

.overview {{
  background:linear-gradient(135deg,#fff,#f7fbff);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:24px;
  box-shadow:var(--shadow);
  margin-bottom:20px;
}}

.overview h2 {{
  margin:0 0 6px;
  font-size:1.65rem;
}}

.overview p {{
  margin:0;
  color:var(--muted);
}}

.summary-grid {{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:12px;
  margin-top:20px;
}}

.summary-card {{
  background:white;
  border:1px solid var(--border);
  border-radius:12px;
  padding:14px;
}}

.summary-card span {{
  display:block;
  color:var(--muted);
  font-size:.75rem;
  margin-bottom:6px;
}}

.summary-card strong {{
  font-size:1rem;
  overflow-wrap:anywhere;
}}

.collection {{
  margin-top:12px;
}}

.query-panel {{
  display:none;
}}

.query-panel.active {{
  display:block;
}}

.query-header {{
  display:grid;
  grid-template-columns:minmax(0,1.5fr) minmax(270px,.65fr);
  gap:20px;
  padding:22px;
  background:white;
  border:1px solid var(--border);
  border-radius:var(--radius);
  box-shadow:var(--shadow);
}}

.eyebrow {{
  color:var(--accent);
  font-size:.76rem;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.08em;
  margin-bottom:8px;
}}

.query-header h2 {{
  margin:0;
  line-height:1.35;
  font-size:1.35rem;
}}

.score-stats {{
  display:grid;
  gap:8px;
}}

.score-stats div {{
  display:flex;
  justify-content:space-between;
  gap:12px;
  padding:9px 11px;
  border:1px solid var(--border);
  border-radius:9px;
  background:#f8fafc;
}}

.score-stats span {{
  color:var(--muted);
  font-size:.78rem;
}}

.score-stats strong {{
  font-size:.84rem;
}}

.metrics-box {{
  margin-top:16px;
  background:white;
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:18px;
  box-shadow:var(--shadow);
}}

.metrics-title {{
  font-weight:800;
  margin-bottom:12px;
}}

.metric-grid {{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px;
}}

.metric {{
  padding:12px;
  border:1px solid var(--border);
  background:#f8fafc;
  border-radius:10px;
}}

.metric span {{
  display:block;
  color:var(--muted);
  font-size:.74rem;
  margin-bottom:6px;
}}

.metric strong {{
  font-size:.95rem;
}}

.metric.pending strong {{
  color:var(--warn);
}}

.metrics-box p {{
  color:var(--muted);
  font-size:.82rem;
  margin:10px 0 0;
}}

.results-titlebar {{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:12px;
  margin:22px 0 10px;
}}

.results-titlebar h3 {{
  margin:0;
}}

.utility {{
  padding:7px 10px;
  border:1px solid var(--border);
  background:white;
  border-radius:8px;
  cursor:pointer;
}}

.utility:hover {{
  background:#f8fafc;
}}

.results-list {{
  display:grid;
  gap:11px;
}}

.result-card {{
  background:white;
  border:1px solid var(--border);
  border-radius:13px;
  padding:16px;
  box-shadow:0 4px 14px rgba(15,23,42,.04);
}}

.result-head {{
  display:flex;
  justify-content:space-between;
  gap:10px;
  margin-bottom:12px;
}}

.rank,
.score {{
  padding:6px 9px;
  border-radius:999px;
  font-size:.78rem;
  font-weight:800;
}}

.rank {{
  background:var(--accent-soft);
  color:var(--accent);
}}

.score {{
  background:#ecfdf5;
  color:var(--good);
}}

.meta-row {{
  display:grid;
  grid-template-columns:86px minmax(0,1fr);
  gap:10px;
  margin:7px 0;
  font-size:.84rem;
}}

.meta-label {{
  color:var(--muted);
  font-weight:700;
}}

.mono {{
  font-family:Consolas,monospace;
  color:#475569;
  overflow-wrap:anywhere;
}}

details {{
  margin-top:12px;
  border-top:1px solid var(--border);
  padding-top:11px;
}}

summary {{
  cursor:pointer;
  color:var(--accent);
  font-weight:700;
  font-size:.85rem;
}}

.chunk {{
  margin-top:11px;
  padding:13px;
  background:#f8fafc;
  border-radius:9px;
  white-space:pre-wrap;
  line-height:1.62;
  color:#334155;
  font-size:.9rem;
  overflow-wrap:anywhere;
}}

@media (max-width:950px) {{
  .app {{
    grid-template-columns:1fr;
  }}

  .sidebar {{
    position:static;
    height:auto;
  }}

  .query-header {{
    grid-template-columns:1fr;
  }}

  .summary-grid,
  .metric-grid {{
    grid-template-columns:repeat(2,minmax(0,1fr));
  }}
}}

@media (max-width:600px) {{
  .main {{
    padding:18px;
  }}

  .summary-grid,
  .metric-grid {{
    grid-template-columns:1fr;
  }}

  .results-titlebar {{
    align-items:flex-start;
    flex-direction:column;
  }}

  .meta-row {{
    grid-template-columns:1fr;
    gap:3px;
  }}
}}
</style>
</head>

<body>
<div class="app">

  <aside class="sidebar">
    <div class="brand">
      <h1>Retrieval Evaluation</h1>
      <p>Dense retrieval baseline dashboard</p>
    </div>

    <div class="side-label">Queries</div>

    {"".join(nav_items)}
  </aside>

  <main class="main">

    <section class="overview">
      <h2>Retrieval Evaluation Dashboard</h2>
      <p>Readable inspection of the dense retrieval baseline.</p>

      <div class="summary-grid">
        <div class="summary-card">
          <span>Queries</span>
          <strong>{len(queries)}</strong>
        </div>

        <div class="summary-card">
          <span>Total chunks</span>
          <strong>{total_results}</strong>
        </div>

        <div class="summary-card">
          <span>Top K</span>
          <strong>{esc(metadata.get("top_k",""))}</strong>
        </div>

        <div class="summary-card">
          <span>Embedding model</span>
          <strong>{esc(metadata.get("embedding_model",""))}</strong>
        </div>
      </div>

      <div class="summary-card collection">
        <span>Collection</span>
        <strong>{esc(metadata.get("collection_name",""))}</strong>
      </div>
    </section>

    {"".join(query_panels)}

  </main>
</div>

<script>
const tabs = document.querySelectorAll(".query-tab");
const panels = document.querySelectorAll(".query-panel");

tabs.forEach(tab => {{
  tab.addEventListener("click", () => {{
    const target = tab.dataset.query;

    tabs.forEach(item => {{
      item.classList.toggle("active", item === tab);
    }});

    panels.forEach(panel => {{
      panel.classList.toggle("active", panel.id === target);
    }});

    window.scrollTo({{
      top:0,
      behavior:"smooth"
    }});
  }});
}});

function toggleAll(queryId, openState) {{
  const panel = document.getElementById(queryId);

  if (!panel) {{
    return;
  }}

  panel.querySelectorAll("details").forEach(details => {{
    details.open = openState;
  }});
}}
</script>
</body>
</html>
"""


with open(output_path, "w", encoding="utf-8") as file:
    file.write(html)


print(f"Dashboard saved to: {output_path}")

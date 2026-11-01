"""Generate paired HTML and Markdown reports for the RAG_V2 dense retrieval baseline.

Place this file at: evaluations/generate_report.py
Run from project root: python -m evaluations.generate_report

The default mode reads the FULL V2 judgement batch and imports your existing
metric functions; it does not change metrics.py. For previewing the supplied
summary-only report, use: python -m evaluations.generate_report --snapshot
"""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

SOURCE_PATH = Path("evaluations/results/judge_proposals_gpt-oss-20b_v2_batch.json")
REPORT_DIR = Path("evaluations/reports")
REPORT_STEM = "dense_baseline_v2_2026-11-01"
DATE_LABEL = "1 November 2026"
TOPICS = {
    "q1": "Proposition-level vs passage-level retrieval",
    "q2": "DPO vs PPO-based RLHF",
    "q3": "ReAct reasoning and acting",
    "q4": "HELM and multidimensional evaluation",
    "q5": "Advanced vs naive RAG",
}
ORDER = list(TOPICS)

# Summary figures reported from the user's completed 100-label V2 run.
# Used ONLY to build the supplied preview when --snapshot is specified.
SNAPSHOT_METRICS = {
    "Mean P@5": 0.8000,
    "Mean P@10": 0.8400,
    "Mean strict P@5": 0.4000,
    "MRR@20": 0.7125,
    "Mean NDCG@5": 0.5799,
    "Mean NDCG@10": 0.6397,
}
SNAPSHOT_TOP5 = {
    "q1": [1, 2, 1, 1, 2],
    "q2": [2, 2, 2, 1, 2],
    "q3": [0, 1, 0, 1, 1],
    "q4": [2, 2, 1, 2, 0],
    "q5": [2, 0, 1, 1, 0],
}


def build_data(snapshot=False, source_path=SOURCE_PATH):
    if snapshot:
        return {
            "mode": "snapshot",
            "judge_model": "gpt-oss:20b (V2 rubric)",
            "metrics": SNAPSHOT_METRICS,
            "queries": [
                {"id": q, "topic": TOPICS[q], "grades": SNAPSHOT_TOP5[q], "records": []}
                for q in ORDER
            ],
        }

    # Import only in full mode, so the supplied preview can be generated
    # without having your local RAG_V2 project files here.
    from evaluations.metrics import (
        load_grades_by_query,
        mean_ndcg_at_k,
        mean_precision_at_k,
        mean_reciprocal_rank,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
    )

    grades_by_query = load_grades_by_query(str(source_path))
    if set(grades_by_query) != set(ORDER):
        raise ValueError("Expected exactly q1, q2, q3, q4 and q5")

    with source_path.open("r", encoding="utf-8") as f:
        batch = json.load(f)

    records_by_query = {q: [] for q in ORDER}
    for record in batch["results"]:
        records_by_query[record["query_id"]].append(record)

    queries = []
    for q in ORDER:
        grades = grades_by_query[q]
        records = sorted(records_by_query[q], key=lambda row: row["rank"])
        if len(records) != 20:
            raise ValueError(f"Expected 20 judged chunks for {q}")
        first_answer_rank = next((i for i, grade in enumerate(grades, 1) if grade == 2), None)
        queries.append({
            "id": q,
            "topic": TOPICS[q],
            "grades": grades,
            "records": records,
            "p5": precision_at_k(grades, 5),
            "p10": precision_at_k(grades, 10),
            "strict_p5": precision_at_k(grades, 5, 2),
            "rr20": reciprocal_rank(grades, 2),
            "first_answer_rank": first_answer_rank,
            "ndcg5": ndcg_at_k(grades, 5),
            "ndcg10": ndcg_at_k(grades, 10),
        })

    return {
        "mode": "full",
        "judge_model": batch.get("metadata", {}).get("judge_model", "gpt-oss:20b"),
        "metrics": {
            "Mean P@5": mean_precision_at_k(grades_by_query, 5),
            "Mean P@10": mean_precision_at_k(grades_by_query, 10),
            "Mean strict P@5": mean_precision_at_k(grades_by_query, 5, 2),
            "MRR@20": mean_reciprocal_rank(grades_by_query),
            "Mean NDCG@5": mean_ndcg_at_k(grades_by_query, 5),
            "Mean NDCG@10": mean_ndcg_at_k(grades_by_query, 10),
        },
        "queries": queries,
    }


def grade_cell(grade):
    return f'<span class="grade grade-{grade}" title="Grade {grade}">{grade}</span>'


def fmt(value):
    return f"{value:.4f}"


def build_html(data):
    metrics = data["metrics"]
    queries = data["queries"]
    full = data["mode"] == "full"
    mode_caption = (
        "FULL BATCH · 100 JUDGEMENTS · ALL 20 RANKS PER QUERY"
        if full else "SUMMARY PREVIEW · REPORTED METRICS + FIRST FIVE GRADES"
    )
    cards = []
    for label, value in metrics.items():
        label_safe = escape(label)
        pct = max(0, min(100, value * 100))
        cards.append(f'''
          <div class="metric"><div class="metric-label">{label_safe}</div>
          <div class="metric-value">{fmt(value)}</div>
          <div class="track"><div class="fill" style="width:{pct:.2f}%"></div></div>
          <div class="metric-foot">{pct:.1f}% of maximum</div></div>''')

    top5_rows = []
    for item in queries:
        grades = item["grades"][:5]
        p5 = sum(g >= 1 for g in grades) / 5
        strict_p5 = sum(g == 2 for g in grades) / 5
        top5_rows.append(
            "<tr><th scope='row'><b>{}</b><small>{}</small></th>{}<td class='num'>{:.2f}</td>"
            "<td class='num'>{:.2f}</td></tr>".format(
                escape(item["id"]), escape(item["topic"]),
                "".join(f"<td class='grade-td'>{grade_cell(g)}</td>" for g in grades),
                p5, strict_p5,
            )
        )

    per_query = ""
    evidence = ""
    if full:
        rows = []
        for item in queries:
            rank = str(item["first_answer_rank"]) if item["first_answer_rank"] else "—"
            rows.append(
                f'<tr><td><b>{escape(item["id"])}</b></td><td class="num">{fmt(item["p5"])}</td>'
                f'<td class="num">{fmt(item["p10"])}</td><td class="num">{fmt(item["strict_p5"])}</td>'
                f'<td class="num">{rank}</td><td class="num">{fmt(item["rr20"])}</td>'
                f'<td class="num">{fmt(item["ndcg5"])}</td><td class="num">{fmt(item["ndcg10"])}</td></tr>'
            )
        per_query = '''<section id="per-query"><div class="section-heading"><h2>Query-level diagnostics</h2>
          <p>Per-query scores, rather than aggregates alone, make ranking failures easier to locate.</p></div>
          <div class="table-wrap"><table><thead><tr><th>Query</th><th>P@5</th><th>P@10</th><th>Strict P@5</th>
          <th>First grade-2 rank</th><th>RR@20</th><th>NDCG@5</th><th>NDCG@10</th></tr></thead><tbody>'''
        per_query += "".join(rows) + "</tbody></table></div></section>"

        detail_blocks = []
        for item in queries:
            evidence_rows = []
            for rec in item["records"]:
                rank = rec["rank"]
                grade = rec["grade"]
                reason = escape(str(rec.get("reason", "")))
                chunk = escape(str(rec.get("text", "")))
                pid = escape(str(rec.get("point_id", "")))
                evidence_rows.append(f'''
                  <tr><td class="num">{rank}</td><td>{grade_cell(grade)}</td>
                  <td><details class="evidence"><summary>{escape(str(rec.get('text',''))[:140]) or 'Open chunk text'}</summary>
                  <div class="chunk">{chunk}</div><div class="pid">Point ID: {pid}</div></details></td>
                  <td class="reason">{reason}</td></tr>''')
            detail_blocks.append(f'''
              <details class="query-block"><summary><b>{escape(item['id'].upper())}</b> · {escape(item['topic'])}
              <span class="summary-right">20 judged chunks</span></summary>
              <div class="table-wrap"><table class="evidence-table"><thead><tr><th>Rank</th><th>Grade</th>
              <th>Retrieved chunk</th><th>LLM judge reasoning</th></tr></thead>
              <tbody>{''.join(evidence_rows)}</tbody></table></div></details>''')
        evidence = ('''<section id="evidence"><div class="section-heading"><h2>Evidence inspection</h2>
          <p>Expand each query to inspect all 20 ranks, chunk text and the judge's stated reason. These are proposed LLM labels, not adjudicated human ground truth.</p></div>'''
          + "".join(detail_blocks) + "</section>")

    snapshot_note = "" if full else '''<div class="notice"><b>Preview scope.</b> The six aggregate values and the first five ranks of each query were available from the completed run. The full V2 JSON file was not available when this preview was authored; per-query NDCG@k and rank-by-rank evidence are deliberately omitted rather than invented. Run the included generator on your PC to populate them automatically.</div>'''

    all_top5 = [g for q in queries for g in q["grades"][:5]]
    counts = Counter(all_top5)
    total = len(all_top5)
    dist = "".join(
        f'<div class="distribution-row"><span>Grade {g} <em>{desc}</em></span><div class="track"><div class="fill dist-{g}" style="width:{counts[g]*100/total:.0f}%"></div></div><strong>{counts[g]} / {total}</strong></div>'
        for g, desc in [(2, "direct evidence"), (1, "supporting"), (0, "not useful")]
    )

    style = r'''
    :root{color-scheme:light;--ink:#183040;--muted:#617582;--teal:#087f73;--navy:#132f43;--bg:#f4f7f8;--line:#dce6e9;--panel:#fff;--amber:#bd7c22;--red:#a7524b;}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Arial,sans-serif;line-height:1.55}
    .wrap{max-width:1160px;margin:0 auto;padding:0 30px 72px}.hero{background:var(--navy);color:#fff;padding:50px 0 54px;position:relative;overflow:hidden}
    .hero:before{content:"";position:absolute;width:480px;height:480px;border:1px solid #ffffff22;border-radius:50%;right:-110px;top:-300px;box-shadow:0 0 0 85px #ffffff06,0 0 0 170px #ffffff05}
    .hero .wrap{padding-bottom:0;position:relative}.eyebrow{font-size:12px;text-transform:uppercase;font-weight:800;letter-spacing:.18em;color:#87dcd1}
    h1{font-size:clamp(30px,5vw,52px);letter-spacing:-.047em;line-height:1.08;margin:17px 0 15px;font-weight:780;max-width:800px}
    .lead{font-size:17px;color:#d3e4e9;max-width:760px}.meta{display:flex;flex-wrap:wrap;gap:12px;margin-top:29px}
    .pill{border:1px solid #ffffff42;border-radius:30px;padding:6px 13px;font-size:12px;color:#dfedf0}.muted-pill{color:#98dcd6}
    .quick-nav{padding:16px 0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:22px;font-size:13px;font-weight:650}
    a{color:#076e6c;text-decoration:none}a:hover{text-decoration:underline}.quick-nav a{color:#385663}
    section{margin-top:42px}.section-heading{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:18px;flex-wrap:wrap}
    h2{font-size:22px;letter-spacing:-.022em;margin:0}h3{font-size:16px;margin:0 0 11px}.section-heading p{color:var(--muted);margin:0;font-size:14px;max-width:540px}
    .metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
    .metric-label{font-size:13px;color:#4c6572;font-weight:650}.metric-value{font-variant-numeric:tabular-nums;letter-spacing:-.035em;font-size:31px;font-weight:790;margin:8px 0 10px}
    .metric-foot{font-size:11px;color:#84959c;margin-top:9px}.track{height:8px;background:#e8eef0;border-radius:5px;overflow:hidden;flex:1}.fill{height:100%;background:var(--teal);border-radius:5px}
    .notice{background:#fff9e8;border-left:3px solid #caa14f;border-radius:5px;padding:13px 17px;margin-top:22px;font-size:13px;color:#68532e}
    .two-col{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:18px}.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px}
    .takeaways{padding-left:19px;margin:4px 0}.takeaways li{padding:4px 0}.takeaways strong{color:#113d49}
    .distribution-row{display:flex;gap:12px;align-items:center;margin:18px 0;font-size:13px}.distribution-row>span{width:170px;flex-shrink:0;font-weight:700}.distribution-row em{display:block;font-weight:400;color:var(--muted);font-size:11px;font-style:normal}
    .distribution-row strong{width:57px;text-align:right;white-space:nowrap;font-size:12px}.dist-1{background:#d09b4b}.dist-0{background:#9bafba}
    .table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{border-bottom:1px solid var(--line);padding:12px 14px;text-align:left;vertical-align:middle}thead th{background:#eaf1f3;color:#425d6a;font-weight:750;white-space:nowrap}
    tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}tbody th{width:32%;min-width:210px}tbody th b{display:block}tbody th small{display:block;color:var(--muted);font-size:11px;font-weight:400;margin-top:4px}
    td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.grade-td{text-align:center;width:68px}
    .grade{display:inline-grid;place-items:center;min-width:28px;height:29px;border-radius:7px;font-weight:750;font-variant-numeric:tabular-nums}
    .grade-0{color:#526671;background:#e7edf0}.grade-1{color:#8a5e1c;background:#f8ead2}.grade-2{color:#086a62;background:#d5f0e9}
    .legend{margin-top:11px;color:var(--muted);font-size:12px;display:flex;gap:16px;flex-wrap:wrap}
    .method{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.method .panel p{font-size:13px;color:#536976;margin:0}
    code{padding:1px 4px;background:#eaf0f3;border-radius:4px;font-size:12px}.query-block{background:#fff;border:1px solid var(--line);border-radius:11px;margin-bottom:10px}
    .query-block>summary{padding:15px 18px;cursor:pointer;list-style-position:inside;font-size:14px}.summary-right{float:right;color:var(--muted);font-size:12px}
    .query-block .table-wrap{border:0;border-top:1px solid var(--line);border-radius:0 0 11px 11px}.evidence-table td{vertical-align:top}
    .evidence-table th:nth-child(3){width:42%}.evidence-table th:nth-child(4){width:40%}.evidence-table .reason{font-size:12px;color:#405c6a;min-width:250px}
    .evidence>summary{cursor:pointer;color:#116b6b;font-size:12px;line-height:1.5}.chunk{white-space:pre-wrap;line-height:1.6;color:#304c5c;padding:10px 0;font-size:12px}
    .pid{font-family:ui-monospace,monospace;font-size:10px;color:#8599a3;overflow-wrap:anywhere}
    footer{font-size:12px;color:#647d88;margin-top:48px;border-top:1px solid var(--line);padding-top:20px}
    @media(max-width:760px){.wrap{padding:0 18px 48px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.two-col,.method{grid-template-columns:1fr}.hero{padding:39px 0 48px}.metric{padding:15px}.metric-value{font-size:25px}}
    @media(max-width:430px){.metrics{grid-template-columns:1fr}.distribution-row>span{width:120px}}
    @media print{.hero{background:#fff;color:#132f43;padding:20px 0}.lead,.eyebrow{color:#264a56}.pill{color:#294653;border-color:#c7d5da}.quick-nav{display:none}.wrap{max-width:100%;padding:0 14px}.metric,.panel{break-inside:avoid}body{background:#fff}a{color:inherit}}
    '''

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG_V2 — Dense Retrieval Evaluation (V2)</title><style>{style}</style></head><body>
<header class="hero"><div class="wrap"><div class="eyebrow">RAG_V2 / EVALUATION REPORT / V2</div>
<h1>Dense retrieval,<br>measured and inspected.</h1>
<p class="lead">Relevance, answer-bearing evidence, and ranking quality across the five-paper-query baseline.</p>
<div class="meta"><span class="pill">{escape(DATE_LABEL)}</span><span class="pill">5 queries · 20 chunks/query</span>
<span class="pill">Judge: {escape(data['judge_model'])}</span><span class="pill muted-pill">{escape(mode_caption)}</span></div></div></header>
<main class="wrap"><nav class="quick-nav" aria-label="Sections"><a href="#overview">Overview</a><a href="#ranking">Top-five inspection</a>
<a href="#interpretation">Interpretation</a>{'<a href="#per-query">Per-query metrics</a><a href="#evidence">Evidence</a>' if full else ''}
<a href="#method">Methodology</a></nav>
<section id="overview"><div class="section-heading"><h2>Baseline scorecard</h2><p>Macro-average across five queries. Higher is better; metrics answer different questions.</p></div>
<div class="metrics">{''.join(cards)}</div>{snapshot_note}</section>
<section id="interpretation" class="two-col"><div class="panel"><h3>What the numbers suggest</h3><ul class="takeaways">
<li><strong>Useful context is usually retrieved.</strong> Mean P@5 is <b>{fmt(metrics['Mean P@5'])}</b> using grades 1–2 as relevant.</li>
<li><strong>Direct evidence is harder to surface.</strong> Strict P@5 is <b>{fmt(metrics['Mean strict P@5'])}</b>, half the broad P@5 score.</li>
<li><strong>Ordering can improve.</strong> Mean NDCG@5 (<b>{fmt(metrics['Mean NDCG@5'])}</b>) is below NDCG@10 (<b>{fmt(metrics['Mean NDCG@10'])}</b>), consistent with stronger graded evidence sometimes appearing deeper.</li>
<li><strong>First answer-bearing result often appears early.</strong> MRR@20 is <b>{fmt(metrics['MRR@20'])}</b>; it is an average reciprocal rank, not a literal average rank.</li>
</ul><p style="font-size:12px;color:#617582;margin:12px 0 0">These are descriptive observations, not a validated comparison against a separate retrieval system.</p></div>
<div class="panel"><h3>Top-five relevance composition</h3><p style="color:#617582;font-size:13px;margin:0">Across 25 ranked chunks (5 queries × top 5)</p>{dist}
<p style="color:#617582;font-size:12px;margin-top:20px">Grade 2 = directly answer-bearing; grade 1 = supporting context; grade 0 = no useful evidence.</p></div></section>
<section id="ranking"><div class="section-heading"><h2>Rank-by-rank top-five inspection</h2>
<p>The grade pattern is more informative than a single mean. The leftmost grade is rank 1.</p></div>
<div class="table-wrap"><table><thead><tr><th>Question theme</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>Broad P@5</th><th>Strict P@5</th>
</tr></thead><tbody>{''.join(top5_rows)}</tbody></table></div>
<div class="legend"><span>{grade_cell(2)} Direct answer evidence</span><span>{grade_cell(1)} Supporting/context</span>
<span>{grade_cell(0)} No useful evidence</span></div></section>
{per_query}
{evidence}
<section id="method"><div class="section-heading"><h2>Methodology & limitations</h2>
<p>Interpret scores in the context of how relevance was assigned.</p></div><div class="method">
<div class="panel"><h3>Judgement rubric</h3><p>Grade <b>0</b>: no useful evidence. Grade <b>1</b>: supporting/background evidence. Grade <b>2</b>: directly answer-bearing evidence for at least one substantive part of the question.</p></div>
<div class="panel"><h3>Metric definitions</h3><p>P@k: fraction of top-k grades ≥ 1; strict P@5: fraction equal to 2. RR@20: reciprocal of the first grade-2 rank. NDCG@k uses gain <code>2^grade − 1</code> and <code>log₂(rank + 1)</code>, with ideal order computed from all 20 judged chunks.</p></div>
<div class="panel"><h3>Evidence constraints</h3><p>Only 5 queries. Judgements proposed by a local LLM, not adjudicated ground truth. Scores are exploratory, not statistical proof of retrieval superiority. No corpus-wide recall is reported.</p></div>
</div><div class="notice"><b>Next engineering step:</b> Freeze this baseline, construct context from retrieved chunks, and test grounded answer generation. Compare future retrieval/reranking changes against this same evaluation set.</div></section>
<footer>RAG_V2 · Dense baseline · V2 relevance rubric · {escape(DATE_LABEL)} · Generated locally from evaluation data where available. No external CSS or scripts required.</footer>
</main></body></html>'''


def build_markdown(data):
    metrics = data["metrics"]
    queries = data["queries"]
    full = data["mode"] == "full"
    lines = [
        "# RAG_V2 — Dense Retrieval Evaluation (V2)",
        "",
        f"**Date:** {DATE_LABEL}  ",
        "**Scope:** 5 queries × 20 retrieved chunks; 100 relevance judgements  ",
        f"**Judge:** {data['judge_model']}  ",
        "**Relevance labels:** LLM proposals, not adjudicated human ground truth  ",
        "**Data mode:** " + ("Full 100-judgement V2 batch" if full else "Summary preview: reported aggregates + top-five grades only"),
        "",
        "## Aggregate results",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {fmt(score)} |" for label, score in metrics.items())
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- **Breadth of evidence:** Mean P@5 is **{fmt(metrics['Mean P@5'])}**, using grades 1 and 2 as relevant.",
        f"- **Direct answer-bearing evidence:** Mean strict P@5 is **{fmt(metrics['Mean strict P@5'])}** (grade 2 only), substantially below broad P@5.",
        f"- **Ranking quality:** NDCG@5 is **{fmt(metrics['Mean NDCG@5'])}** and NDCG@10 is **{fmt(metrics['Mean NDCG@10'])}**. Strong evidence is not consistently at the earliest ranks.",
        f"- **First direct evidence:** MRR@20 is **{fmt(metrics['MRR@20'])}**. This is an average of reciprocal ranks, not the reciprocal of the average rank.",
        "",
        "These results describe five specific queries with provisional LLM-generated grades. They do not establish statistical significance or superiority over another retriever.",
        "",
        "## Top-five rank inspection",
        "",
        "| Query | Topic | R1 | R2 | R3 | R4 | R5 | Broad P@5 | Strict P@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in queries:
        grades = item["grades"][:5]
        broad = sum(g >= 1 for g in grades) / 5
        strict = sum(g == 2 for g in grades) / 5
        lines.append(f"| {item['id']} | {item['topic']} | " + " | ".join(str(g) for g in grades)
                     + f" | {broad:.2f} | {strict:.2f} |")
    lines += ["", "Grade 0: no useful evidence. Grade 1: background/supporting evidence. Grade 2: directly answer-bearing evidence.", ""]
    if full:
        lines += [
            "## Per-query metrics", "",
            "| Query | P@5 | P@10 | Strict P@5 | First grade-2 rank | RR@20 | NDCG@5 | NDCG@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for q in queries:
            rank = str(q["first_answer_rank"]) if q["first_answer_rank"] else "—"
            lines.append(f"| {q['id']} | {fmt(q['p5'])} | {fmt(q['p10'])} | {fmt(q['strict_p5'])} | "
                         f"{rank} | {fmt(q['rr20'])} | {fmt(q['ndcg5'])} | {fmt(q['ndcg10'])} |")
        lines += ["", "## Judged rank records", "", "For full chunk text and judge reasoning, expand each query in the HTML report.", ""]
        for q in queries:
            lines += [f"### {q['id']}: {q['topic']}", "", "| Rank | Grade | Judge reason (excerpt) |", "|---:|---:|---|"]
            for r in q["records"]:
                reason = " ".join(str(r.get("reason", "")).split())
                reason = reason.replace("|", "\\|")
                lines.append(f"| {r['rank']} | {r['grade']} | {reason[:180]}{'…' if len(reason)>180 else ''} |")
            lines.append("")
    else:
        lines += [
            "> **Preview limitation:** The complete V2 batch was not available when these files were generated. "
            "Per-query metrics beyond top-five grades and full judge evidence have not been fabricated. "
            "Run `python -m evaluations.generate_report` from the project root with the V2 batch file present "
            "to regenerate both reports with all 100 judgements.", "",
        ]
    lines += [
        "## Evaluation method", "",
        "- **Precision@k:** grades ≥ 1 are relevant; strict P@5 counts only grade 2.",
        "- **RR / MRR@20:** first grade-2 chunk per query, then arithmetic mean across the five queries.",
        "- **NDCG@k:** DCG discounts gain `(2**grade - 1)` by `log2(rank + 1)`, "
        "and IDCG sorts all 20 judged grades before truncating to k. Macro-average across queries.",
        "- **Integrity:** five expected query IDs and one judgement at every rank from 1 through 20 for each query.",
        "- **Limitations:** only five questions; judgements are LLM proposals rather than adjudicated human ground truth; "
        "no independent comparison system or full-corpus recall measurement.",
        "",
        "## Follow-on work", "",
        "1. Freeze the V2 retrieval baseline and judgement rubric.",
        "2. Assemble retrieved text into a context window with citations/metadata.",
        "3. Generate grounded answers and inspect whether supporting chunks actually answer the questions.",
        "4. Later compare dense retrieval against reranking and context expansion with the same query set.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="Generate the summary-only preview without JSON")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="Path to 100-judgement V2 JSON")
    args = parser.parse_args()
    data = build_data(snapshot=args.snapshot, source_path=args.source)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORT_DIR / f"{REPORT_STEM}.html"
    md_path = REPORT_DIR / f"{REPORT_STEM}.md"
    html_path.write_text(build_html(data), encoding="utf-8")
    md_path.write_text(build_markdown(data), encoding="utf-8")
    print(f"Wrote HTML report: {html_path}")
    print(f"Wrote Markdown report: {md_path}")
    print("Mode:", "Full V2 data" if data["mode"] == "full" else "Summary preview")


if __name__ == "__main__":
    main()

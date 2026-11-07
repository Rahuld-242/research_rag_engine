
"""Generate HTML and Markdown reports comparing dense retrieval and reranking.

Run from the RAG_V2 project root:

    python -m evaluations.generate_rerank_comparison

Uses the existing evaluations.metrics functions.
Does not run any LLM or cross-encoder inference.
"""

import json
from html import escape
from pathlib import Path

from evaluations.metrics import (
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


EVALUATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALUATION_DIR / "results"
REPORTS_DIR = EVALUATION_DIR / "reports"

METRICS = {
    "p5": "P@5 (grades 1–2)",
    "p10": "P@10 (grades 1–2)",
    "strict_p5": "Strict P@5 (grade 2)",
    "mrr": "MRR@20 (grade 2)",
    "ndcg5": "NDCG@5",
    "ndcg10": "NDCG@10",
}


def load_json(filename):
    with open(RESULTS_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


def query_metrics(grades):
    """Calculate metrics for one query."""
    return {
        "p5": precision_at_k(grades, 5, relevant_threshold=1),
        "p10": precision_at_k(grades, 10, relevant_threshold=1),
        "strict_p5": precision_at_k(grades, 5, relevant_threshold=2),
        "mrr": reciprocal_rank(grades, relevant_threshold=2),
        "ndcg5": ndcg_at_k(grades, 5),
        "ndcg10": ndcg_at_k(grades, 10),
    }


def average_metrics(grades_by_query):
    """Calculate mean metrics across queries."""
    return {
        "p5": mean_precision_at_k(
            grades_by_query, 5, relevant_threshold=1
        ),
        "p10": mean_precision_at_k(
            grades_by_query, 10, relevant_threshold=1
        ),
        "strict_p5": mean_precision_at_k(
            grades_by_query, 5, relevant_threshold=2
        ),
        "mrr": mean_reciprocal_rank(
            grades_by_query, relevant_threshold=2
        ),
        "ndcg5": mean_ndcg_at_k(grades_by_query, 5),
        "ndcg10": mean_ndcg_at_k(grades_by_query, 10),
    }


def first_direct_rank(grades):
    """Find the first rank containing direct answer evidence."""
    for rank, grade in enumerate(grades, start=1):
        if grade == 2:
            return rank

    return None


def build_report_data(baseline, reranked, judgements):
    """
    Combine the three JSON files.

    Relevance grades are matched using:
        (query_id, point_id)

    Never match grades using rank, because ranks change.
    """
    grades_by_id = {}

    for judgement in judgements["results"]:
        key = (judgement["query_id"], judgement["point_id"])

        if key in grades_by_id:
            raise ValueError(f"Duplicate judgement: {key}")

        if judgement["grade"] not in (0, 1, 2):
            raise ValueError(f"Unexpected grade: {key}")

        grades_by_id[key] = judgement

    baseline_ids = [
        query["query_id"]
        for query in baseline["queries"]
    ]

    if len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("Duplicate query ID in baseline")

    if set(baseline_ids) != set(reranked):
        raise ValueError("Baseline and reranked query IDs differ")

    queries = []
    old_grades_by_query = {}
    new_grades_by_query = {}
    used_labels = set()

    for query in baseline["queries"]:
        query_id = query["query_id"]

        old_rows = sorted(
            query["results"],
            key=lambda row: row["rank"]
        )

        new_rows = sorted(
            reranked[query_id],
            key=lambda row: row["reranked_rank"]
        )

        if len(old_rows) < 10 or len(old_rows) != len(new_rows):
            raise ValueError(
                f"{query_id}: expected matching sets of at least 10 chunks"
            )

        old_ranks = [row["rank"] for row in old_rows]
        new_ranks = [row["reranked_rank"] for row in new_rows]

        expected_ranks = list(range(1, len(old_rows) + 1))

        if old_ranks != expected_ranks:
            raise ValueError(f"{query_id}: invalid original ranks")

        if new_ranks != expected_ranks:
            raise ValueError(f"{query_id}: invalid new ranks")

        old_by_id = {
            row["point_id"]: row
            for row in old_rows
        }

        new_by_id = {
            row["point_id"]: row
            for row in new_rows
        }

        if len(old_by_id) != len(old_rows):
            raise ValueError(f"{query_id}: duplicate original point IDs")

        if len(new_by_id) != len(new_rows):
            raise ValueError(f"{query_id}: duplicate reranked point IDs")

        if set(old_by_id) != set(new_by_id):
            raise ValueError(
                f"{query_id}: reranking changed the candidate set"
            )

        detailed_rows = []

        for row in new_rows:
            point_id = row["point_id"]
            label_key = (query_id, point_id)

            if label_key not in grades_by_id:
                raise ValueError(
                    f"Missing judgement for {label_key}"
                )

            original = old_by_id[point_id]

            if row["rank"] != original["rank"]:
                raise ValueError(
                    f"{query_id}: original rank changed for {point_id}"
                )

            if row["text"] != original["text"]:
                raise ValueError(
                    f"{query_id}: original text changed for {point_id}"
                )

            if "reranker_score" not in row:
                raise ValueError(
                    f"{query_id}: missing reranker score for {point_id}"
                )

            judgement = grades_by_id[label_key]
            used_labels.add(label_key)

            detailed_rows.append({
                "old_rank": original["rank"],
                "new_rank": row["reranked_rank"],
                "movement": original["rank"] - row["reranked_rank"],
                "grade": judgement["grade"],
                "reason": judgement.get("reason", ""),
                "document": row["document"],
                "text": row["text"],
                "dense_score": original["score"],
                "reranker_score": row["reranker_score"],
                "point_id": point_id,
            })

        old_grades = []

        for row in old_rows:
            label_key = (query_id, row["point_id"])
            old_grades.append(grades_by_id[label_key]["grade"])

        new_grades = [
            row["grade"]
            for row in detailed_rows
        ]

        old_grades_by_query[query_id] = old_grades
        new_grades_by_query[query_id] = new_grades

        queries.append({
            "id": query_id,
            "topic": query.get("topic", ""),
            "question": query["query"],
            "old": query_metrics(old_grades),
            "new": query_metrics(new_grades),
            "first_old": first_direct_rank(old_grades),
            "first_new": first_direct_rank(new_grades),
            "rows": detailed_rows,
        })

    if used_labels != set(grades_by_id):
        raise ValueError(
            "Some judgements do not match a retrieved chunk"
        )

    return {
        "queries": queries,
        "overall_old": average_metrics(old_grades_by_query),
        "overall_new": average_metrics(new_grades_by_query),
        "total_chunks": sum(
            len(query["rows"])
            for query in queries
        ),
        "judge": judgements.get(
            "metadata", {}
        ).get("judge_model", "unknown"),
    }


def preview(text, max_length=155):
    """Shorten chunk text for table previews."""
    clean = " ".join(text.split())

    if len(clean) > max_length:
        clean = clean[:max_length].rstrip() + "…"

    return escape(clean)


def grade_badge(grade):
    names = {
        0: "0 · Irrelevant",
        1: "1 · Supporting",
        2: "2 · Direct evidence",
    }

    return (
        f'<span class="grade g{grade}">'
        f'{names[grade]}'
        f'</span>'
    )


CSS = """
:root {
    font-family: system-ui, Segoe UI, Arial, sans-serif;
    color: #203341;
    background: #f2f6f8;
}
* { box-sizing: border-box; }
body { margin: 0; line-height: 1.55; }
header {
    background: #173b4c;
    color: white;
    padding: 38px max(24px, calc((100vw - 1100px)/2));
}
header h1 { font-size: 32px; margin: 5px 0; }
header p { color: #c3dde6; max-width: 850px; }
main { max-width: 1150px; padding: 25px; margin: auto; }
section {
    background: white;
    border: 1px solid #dce5e9;
    border-radius: 13px;
    padding: 22px;
    margin-bottom: 20px;
}
h2, h3 { margin-top: 0; }
p.muted, .note { color: #667b87; font-size: 13px; }
.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 13px;
}
.stat {
    background: white;
    border: 1px solid #dce5e9;
    border-radius: 12px;
    padding: 17px;
}
.stat strong { display: block; font-size: 27px; }
.positive { color: #127b67; font-weight: bold; }
.negative { color: #bd4b4b; font-weight: bold; }
.neutral { color: #687c86; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
    text-align: left;
    background: #eff4f6;
    color: #415e6d;
    font-size: 11px;
    text-transform: uppercase;
}
th, td {
    border-bottom: 1px solid #e4ebee;
    padding: 10px 12px;
    vertical-align: top;
}
.num {
    text-align: right;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.grade {
    display: inline-block;
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 650;
    white-space: nowrap;
}
.g0 { background: #fbe6e6; color: #aa4047; }
.g1 { background: #fff0d5; color: #875c19; }
.g2 { background: #dcf4eb; color: #166f53; }
.bars {
    display: grid;
    grid-template-columns: 125px 1fr;
    gap: 11px 15px;
    align-items: center;
}
.track {
    height: 11px;
    background: #edf1f3;
    border-radius: 5px;
    overflow: hidden;
    margin: 4px 0;
}
.dense { height: 100%; background: #9cafb9; }
.reranked { height: 100%; background: #178775; }
.top5 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}
.panel {
    border: 1px solid #dde6ea;
    border-radius: 9px;
    overflow: hidden;
}
.panel h4 {
    padding: 9px 13px;
    background: #f1f6f7;
    margin: 0;
}
.entry {
    padding: 12px 13px;
    border-top: 1px solid #e5edf0;
    font-size: 13px;
}
.entry small {
    display: block;
    color: #647a86;
    margin-top: 5px;
}
details { margin-top: 15px; }
summary {
    cursor: pointer;
    color: #245f7a;
    font-weight: 650;
}
.fulltext {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 320px;
    overflow: auto;
    font-size: 12px;
}
.document {
    overflow-wrap: anywhere;
    font-size: 12px;
    font-weight: 650;
}
footer {
    text-align: center;
    color: #778993;
    font-size: 12px;
    padding: 15px;
}
@media(max-width: 720px) {
    .top5 { grid-template-columns: 1fr; }
    header h1 { font-size: 25px; }
}
"""


def render_metric_rows(old, new):
    rows = []

    for key, label in METRICS.items():
        difference = new[key] - old[key]

        if difference > 0:
            color = "positive"
        elif difference < 0:
            color = "negative"
        else:
            color = "neutral"

        rows.append(
            f'<tr><td>{label}</td>'
            f'<td class="num">{old[key]:.4f}</td>'
            f'<td class="num"><b>{new[key]:.4f}</b></td>'
            f'<td class="num {color}">{difference:+.4f}</td></tr>'
        )

    return "".join(rows)


def render_top_five(rows, use_new_rank):
    """Display the top five chunks from either ranking."""

    if use_new_rank:
        rank_field = "new_rank"
    else:
        rank_field = "old_rank"

    ordered = sorted(
        rows,
        key=lambda row: row[rank_field]
    )[:5]

    entries = []

    for row in ordered:
        entries.append(
            f'<div class="entry">'
            f'<b>#{row[rank_field]}</b> '
            f'{grade_badge(row["grade"])}'
            f'<small>{preview(row["text"], 190)}</small>'
            f'</div>'
        )

    return "".join(entries)


def render_detail_rows(rows):
    """Create the detailed chunk-by-chunk comparison table."""

    output = []

    for row in rows:
        movement = row["movement"]

        if movement > 0:
            direction = f"↑ {movement}"
        elif movement < 0:
            direction = f"↓ {abs(movement)}"
        else:
            direction = "—"

        output.append(f"""
        <tr>
          <td class="num">{row['new_rank']}</td>
          <td class="num">{row['old_rank']}</td>
          <td class="num">{direction}</td>
          <td>{grade_badge(row['grade'])}</td>
          <td class="num">{row['reranker_score']:.3f}</td>
          <td>
            <div class="document">{escape(row['document'])}</div>
            <div class="note">{preview(row['text'], 135)}</div>

            <details>
              <summary>Read chunk and judge's reason</summary>
              <p class="fulltext">{escape(row['text'])}</p>
              <p><b>Judge reason:</b> {escape(row['reason'])}</p>
              <p class="note">
                ID: {escape(str(row['point_id']))}
                · Dense score: {row['dense_score']:.4f}
              </p>
            </details>
          </td>
        </tr>
        """)

    return "".join(output)


def render_html(data):
    """Generate the complete standalone HTML report."""

    old = data["overall_old"]
    new = data["overall_new"]

    # Summary cards.
    cards = []

    for key in ("p5", "strict_p5", "ndcg5", "mrr"):
        difference = new[key] - old[key]

        if difference > 0:
            color = "positive"
        elif difference < 0:
            color = "negative"
        else:
            color = "neutral"

        cards.append(
            f'<div class="stat">'
            f'<span>{METRICS[key]}</span>'
            f'<strong>{new[key]:.4f}</strong>'
            f'<span class="{color}">'
            f'{difference:+.4f} vs dense'
            f'</span></div>'
        )

    # Visual bar comparison.
    chart_rows = []

    for key in ("p5", "strict_p5", "ndcg5", "ndcg10"):
        chart_rows.append(
            f'<span>{METRICS[key]}</span>'
            f'<div>'
            f'<div class="track" title="Dense {old[key]:.4f}">'
            f'<div class="dense" style="width:{old[key]*100:.2f}%"></div>'
            f'</div>'
            f'<div class="track" title="Reranked {new[key]:.4f}">'
            f'<div class="reranked" style="width:{new[key]*100:.2f}%"></div>'
            f'</div>'
            f'</div>'
        )

    query_summary = []
    sections = []

    for query in data["queries"]:
        qold = query["old"]
        qnew = query["new"]

        first_old = query["first_old"] or "—"
        first_new = query["first_new"] or "—"

        # One row in the per-query metrics table.
        query_summary.append(
            f'<tr>'
            f'<td><a href="#{escape(query["id"])}">'
            f'{escape(query["id"].upper())}</a></td>'
            f'<td>{qold["p5"]:.2f} → <b>{qnew["p5"]:.2f}</b></td>'
            f'<td>{qold["strict_p5"]:.2f} → '
            f'<b>{qnew["strict_p5"]:.2f}</b></td>'
            f'<td>{qold["ndcg5"]:.3f} → '
            f'<b>{qnew["ndcg5"]:.3f}</b></td>'
            f'<td class="num">{first_old} → {first_new}</td>'
            f'</tr>'
        )

        # Detailed section for this query.
        sections.append(f"""
        <section id="{escape(query['id'])}">
          <h2>
            {escape(query['id'].upper())}
            · {escape(query['topic'].replace('_', ' '))}
          </h2>

          <p>{escape(query['question'])}</p>

          <p class="note">
            First grade-2 chunk: rank {first_old} → {first_new}
          </p>

          <div class="top5">
            <div class="panel">
              <h4>Dense top 5</h4>
              {render_top_five(query['rows'], False)}
            </div>

            <div class="panel">
              <h4>Cross-encoder top 5</h4>
              {render_top_five(query['rows'], True)}
            </div>
          </div>

          <details>
            <summary>
              Show all {len(query['rows'])} chunk movements
              and judge explanations
            </summary>

            <p class="note">
              ↑ = moved closer to rank 1.
              Cross-encoder scores are not probabilities.
            </p>

            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th class="num">New</th>
                    <th class="num">Old</th>
                    <th>Move</th>
                    <th>Grade</th>
                    <th class="num">CE score</th>
                    <th>Source and evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {render_detail_rows(query['rows'])}
                </tbody>
              </table>
            </div>
          </details>
        </section>
        """)

    change_p5 = new["p5"] - old["p5"]
    change_ndcg = new["ndcg5"] - old["ndcg5"]

    if change_p5 > 0:
        conclusion = "improves"
    elif change_p5 < 0:
        conclusion = "reduces"
    else:
        conclusion = "does not change"

    # Assemble all sections into one HTML document.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG_V2 — Dense vs Cross-Encoder</title>
  <style>{CSS}</style>
</head>

<body>
<header>
  <h1>Dense retrieval vs cross-encoder reranking</h1>
  <p>
    Controlled experiment using the same candidate chunks,
    evaluated with provisional V2 LLM relevance judgements.
  </p>
  <p>
    {len(data['queries'])} queries ·
    {data['total_chunks']} judged chunks ·
    Judge: {escape(data['judge'])}
  </p>
</header>

<main>
  <div class="stats">
    {''.join(cards)}
  </div>

  <section>
    <h2>Experiment summary</h2>

    <p>
      Cross-encoder reranking <b>{conclusion}</b> Mean P@5 by
      {change_p5:+.4f} and changes Mean NDCG@5 by
      {change_ndcg:+.4f} in this five-query experiment.
    </p>

    <p class="muted">
      The reranker only reorders the dense retriever's existing candidates.
      It cannot recover documents absent from the initial top-20 pool.

      The grades are LLM proposals, not human-verified ground truth.

      This report does not measure latency or answer generation.
    </p>
  </section>

  <section>
    <h2>Overall comparison</h2>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th class="num">Dense</th>
            <th class="num">Reranked</th>
            <th class="num">Change</th>
          </tr>
        </thead>

        <tbody>
          {render_metric_rows(old, new)}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Visual metric comparison</h2>
    <p class="note">
      Grey = dense; teal = cross-encoder. Bar scale: 0–1.
    </p>

    <div class="bars">
      {''.join(chart_rows)}
    </div>
  </section>

  <section>
    <h2>Per-query results</h2>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Query</th>
            <th>P@5</th>
            <th>Strict P@5</th>
            <th>NDCG@5</th>
            <th class="num">First grade-2 rank</th>
          </tr>
        </thead>

        <tbody>
          {''.join(query_summary)}
        </tbody>
      </table>
    </div>
  </section>

  {''.join(sections)}

  <section>
    <h2>Evaluation notes</h2>

    <p>
      Grades:
      {grade_badge(0)}
      {grade_badge(1)}
      {grade_badge(2)}
    </p>

    <p>
      For P@k, grades 1 or 2 count as relevant.
      Strict P@5 and MRR require grade 2.
      NDCG uses grades 0, 1, and 2.

      Labels are matched to
      <code>(query_id, point_id)</code>,
      never to a mutable rank.

      All metrics use the existing
      <code>evaluations.metrics</code> functions.
    </p>
  </section>
</main>

<footer>
  RAG_V2 · Offline report · No external dependencies
</footer>
</body>
</html>
"""

    return html


def render_markdown(data):
    """Generate a companion Markdown report."""

    lines = [
        "# RAG_V2: Dense retrieval vs cross-encoder",
        "",
        "Provisional LLM relevance judgements; same candidate pool.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Dense | Reranked | Change |",
        "|---|---:|---:|---:|",
    ]

    for key, name in METRICS.items():
        old = data["overall_old"][key]
        new = data["overall_new"][key]

        lines.append(
            f"| {name} | {old:.4f} | {new:.4f} | "
            f"{new-old:+.4f} |"
        )

    lines.extend([
        "",
        "## Per-query metrics",
        "",
        "| Query | P@5 old → new | Strict P@5 old → new | NDCG@5 old → new |",
        "|---|---:|---:|---:|",
    ])

    for query in data["queries"]:
        old = query["old"]
        new = query["new"]

        lines.append(
            f'| {query["id"]} | '
            f'{old["p5"]:.2f} → {new["p5"]:.2f} | '
            f'{old["strict_p5"]:.2f} → {new["strict_p5"]:.2f} | '
            f'{old["ndcg5"]:.3f} → {new["ndcg5"]:.3f} |'
        )

    for query in data["queries"]:
        lines.extend([
            "",
            f'## {query["id"].upper()}: {query["question"]}',
            "",
            "| New | Old | Move (+ up) | Grade | CE score | Excerpt |",
            "|---:|---:|---:|---:|---:|---|",
        ])

        for row in query["rows"]:
            excerpt = " ".join(row["text"].split())[:100]
            excerpt = excerpt.replace("|", "\\|")

            lines.append(
                f'| {row["new_rank"]} | '
                f'{row["old_rank"]} | '
                f'{row["movement"]:+d} | '
                f'{row["grade"]} | '
                f'{row["reranker_score"]:.3f} | '
                f'{excerpt} |'
            )

    lines.extend([
        "",
        "## Limits",
        "",
        "- Only five queries; LLM judge labels are provisional.",
        "- No end-to-end answer quality or inference latency measurement.",
        "",
    ])

    return "\n".join(lines)


def main():
    # Load the three original experiment files.
    baseline = load_json("dense_baseline.json")

    reranked = load_json("cross_encoder_reranked.json")

    judgements = load_json(
        "judge_proposals_gpt-oss-20b_v2_batch.json"
    )

    # Validate the inputs and calculate the comparison.
    report_data = build_report_data(
        baseline,
        reranked,
        judgements
    )

    # Create the reports folder if it does not exist.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    html_path = REPORTS_DIR / "cross_encoder_comparison.html"
    markdown_path = REPORTS_DIR / "cross_encoder_comparison.md"

    # Generate both reports.
    html_path.write_text(
        render_html(report_data),
        encoding="utf-8"
    )

    markdown_path.write_text(
        render_markdown(report_data),
        encoding="utf-8"
    )

    # Print a short summary in the terminal.
    print(
        f'Validated {len(report_data["queries"])} queries, '
        f'{report_data["total_chunks"]} chunks'
    )

    for key in METRICS:
        old = report_data["overall_old"][key]
        new = report_data["overall_new"][key]

        print(
            f"{key:10s} {old:.4f} -> {new:.4f} "
            f"({new-old:+.4f})"
        )

    print(f"HTML saved to: {html_path}")
    print(f"Markdown saved to: {markdown_path}")


if __name__ == "__main__":
    main()

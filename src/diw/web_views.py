from __future__ import annotations

import html
import json
from pathlib import Path

from diw.db.models import AIRun, AISuggestion, SourceDocument

_TEMPLATE_DIR = Path(__file__).with_name("templates")


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


PUBLIC_DEMO_EXAMPLES = (
    "What does the corrected note say is preferred and why?",
    "What can cause retrieval quality to fail?",
    "What should the system do when the evidence does not support a question?",
)


def _dashboard_html(
    *,
    documents: list[SourceDocument],
    pending: list[AISuggestion],
    runs: list[AIRun],
    document_count: int,
    version_count: int,
    chunk_count: int,
    embedding_count: int,
) -> str:
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Dashboard · Document Intelligence Workspace</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; line-height: 1.45; }}
          h1 {{ margin-bottom: 8px; }}
          section {{ margin-top: 28px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
          code {{ background: #f4f4f4; padding: 2px 4px; }}
          .metric {{ display: inline-block; margin-right: 18px; }}
        </style>
      </head>
      <body>
        <h1>Document Intelligence Workspace</h1>
        <p>Inspectable source-cited QA, AI-run provenance, and human review.</p>
        <section>
          <strong class="metric">Documents: {document_count}</strong>
          <strong class="metric">Versions: {version_count}</strong>
          <strong class="metric">Chunks: {chunk_count}</strong>
          <strong class="metric">Embeddings: {embedding_count}</strong>
          <strong class="metric">Pending review: {len(pending)}</strong>
        </section>
        <section>
          <h2>Documents</h2>
          {_html_documents_table(documents)}
        </section>
        <section>
          <h2>Pending Review</h2>
          {_html_suggestions_table(pending)}
        </section>
        <section>
          <h2>Recent AI Runs</h2>
          {_html_runs_table(runs)}
        </section>
        <section>
          <h2>API</h2>
          <p>Use <code>/docs</code> for the interactive API console.</p>
        </section>
      </body>
    </html>
    """


def _public_page_styles() -> str:
    return """
      :root {
        color-scheme: light;
        --ink: #14211f;
        --muted: #5d6b67;
        --paper: #fbfaf5;
        --panel: #ffffff;
        --line: #d8ded9;
        --forest: #174c3c;
        --forest-2: #236b55;
        --mint: #dff3e8;
        --amber: #f2c14e;
        --shadow: 0 18px 55px rgba(23, 76, 60, .12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--ink);
        background:
          radial-gradient(circle at 88% 4%, rgba(242, 193, 78, .22), transparent 27rem),
          linear-gradient(180deg, #f3f7f2 0, var(--paper) 38rem);
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.55;
      }
      a { color: inherit; }
      nav {
        width: min(1120px, calc(100% - 40px));
        margin: 0 auto;
        padding: 22px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
      }
      .brand { font-weight: 760; letter-spacing: -.02em; text-decoration: none; }
      .nav-links { display: flex; gap: 18px; align-items: center; }
      .nav-links a { color: var(--muted); font-size: 14px; text-decoration: none; }
      .nav-links a:hover { color: var(--forest); }
      main { width: min(1120px, calc(100% - 40px)); margin: 0 auto; }
      .eyebrow {
        color: var(--forest-2);
        font-size: 13px;
        font-weight: 760;
        letter-spacing: .11em;
        text-transform: uppercase;
      }
      h1, h2, h3 { line-height: 1.08; letter-spacing: -.035em; }
      h1 { font-size: clamp(44px, 7vw, 78px); max-width: 920px; margin: 20px 0; }
      h2 { font-size: clamp(30px, 4vw, 46px); }
      h3 { font-size: 20px; }
      .lede { color: var(--muted); font-size: clamp(18px, 2.2vw, 23px); max-width: 760px; }
      .actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 30px; }
      .button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 46px;
        padding: 0 20px;
        border: 1px solid var(--forest);
        border-radius: 999px;
        color: var(--forest);
        background: transparent;
        font-weight: 700;
        text-decoration: none;
        cursor: pointer;
      }
      .button.primary { color: #fff; background: var(--forest); }
      .button:hover { transform: translateY(-1px); }
      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 11px;
        border: 1px solid #b8d7c8;
        border-radius: 999px;
        background: rgba(223, 243, 232, .72);
        color: var(--forest);
        font-size: 13px;
        font-weight: 700;
      }
      .status::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #28a36a;
        box-shadow: 0 0 0 4px rgba(40, 163, 106, .12);
      }
      .hero { padding: 68px 0 84px; }
      .metrics, .grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }
      .metric, .card {
        padding: 24px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255, 255, 255, .82);
        box-shadow: var(--shadow);
      }
      .metric strong { display: block; color: var(--forest); font-size: 38px; line-height: 1; }
      .metric span { color: var(--muted); font-size: 14px; }
      .section { padding: 42px 0 76px; }
      .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .card p, .note { color: var(--muted); }
      .card code, .trace code {
        padding: 2px 5px;
        border-radius: 5px;
        background: #edf1ee;
        font-size: .88em;
      }
      footer {
        width: min(1120px, calc(100% - 40px));
        margin: 0 auto;
        padding: 32px 0 50px;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 13px;
      }
      @media (max-width: 760px) {
        nav { align-items: flex-start; }
        .nav-links { flex-wrap: wrap; justify-content: flex-end; gap: 8px 14px; }
        .hero { padding-top: 42px; }
        .metrics, .grid { grid-template-columns: 1fr; }
      }
    """


def _public_landing_html() -> str:
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="description" content="A source-cited research assistant with protected write actions.">
        <title>Document Intelligence Workspace</title>
        <style>{_public_page_styles()}</style>
      </head>
      <body>
        <nav>
          <a class="brand" href="/">Document Intelligence Workspace</a>
          <div class="nav-links">
            <a href="/demo">Live demo</a>
            <a href="/evidence">Evidence</a>
            <a href="/signin">Google sign-in</a>
          </div>
        </nav>
        <main>
          <section class="hero">
            <span class="status">Live on Cloud Run</span>
            <p class="eyebrow">Evidence-first AI research assistant</p>
            <h1>Answers over ML papers, with citations you can inspect.</h1>
            <p class="lede">
              A research workspace that retrieves source evidence,
              refuses unsupported questions, and keeps write actions behind identity
              and approval boundaries.
            </p>
            <div class="actions">
              <a class="button primary" href="/demo">Try the read-only demo</a>
              <a class="button" href="/evidence">See what was measured</a>
              <a class="button" href="/signin">Verify Google OIDC</a>
            </div>
          </section>
          <section class="metrics" aria-label="Project facts">
            <div class="metric">
              <strong>140</strong>
              <span>balanced human-calibration questions; labels remain pending</span>
            </div>
            <div class="metric">
              <strong>0</strong>
              <span>write tools exposed by the public demo</span>
            </div>
          </section>
          <section class="section">
            <p class="eyebrow">Designed for scrutiny</p>
            <h2>One URL, three trust levels.</h2>
            <div class="grid">
              <article class="card">
                <h3>Public demo</h3>
                <p>No account required. The public page presents the product,
                deployment, and measured evidence without exposing a write surface.</p>
              </article>
              <article class="card">
                <h3>Evaluation</h3>
                <p>The read-only demo returns exact quotes, citations, retrieval
                scores, and a compact execution trace.</p>
              </article>
              <article class="card">
                <h3>Technical reviewer</h3>
                <p>Google OIDC is real. Tenant-scoped agent routes require a bearer
                token; unscoped data routes fail closed in production mode.</p>
              </article>
            </div>
          </section>
        </main>
        <footer>
          Public demo data is synthetic. The 140-question calibration instrument is
          ready; independent human labels and agreement remain pending.
        </footer>
      </body>
    </html>
    """


def _public_demo_html() -> str:
    examples = "".join(
        f'<button class="example" type="button" data-question="{html.escape(question)}">'
        f"{html.escape(question)}</button>"
        for question in PUBLIC_DEMO_EXAMPLES
    )
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Read-only demo · Document Intelligence Workspace</title>
        <style>
          {_public_page_styles()}
          .demo-shell {{ padding: 52px 0 80px; }}
          .demo-shell h1 {{ font-size: clamp(38px, 5vw, 62px); max-width: 820px; }}
          .composer {{
            margin-top: 32px; padding: 18px; border: 1px solid var(--line);
            border-radius: 20px; background: var(--panel); box-shadow: var(--shadow);
          }}
          form {{ display: flex; gap: 10px; }}
          input {{
            min-width: 0; flex: 1; border: 1px solid #bfc9c3; border-radius: 12px;
            padding: 14px 16px; color: var(--ink); background: #fff; font: inherit;
          }}
          .examples {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
          .example {{
            border: 1px solid var(--line); border-radius: 999px; padding: 7px 11px;
            color: var(--muted); background: #f7f9f7; cursor: pointer; font: inherit;
            font-size: 12px;
          }}
          #result {{ display: none; margin-top: 22px; }}
          .answer {{ font-size: 17px; white-space: pre-wrap; }}
          .citation {{
            padding: 14px 0; border-top: 1px solid var(--line);
          }}
          .citation:first-child {{ border-top: 0; }}
          .citation-label {{ color: var(--forest); font-weight: 800; }}
          blockquote {{ margin: 8px 0 0; padding-left: 14px; border-left: 3px solid var(--amber); }}
          .trace {{
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px; margin-top: 12px;
          }}
          .trace div {{ padding: 10px; border-radius: 10px; background: #f1f5f2; font-size: 12px; }}
          .error {{ color: #a12626; }}
          @media (max-width: 700px) {{
            form {{ flex-direction: column; }}
            .trace {{ grid-template-columns: 1fr; }}
          }}
        </style>
      </head>
      <body>
        <nav>
          <a class="brand" href="/">Document Intelligence Workspace</a>
          <div class="nav-links">
            <a href="/evidence">Evidence</a>
            <a href="/signin">Google sign-in</a>
          </div>
        </nav>
        <main class="demo-shell">
          <span class="status">Public and read-only</span>
          <p class="eyebrow">Live cited retrieval</p>
          <h1>Ask the bundled research-paper corpus.</h1>
          <p class="lede">
            No login, database write, or external model call. Every answer is an
            extractive response whose quotes are validated against retrieved chunks.
          </p>
          <section class="composer">
            <form id="askForm">
              <input id="question" maxlength="500" required
                value="{html.escape(PUBLIC_DEMO_EXAMPLES[0])}"
                aria-label="Question">
              <button class="button primary" type="submit">Find cited evidence</button>
            </form>
            <div class="examples">{examples}</div>
          </section>
          <section id="result" class="grid" aria-live="polite">
            <article class="card">
              <p class="eyebrow">Response</p>
              <div id="answer" class="answer"></div>
              <div id="citations"></div>
            </article>
            <aside class="card">
              <p class="eyebrow">Execution trace</p>
              <div id="trace" class="trace"></div>
              <p class="note">
                The public route cannot create suggestions, tasks, reviews, or agent runs.
              </p>
            </aside>
          </section>
        </main>
        <footer>Try an unrelated question to see the evidence threshold refuse it.</footer>
        <script>
          const form = document.getElementById("askForm");
          const question = document.getElementById("question");
          const result = document.getElementById("result");
          const answer = document.getElementById("answer");
          const citations = document.getElementById("citations");
          const trace = document.getElementById("trace");
          const escapeHtml = (value) => String(value ?? "")
            .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");

          document.querySelectorAll(".example").forEach((button) => {{
            button.addEventListener("click", () => {{
              question.value = button.dataset.question;
              question.focus();
            }});
          }});

          form.addEventListener("submit", async (event) => {{
            event.preventDefault();
            result.style.display = "grid";
            answer.className = "answer";
            answer.textContent = "Searching evidence…";
            citations.innerHTML = "";
            trace.innerHTML = "";
            try {{
              const response = await fetch("/demo/ask", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ query: question.value }})
              }});
              const payload = await response.json();
              if (!response.ok) throw new Error(payload.detail || "Request failed");
              answer.textContent = payload.answer.answer;
              if (payload.answer.insufficient_evidence) answer.classList.add("error");
              citations.innerHTML = payload.answer.citations.map((citation) => `
                <div class="citation">
                  <span class="citation-label">${{escapeHtml(citation.label)}}</span>
                  · ${{escapeHtml(citation.heading_path.join(" › "))}}
                  <blockquote>${{escapeHtml(citation.quote)}}</blockquote>
                </div>`).join("");
              const fields = [
                ["Access", payload.trace.access],
                ["Retrieval", payload.trace.retrieval],
                ["Reranker", payload.trace.reranker],
                ["Generation", payload.trace.generation],
                ["Writes", payload.trace.writes_performed],
                ["Latency", payload.trace.latency_ms + " ms"]
              ];
              trace.innerHTML = fields.map(([label, value]) =>
                `<div><strong>${{escapeHtml(label)}}</strong><br>${{escapeHtml(value)}}</div>`
              ).join("");
            }} catch (error) {{
              answer.className = "answer error";
              answer.textContent = error.message;
            }}
          }});
        </script>
      </body>
    </html>
    """


def _public_evidence_html() -> str:
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Measured evidence · Document Intelligence Workspace</title>
        <style>
          {_public_page_styles()}
          .evidence {{ padding: 52px 0 80px; }}
          .evidence h1 {{ font-size: clamp(40px, 5vw, 62px); }}
          table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
          th, td {{ padding: 13px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
          th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
          .truth {{ margin-top: 18px; padding: 18px; border-left: 4px solid var(--amber); background: #fffaf0; }}
          .evidence .grid {{ grid-template-columns: 1.2fr .8fr; margin-top: 28px; }}
          @media (max-width: 760px) {{ .evidence .grid {{ grid-template-columns: 1fr; }} }}
        </style>
      </head>
      <body>
        <nav>
          <a class="brand" href="/">Document Intelligence Workspace</a>
          <div class="nav-links">
            <a href="/demo">Live demo</a>
            <a href="/signin">Google sign-in</a>
          </div>
        </nav>
        <main class="evidence">
          <p class="eyebrow">Claim-to-evidence boundary</p>
          <h1>Measured results and current limitations.</h1>
          <p class="lede">
            Results below come from the same frozen 40-question retrieval set.
            They compare the local hashing baseline with a semantic-embedding plus
            reciprocal-rank-fusion arm.
          </p>
          <div class="grid">
            <section class="card">
              <h2>Frozen retrieval comparison</h2>
              <table>
                <thead><tr><th>Configuration</th><th>Recall@5</th><th>MRR</th><th>Gold citation recall</th></tr></thead>
                <tbody>
                  <tr><td>Hashing + weighted fusion</td><td>0.2609</td><td>0.1935</td><td>0.1087</td></tr>
                  <tr><td>Semantic embeddings + RRF</td><td>0.2826</td><td>0.3022</td><td>0.2536</td></tr>
                </tbody>
              </table>
              <p class="note">
                This supports a claim about the combined configuration, not that
                RRF alone caused the improvement.
              </p>
            </section>
            <aside class="card">
              <h2>Current evidence gates</h2>
              <p><strong>Verified:</strong> Cloud Run deployment, Google OIDC token
              verification, fail-closed production routes, exact-quote citation checks,
              automated retrieval evaluation, and a reproducible 140-question calibration
              instrument with 112 aligned claim-citation pairs per annotator.</p>
              <p><strong>In progress:</strong> two independent human label sets,
              adjudication, and agreement calculation.</p>
              <div class="truth">
                No human-calibrated accuracy or inter-annotator agreement number is
                published until those labels exist.
              </div>
            </aside>
          </div>
        </main>
        <footer>Evidence labels distinguish deployed, tested, measured, and human-validated work.</footer>
      </body>
    </html>
    """


def _html_documents_table(documents: list[SourceDocument]) -> str:
    if not documents:
        return "<p>No documents loaded.</p>"
    rows = []
    for document in documents:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(document.id)}</code></td>"
            f"<td>{html.escape(document.source_name)}</td>"
            f"<td>{html.escape(document.source_type)}</td>"
            "</tr>"
        )
    return "<table><tr><th>ID</th><th>Name</th><th>Type</th></tr>" + "".join(rows) + "</table>"


def _google_signin_html(client_id: str) -> str:
    encoded_client_id = json.dumps(client_id)
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Sign in · Document Intelligence Workspace</title>
        <script src="https://accounts.google.com/gsi/client" async defer></script>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background:#f5f7fb;
                  color:#14213d; margin:0; min-height:100vh; display:grid; place-items:center; }}
          main {{ background:white; border:1px solid #d9e1ec; border-radius:16px; padding:30px;
                  width:min(440px, calc(100% - 48px)); box-shadow:0 16px 45px rgba(20,33,61,.1); }}
          h1 {{ margin-top:0; }} #identity {{ margin-top:20px; white-space:pre-wrap; }}
          .error {{ color:#a51c30; }}
        </style>
      </head>
      <body>
        <main>
          <h1>Document Intelligence Workspace</h1>
          <p>Sign in with Google to verify the deployed OIDC boundary.</p>
          <div id="googleButton"></div>
          <div id="identity"></div>
        </main>
        <script>
          const clientId = {encoded_client_id};
          async function handleCredentialResponse(response) {{
            const result = await fetch("/auth/whoami", {{
              headers: {{ "Authorization": "Bearer " + response.credential }}
            }});
            const payload = await result.json();
            const target = document.getElementById("identity");
            if (!result.ok) {{
              target.className = "error";
              target.textContent = payload.detail || "Authentication failed";
              return;
            }}
            target.className = "";
            target.textContent = "Authenticated as " + (payload.email || payload.subject);
          }}
          window.onload = () => {{
            google.accounts.id.initialize({{
              client_id: clientId,
              callback: handleCredentialResponse
            }});
            google.accounts.id.renderButton(
              document.getElementById("googleButton"),
              {{ theme: "outline", size: "large", width: 300 }}
            );
          }};
        </script>
      </body>
    </html>
    """


def _html_suggestions_table(suggestions: list[AISuggestion]) -> str:
    if not suggestions:
        return "<p>No pending suggestions.</p>"
    rows = []
    for suggestion in suggestions:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(suggestion.id)}</code></td>"
            f"<td>{html.escape(suggestion.status)}</td>"
            f"<td>{html.escape(suggestion.title)}</td>"
            "</tr>"
        )
    return "<table><tr><th>ID</th><th>Status</th><th>Title</th></tr>" + "".join(rows) + "</table>"


def _html_runs_table(runs: list[AIRun]) -> str:
    if not runs:
        return "<p>No AI runs yet.</p>"
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(run.id)}</code></td>"
            f"<td>{html.escape(run.run_type)}</td>"
            f"<td>{html.escape(run.query or '')}</td>"
            f"<td>{html.escape(str(run.citation_valid))}</td>"
            "</tr>"
        )
    return "<table><tr><th>ID</th><th>Type</th><th>Query</th><th>Citations valid</th></tr>" + "".join(rows) + "</table>"



def _workspace_html() -> str:
    return _load_template("workspace.html")

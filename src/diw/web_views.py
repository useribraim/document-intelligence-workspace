from __future__ import annotations

import html
import json

from diw.db.models import AIRun, AISuggestion, SourceDocument


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
        <meta name="description" content="A production-style AI research assistant with cited evidence and protected write actions.">
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
              A production-style research workspace that retrieves source evidence,
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
              <strong>40</strong>
              <span>frozen retrieval questions in the measured evaluation</span>
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
                <h3>Recruiter</h3>
                <p>No account required. The public page communicates the product,
                deployment, and measured evidence in under 30 seconds.</p>
              </article>
              <article class="card">
                <h3>Hiring manager</h3>
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
          Public demo data is synthetic. Human calibration is in progress and no
          inter-annotator agreement claim is published yet.
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
          <h1>What I measured—and what I refuse to claim.</h1>
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
              and automated retrieval evaluation.</p>
              <p><strong>In progress:</strong> primary human labels, independent second
              labels, adjudication, and agreement calculation.</p>
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
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Document Intelligence Workspace</title>
    <style>
      :root {
        --bg: #f6f7f9;
        --panel: #ffffff;
        --panel-subtle: #f1f3f5;
        --text: #17191c;
        --muted: #636b74;
        --border: #d8dde3;
        --accent: #216869;
        --accent-dark: #174d4f;
        --danger: #9f2f2f;
        --warning: #8a5a00;
        --ok: #216e39;
        --code: #20242a;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.45;
      }

      button,
      input,
      select,
      textarea {
        font: inherit;
      }

      .app-shell {
        min-height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr auto;
      }

      .topbar {
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 0 20px;
        background: var(--panel);
        border-bottom: 1px solid var(--border);
      }

      .brand {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 1px;
      }

      .brand strong {
        font-size: 15px;
        line-height: 1.1;
      }

      .brand span {
        color: var(--muted);
        font-size: 12px;
      }

      .top-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
      }

      .status-chip {
        color: var(--muted);
        font-size: 12px;
        white-space: nowrap;
      }

      .layout {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(560px, 1fr) minmax(320px, 34vw);
      }

      .thread {
        min-width: 0;
        display: flex;
        flex-direction: column;
        border-right: 1px solid var(--border);
        background: #fbfcfd;
      }

      .thread-scroll {
        min-height: 0;
        flex: 1;
        overflow: auto;
        padding: 16px;
      }

      .message {
        max-width: 920px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
      }

      .message.user {
        border-left: 4px solid var(--accent);
      }

      .message.assistant {
        border-left: 4px solid #5d6775;
      }

      .message.system {
        background: var(--panel-subtle);
      }

      .paper-workspace {
        max-width: none;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 14px;
        overflow: hidden;
      }

      .paper-workspace-header {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        padding: 14px;
        border-bottom: 1px solid var(--border);
        background: #f8fafc;
      }

      .paper-workspace-title {
        min-width: 0;
      }

      .paper-workspace-title h2 {
        margin: 0 0 4px;
        font-size: 17px;
      }

      .paper-workspace-title p {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
      }

      .paper-workspace-actions {
        display: grid;
        gap: 8px;
        min-width: 360px;
      }

      .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
      }

      .step-row {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 6px;
      }

      .step-button {
        min-width: 0;
        white-space: normal;
        text-align: center;
      }

      .step-button.primary-step {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
      }

      .step-button.completed-step {
        border-color: #b8d7c0;
        background: #f0faf2;
      }

      .step-helper {
        color: var(--muted);
        font-size: 13px;
        text-align: right;
      }

      .paper-workspace-grid {
        display: grid;
        grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.15fr);
        gap: 0;
      }

      .paper-workspace-grid.has-card {
        grid-template-columns: minmax(300px, 0.78fr) minmax(420px, 1.35fr);
      }

      .paper-workspace-column {
        min-width: 0;
        padding: 14px;
      }

      .paper-workspace-column + .paper-workspace-column {
        border-left: 1px solid var(--border);
      }

      .paper-workspace-column h3 {
        margin: 0 0 10px;
        font-size: 14px;
      }

      .paper-summary {
        display: grid;
        grid-template-columns: 92px minmax(0, 1fr);
        gap: 7px 10px;
        margin-bottom: 12px;
        font-size: 13px;
      }

      .paper-summary span:nth-child(odd) {
        color: var(--muted);
      }

      .details-panel {
        margin-bottom: 12px;
      }

      .details-panel summary {
        cursor: pointer;
        color: var(--muted);
        font-size: 13px;
      }

      .paper-chunk-list {
        display: grid;
        gap: 8px;
      }

      .paper-chunk {
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
      }

      .paper-chunk.active {
        border-color: var(--accent);
        background: #eef8f7;
      }

      .paper-artifact-preview {
        max-height: 420px;
        overflow: auto;
      }

      .study-card {
        display: grid;
        gap: 10px;
      }

      .study-section {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: #fbfcfd;
        padding: 10px;
      }

      .study-section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
      }

      .study-section h4 {
        margin: 0;
        font-size: 13px;
      }

      .study-section textarea {
        width: 100%;
        min-height: 74px;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px;
        resize: vertical;
        background: white;
      }

      .study-section.needs-attention {
        border-color: #dec58f;
        background: #fffaf0;
      }

      .study-section.accepted {
        background: #f7fbf8;
      }

      .source-details {
        margin-top: 10px;
      }

      .source-details summary {
        cursor: pointer;
        color: var(--muted);
      }

      .provenance-details {
        border-top: 1px solid var(--border);
        margin-top: 10px;
        padding-top: 10px;
      }

      .provenance-details summary {
        cursor: pointer;
        color: var(--muted);
      }

      .message-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      .answer-text {
        margin: 0 0 12px;
        white-space: pre-wrap;
      }

      .field-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 8px;
        margin-top: 10px;
      }

      .field {
        padding: 10px;
        background: var(--panel-subtle);
        border-radius: 6px;
        border: 1px solid var(--border);
      }

      .field label {
        display: block;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 4px;
      }

      .field div {
        overflow-wrap: anywhere;
      }

      .composer {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
        gap: 10px;
        padding: 14px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 14px;
      }

      .composer textarea {
        min-height: 72px;
        max-height: 180px;
        resize: vertical;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        color: var(--text);
      }

      .controls {
        display: grid;
        gap: 8px;
        min-width: 0;
      }

      .inline-controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      .import-controls {
        display: grid;
        gap: 6px;
      }

      .import-row {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }

      .import-row input[type="file"] {
        max-width: 260px;
        padding: 6px;
      }

      .import-status {
        min-height: 18px;
        font-size: 12px;
        color: var(--muted);
      }

      .technical-actions {
        display: none;
      }

      select,
      input {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 9px;
        background: white;
      }

      button {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 11px;
        background: white;
        color: var(--text);
        cursor: pointer;
        white-space: nowrap;
      }

      button.primary {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
      }

      button.primary:hover {
        background: var(--accent-dark);
      }

      button.secondary {
        background: #e9eef2;
      }

      button.danger {
        color: var(--danger);
      }

      button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }

      .inspector {
        min-width: 0;
        display: grid;
        grid-template-rows: auto 1fr;
        background: var(--panel);
      }

      .tabs {
        display: flex;
        gap: 2px;
        padding: 10px 12px 0;
        border-bottom: 1px solid var(--border);
        background: var(--panel);
        overflow-x: auto;
      }

      .tab {
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        border-bottom-color: transparent;
      }

      .tab.active {
        background: var(--panel-subtle);
        border-color: var(--border);
        border-bottom-color: var(--panel-subtle);
      }

      .inspector-scroll {
        min-height: 0;
        overflow: auto;
        padding: 16px;
        background: var(--panel-subtle);
      }

      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
      }

      .panel h2,
      .panel h3 {
        margin: 0 0 10px;
        font-size: 15px;
      }

      .meta-grid {
        display: grid;
        grid-template-columns: 120px minmax(0, 1fr);
        gap: 7px 10px;
        font-size: 13px;
      }

      .meta-grid span:nth-child(odd) {
        color: var(--muted);
      }

      code,
      pre {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      }

      code {
        overflow-wrap: anywhere;
      }

      pre {
        margin: 10px 0 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        color: var(--code);
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px;
        font-size: 12px;
      }

      .chunk {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: white;
        padding: 12px;
        margin-bottom: 10px;
      }

      .chunk-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 8px;
      }

      .chunk p {
        margin: 0;
        white-space: pre-wrap;
      }

      .pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 12px;
        color: var(--muted);
        background: white;
      }

      .pill.ok {
        color: var(--ok);
        border-color: #b8d7c0;
        background: #f0faf2;
      }

      .pill.warn {
        color: var(--warning);
        border-color: #dec58f;
        background: #fff8e8;
      }

      .pill.danger {
        color: var(--danger);
        border-color: #e4b6b6;
        background: #fff2f2;
      }

      .empty {
        color: var(--muted);
        margin: 0;
      }

      .review-card {
        display: grid;
        gap: 10px;
      }

      .review-actions {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      .review-form {
        display: grid;
        gap: 8px;
      }

      .review-form textarea {
        min-height: 72px;
        resize: vertical;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 9px;
      }

      .history-item {
        width: 100%;
        display: block;
        text-align: left;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
        white-space: normal;
      }

      .history-item small {
        display: block;
        color: var(--muted);
        margin-top: 4px;
      }

      .corpus-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
        gap: 10px;
      }

      .corpus-list {
        display: grid;
        gap: 8px;
      }

      .corpus-item {
        width: 100%;
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
      }

      .corpus-item.active {
        border-color: var(--accent);
        background: #eef8f7;
      }

      .chunk-detail {
        min-height: 180px;
      }

      @media (max-width: 1120px) {
        .corpus-grid {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 920px) {
        .layout {
          grid-template-columns: 1fr;
        }

        .thread {
          border-right: 0;
        }

        .inspector {
          min-height: 520px;
          border-top: 1px solid var(--border);
        }

        .composer {
          grid-template-columns: 1fr;
        }

        .controls {
          min-width: 0;
        }

        .paper-workspace-header,
        .paper-workspace-grid {
          display: block;
        }

        .paper-workspace-actions {
          min-width: 0;
          margin-top: 12px;
        }

        .step-row {
          grid-template-columns: 1fr;
        }

        .step-helper {
          text-align: left;
        }

        .paper-workspace-column + .paper-workspace-column {
          border-left: 0;
          border-top: 1px solid var(--border);
        }
      }

      @media (max-width: 760px) {
        .topbar {
          padding: 0 12px;
          gap: 10px;
        }

        .brand span,
        .status-chip {
          display: none;
        }

        .brand strong {
          font-size: 14px;
        }

        .top-actions {
          gap: 6px;
        }

        .top-actions button {
          padding: 7px 9px;
        }
      }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <header class="topbar">
        <div class="brand">
          <strong>Document Intelligence Workspace</strong>
          <span>Source-cited QA with validation and review</span>
        </div>
        <div class="top-actions">
          <span id="healthStatus" class="status-chip">Checking database...</span>
          <a href="/"><button type="button">Dashboard</button></a>
          <a href="/docs"><button type="button">API</button></a>
          <button id="refreshButton" type="button">Refresh</button>
        </div>
      </header>

      <main class="layout">
        <section class="thread" aria-label="Paper workflow">
          <div id="threadScroll" class="thread-scroll">
            <form id="workflowForm" class="composer">
            <textarea id="queryInput" aria-label="Question or extraction instruction">Extract the method, dataset, metric, and limitation from the cited paper section.</textarea>
            <div class="controls">
              <select id="taskModeSelect" aria-label="Task mode">
                <option value="structured_extraction">extract fields</option>
                <option value="question_answering">ask question</option>
                <option value="comparison">compare documents</option>
                <option value="contradiction_check">find contradictions</option>
                <option value="study_note">study note</option>
              </select>
              <div class="inline-controls">
                <select id="modeSelect" aria-label="Retrieval mode">
                  <option value="hybrid">hybrid</option>
                  <option value="vector">vector</option>
                  <option value="lexical">lexical</option>
                </select>
                <input id="topKInput" aria-label="Top K" type="number" min="1" max="12" value="3">
              </div>
              <div class="import-controls">
                <div class="import-row">
                  <input id="documentImportInput" aria-label="Import Markdown or text document" type="file" accept=".md,.txt,text/markdown,text/plain">
                  <button id="documentImportButton" class="secondary" type="button">Import document</button>
                </div>
                <div id="documentImportStatus" class="import-status">Import .md or .txt to start from a new source.</div>
              </div>
              <div class="technical-actions" hidden aria-hidden="true">
                <button id="previewButton" class="secondary" type="button" tabindex="-1">Preview evidence</button>
                <button id="generateButton" class="primary" type="submit" tabindex="-1" disabled>Generate from evidence</button>
              </div>
            </div>
            </form>
            <section id="paperWorkspace" class="paper-workspace" aria-label="Paper workspace"></section>
            <div id="messages"></div>
          </div>
        </section>

        <aside class="inspector" aria-label="Evidence and review inspector">
          <nav class="tabs" aria-label="Inspector tabs">
            <button type="button" class="tab active" data-tab="evidence">Evidence</button>
            <button type="button" class="tab" data-tab="review">Review</button>
            <button type="button" class="tab" data-tab="runs">Runs</button>
            <button type="button" class="tab" data-tab="eval">Eval</button>
            <button type="button" class="tab" data-tab="corpus">Corpus</button>
            <button type="button" class="tab" data-tab="run">Run</button>
          </nav>
          <div class="inspector-scroll">
            <section id="evidencePanel" class="tab-panel"></section>
            <section id="corpusPanel" class="tab-panel" hidden></section>
            <section id="paperCardPanel" class="tab-panel" hidden></section>
            <section id="runPanel" class="tab-panel" hidden></section>
            <section id="reviewPanel" class="tab-panel" hidden></section>
            <section id="runsPanel" class="tab-panel" hidden></section>
            <section id="evalPanel" class="tab-panel" hidden></section>
          </div>
        </aside>
      </main>
    </div>

    <script>
      const state = {
        answer: null,
        preview: null,
        documents: [],
        versions: [],
        chunks: [],
        selectedDocumentId: null,
        selectedVersionId: null,
        selectedChunkId: null,
        paperCard: null,
        cardAccepted: false,
        cardNeedsReview: false,
        editedCardFields: {},
        editedFieldKeys: [],
        paperCards: [],
        paperCardsDir: "",
        reviewDecision: null,
        workflowStatus: "Ready",
        savedArtifact: null,
        lastAnswerKey: "",
        evalCases: [],
        evalCasesPath: "",
        runs: [],
        suggestions: [],
        activeTab: "evidence"
      };

      const messages = document.getElementById("messages");
      const paperWorkspace = document.getElementById("paperWorkspace");
      const evidencePanel = document.getElementById("evidencePanel");
      const corpusPanel = document.getElementById("corpusPanel");
      const paperCardPanel = document.getElementById("paperCardPanel");
      const runPanel = document.getElementById("runPanel");
      const reviewPanel = document.getElementById("reviewPanel");
      const runsPanel = document.getElementById("runsPanel");
      const evalPanel = document.getElementById("evalPanel");
      const workflowForm = document.getElementById("workflowForm");
      const previewButton = document.getElementById("previewButton");
      const generateButton = document.getElementById("generateButton");
      const queryInput = document.getElementById("queryInput");
      const taskModeSelect = document.getElementById("taskModeSelect");
      const modeSelect = document.getElementById("modeSelect");
      const topKInput = document.getElementById("topKInput");
      const healthStatus = document.getElementById("healthStatus");

      function escapeHtml(value) {
        return String(value ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      function statusPill(label, status) {
        return `<span class="pill ${status}">${escapeHtml(label)}</span>`;
      }

      function renderAnswerBody(value) {
        const lines = String(value || "").split("\\n");
        return lines.map((line) => {
          if (line.startsWith("## ")) {
            return escapeHtml(line.slice(3));
          }
          if (line.startsWith("# ")) {
            return escapeHtml(line.slice(2));
          }
          return escapeHtml(line);
        }).join("<br>");
      }

      function shortId(value) {
        if (!value) return "";
        return String(value).slice(0, 8);
      }

      function renderThread() {
        messages.innerHTML = "";
      }

      function selectedDocument() {
        return state.documents.find((document) => document.id === state.selectedDocumentId) || null;
      }

      function selectedVersion() {
        return state.versions.find((version) => version.id === state.selectedVersionId) || null;
      }

      function selectedChunk() {
        return state.chunks.find((chunk) => chunk.id === state.selectedChunkId) || null;
      }

      function currentRequestKey() {
        return [
          queryInput.value.trim(),
          taskModeSelect.value,
          modeSelect.value,
          topKInput.value || "3"
        ].join("|");
      }

      const studyFieldLabels = {
        core_idea: "Core idea",
        problem: "Problem",
        method: "Method",
        dataset: "Dataset / corpus",
        metric: "Metrics",
        result: "Results",
        limitation: "Limitations"
      };
      const requiredStudyFields = ["core_idea", "problem", "method"];

      function fieldNeedsAttention(value) {
        return !value || value === "Not identified in source." || value === "Needs review.";
      }

      function cardFieldValue(key) {
        return state.editedCardFields[key] ?? state.paperCard?.extracted_fields?.[key] ?? "";
      }

      function requiredMissingCount() {
        return requiredStudyFields.filter((key) => !cardFieldValue(key).trim()).length;
      }

      function needsAttentionCount() {
        return Object.keys(studyFieldLabels).filter((key) => fieldNeedsAttention(cardFieldValue(key))).length;
      }

      function fieldStatus(key, value) {
        if (state.editedFieldKeys.includes(key)) return ["edited", "warn"];
        if (fieldNeedsAttention(value)) return ["needs attention", "warn"];
        return ["source-backed", "ok"];
      }

      function cardLifecycleStatus() {
        if (state.savedArtifact) return "Saved";
        if (state.cardAccepted) return "Accepted";
        if (state.cardNeedsReview) return "Needs review";
        if (state.paperCard) return requiredMissingCount() ? "Needs review" : "Ready to accept";
        if (state.answer) return "Generated";
        if (state.preview) return "Evidence found";
        return "Not started";
      }

      function currentStep() {
        if (state.savedArtifact) return "saved";
        if (state.cardAccepted) return "save";
        if (state.paperCard) return "accept";
        if (state.answer) return "build";
        if (state.preview) return "generate";
        return "evidence";
      }

      function primaryActionForStep() {
        return {
          evidence: "preview",
          generate: "generate",
          build: "draft-card",
          accept: "accept-card",
          save: "save-card",
          saved: ""
        }[currentStep()];
      }

      function stepHelperText() {
        return {
          evidence: "Find source sections before generating a card.",
          generate: "Generate a draft from the selected evidence.",
          build: "Review extracted fields and fill missing required sections.",
          accept: "Accept the card when the required fields are complete.",
          save: "Save the accepted card as a Markdown note.",
          saved: "Note saved."
        }[currentStep()];
      }

      function stepIsComplete(action) {
        if (action === "preview") return Boolean(state.preview || state.answer || state.paperCard);
        if (action === "generate") return Boolean(state.answer || state.paperCard);
        if (action === "draft-card") return Boolean(state.paperCard);
        if (action === "accept-card") return Boolean(state.cardAccepted || state.savedArtifact);
        if (action === "save-card") return Boolean(state.savedArtifact);
        return false;
      }

      function renderStepRow(canAcceptCard, canSaveNote, latestEvidence) {
        const primaryAction = primaryActionForStep();
        const steps = [
          ["preview", "1 Find evidence", true],
          ["generate", "2 Generate draft", Boolean(latestEvidence)],
          ["draft-card", "3 Build card", Boolean(state.selectedVersionId)],
          ["accept-card", "4 Accept", canAcceptCard && !state.cardAccepted],
          ["save-card", "5 Save", canSaveNote]
        ];
        const buttons = steps.map(([action, label, enabled]) => {
          const isPrimary = action === primaryAction;
          const isComplete = stepIsComplete(action);
          return `
            <button type="button"
              class="step-button ${isPrimary ? "primary-step" : ""} ${isComplete ? "completed-step" : ""}"
              data-workspace-action="${escapeHtml(action)}"
              ${enabled ? "" : "disabled"}>
              ${escapeHtml(label)}
            </button>
          `;
        }).join("");
        return `
          <div class="step-row">${buttons}</div>
          <div class="step-helper">${escapeHtml(stepHelperText())}</div>
        `;
      }

      function renderCompactStatus(latestEvidence, attentionCount, missingRequired) {
        return `
          <div class="status-row">
            ${statusPill(`${latestEvidence ? (latestEvidence.retrieved_chunks || []).length : 0} evidence sections`, latestEvidence ? "ok" : "")}
            ${statusPill(cardLifecycleStatus(), state.cardAccepted || state.savedArtifact ? "ok" : state.paperCard ? "warn" : "")}
            ${state.paperCard ? statusPill(`Needs attention: ${attentionCount}`, attentionCount ? "warn" : "ok") : ""}
            ${state.paperCard ? statusPill(`Required missing: ${missingRequired}`, missingRequired ? "danger" : "ok") : ""}
          </div>
        `;
      }

      function renderStudyCardEditor() {
        const fields = Object.keys(studyFieldLabels).map((key) => {
          const value = cardFieldValue(key);
          const needsAttention = fieldNeedsAttention(value);
          const [statusLabel, statusKind] = fieldStatus(key, value);
          return `
            <section class="study-section ${needsAttention ? "needs-attention" : ""} ${state.cardAccepted ? "accepted" : ""}">
              <div class="study-section-header">
                <h4>${escapeHtml(studyFieldLabels[key])}</h4>
                ${statusPill(statusLabel, statusKind)}
              </div>
              <textarea data-card-field="${escapeHtml(key)}" aria-label="${escapeHtml(studyFieldLabels[key])}">${escapeHtml(needsAttention ? "" : value)}</textarea>
            </section>
          `;
        }).join("");
        return `
          <div class="study-card">
            ${fields}
            <details class="provenance-details">
              <summary>Provenance</summary>
              <pre>${escapeHtml((state.paperCard.source_chunk_ids || []).join("\\n"))}</pre>
            </details>
          </div>
        `;
      }

      function buildEditedPaperCardMarkdown() {
        if (!state.paperCard) return "";
        const evidence = (state.paperCard.source_chunk_ids || [])
          .map((chunkId, index) => `- [C${index + 1}] \\`${chunkId}\\``)
          .join("\\n");
        return [
          `# ${state.paperCard.title}`,
          "",
          "## Core Idea",
          "",
          cardFieldValue("core_idea") || "Not identified in source.",
          "",
          "## Problem",
          "",
          cardFieldValue("problem") || "Not identified in source.",
          "",
          "## Method",
          "",
          cardFieldValue("method") || "Not identified in source.",
          "",
          "## Dataset",
          "",
          cardFieldValue("dataset") || "Not identified in source.",
          "",
          "## Metrics",
          "",
          cardFieldValue("metric") || "Not identified in source.",
          "",
          "## Results",
          "",
          cardFieldValue("result") || "Not identified in source.",
          "",
          "## Limitations",
          "",
          cardFieldValue("limitation") || "Not identified in source.",
          "",
          "## Source Evidence",
          "",
          evidence
        ].join("\\n");
      }

      function renderPaperWorkspace() {
        const documentRecord = selectedDocument();
        const version = selectedVersion();
        const chunk = selectedChunk();
        const latestEvidence = state.answer || state.preview;
        const extractedFields = state.answer?.answer?.extracted_fields || {};
        const extractedFieldCount = Object.keys(extractedFields).length;
        const currentSuggestion = state.answer?.suggestion_id
          ? state.suggestions.find((item) => item.id === state.answer.suggestion_id)
          : null;
        const cardState = state.paperCard ? cardLifecycleStatus().toLowerCase() : "not started";
        const reviewState = currentSuggestion ? "pending review" : state.reviewDecision?.decision || "not reviewed";
        const missingRequired = state.paperCard ? requiredMissingCount() : 0;
        const attentionCount = state.paperCard ? needsAttentionCount() : 0;
        const canAcceptCard = state.paperCard && missingRequired === 0;
        const canSaveNote = state.paperCard && state.cardAccepted;

        paperWorkspace.innerHTML = `
          <div class="paper-workspace-header">
            <div class="paper-workspace-title">
              <h2>${escapeHtml(documentRecord?.source_name || "No paper selected")}</h2>
              <p>${escapeHtml(version ? `${state.chunks.length} source sections | Study card: ${cardState}` : "Load a document before using the paper workspace.")}</p>
              <p>${escapeHtml(state.workflowStatus)}</p>
              ${renderCompactStatus(latestEvidence, attentionCount, missingRequired)}
            </div>
            <div class="paper-workspace-actions">
              ${renderStepRow(canAcceptCard, canSaveNote, latestEvidence)}
            </div>
          </div>
          <div class="paper-workspace-grid ${state.paperCard ? "has-card" : ""}">
            <div class="paper-workspace-column">
              <h3>Source and evidence</h3>
              <div class="paper-summary">
                <span>Document</span><span>${escapeHtml(documentRecord?.source_name || "none")}</span>
                <span>Selected</span><span>${escapeHtml(chunk ? ((chunk.heading_path || []).join(" / ") || `chunk ${chunk.chunk_index}`) : "none")}</span>
                <span>Evidence</span><span>${escapeHtml(latestEvidence ? `${(latestEvidence.retrieved_chunks || []).length} sections found` : "not found")}</span>
                <span>Review</span><span>${escapeHtml(reviewState)}</span>
              </div>
              <div class="paper-chunk-list">
                ${state.chunks.length ? state.chunks.slice(0, 6).map((item) => `
                  <button type="button" class="paper-chunk ${item.id === state.selectedChunkId ? "active" : ""}" data-workspace-chunk-id="${escapeHtml(item.id)}">
                    ${escapeHtml((item.heading_path || []).join(" / ") || `Chunk ${item.chunk_index}`)}
                    <small>lines ${escapeHtml(item.start_line)}-${escapeHtml(item.end_line)}</small>
                  </button>
                `).join("") : `<p class="empty">No chunks loaded for the selected version.</p>`}
              </div>
              ${chunk ? `
                <details class="source-details">
                  <summary>Show source text</summary>
                  <pre>${escapeHtml(chunk.text)}</pre>
                </details>
              ` : ""}
            </div>
            <div class="paper-workspace-column">
              <h3>Study card</h3>
              <details class="details-panel">
                <summary>Details</summary>
                <div class="paper-summary">
                  <span>Answer</span><span>${escapeHtml(state.answer ? (extractedFieldCount ? `${extractedFieldCount} extracted fields` : "generated answer") : "not generated")}</span>
                  <span>Citations</span><span>${escapeHtml(state.answer?.citation_validation?.valid === true ? "valid" : state.answer ? "needs attention" : "not checked")}</span>
                  <span>Study card</span><span>${escapeHtml(cardLifecycleStatus())}</span>
                  ${state.paperCard ? `<span>Needs attention</span><span>${escapeHtml(attentionCount)}</span>` : ""}
                  <span>Saved cards</span><span>${escapeHtml(state.paperCards.length)}</span>
                  ${state.savedArtifact ? `
                    <span>Saved path</span><code>${escapeHtml(state.savedArtifact.path)}</code>
                    <span>Saved at</span><span>${escapeHtml(state.savedArtifact.saved_at)}</span>
                  ` : ""}
                </div>
              </details>
              ${state.paperCard ? `
                ${renderStudyCardEditor()}
              ` : state.answer ? `
                <div class="field">
                  <label>Generated answer</label>
                  <div>${renderAnswerBody(state.answer.answer?.answer || "")}</div>
                </div>
                ${extractedFieldCount ? `
                  <div class="field-grid">
                    ${Object.entries(extractedFields).map(([key, value]) => `
                      <div class="field">
                        <label>${escapeHtml(key)}</label>
                        <div>${escapeHtml(value)}</div>
                      </div>
                    `).join("")}
                  </div>
                ` : ""}
              ` : `
                <div class="field">
                  <label>No study card yet</label>
                  <div>Find evidence, generate a draft, then build the editable card.</div>
                </div>
              `}
            </div>
          </div>
        `;
      }

      function renderEvidence() {
        const evidenceSource = state.answer || state.preview;
        if (!evidenceSource) {
          evidencePanel.innerHTML = `
            <div class="panel">
              <h2>Evidence</h2>
              <p class="empty">Preview evidence before generating an answer.</p>
            </div>
          `;
          return;
        }

        const validation = state.answer?.citation_validation;
        const chunks = evidenceSource.retrieved_chunks || [];
        const validationClass = validation?.valid ? "ok" : "danger";
        evidencePanel.innerHTML = `
          <div class="panel">
            <h2>Evidence workflow</h2>
            <div class="meta-grid">
              <span>Task</span><span>${escapeHtml(taskModeSelect.value)}</span>
              <span>Query</span><span>${escapeHtml(evidenceSource.query)}</span>
              <span>Mode</span><span>${escapeHtml(evidenceSource.mode)}</span>
              <span>Chunks</span><span>${escapeHtml(chunks.length)}</span>
            </div>
          </div>
          ${validation ? `
            <div class="panel">
              <h2>Citation validation</h2>
              ${statusPill(validation.valid ? "valid" : "invalid", validationClass)}
              <pre>${escapeHtml(JSON.stringify(validation, null, 2))}</pre>
            </div>
          ` : ""}
          <div class="panel">
            <h2>Retrieved chunks</h2>
            ${chunks.length ? chunks.map(renderChunk).join("") : `<p class="empty">No chunks retrieved.</p>`}
          </div>
        `;
      }

      function renderChunk(chunk) {
        const heading = (chunk.heading_path || []).join(" / ") || "Untitled section";
        return `
          <article class="chunk">
            <div class="chunk-header">
              <span>${escapeHtml(heading)}</span>
              <span>score ${escapeHtml(chunk.score)} | lex ${escapeHtml(chunk.lexical_score)} | vec ${escapeHtml(chunk.vector_score)}</span>
            </div>
            <p>${escapeHtml(chunk.text)}</p>
            <pre>chunk_id: ${escapeHtml(chunk.chunk_id)}
document_id: ${escapeHtml(chunk.document_id)}
version_id: ${escapeHtml(chunk.version_id)}</pre>
          </article>
        `;
      }

      function renderRun() {
        if (!state.answer) {
          runPanel.innerHTML = `
            <div class="panel">
              <h2>AI run</h2>
              <p class="empty">Preview evidence first, then generate to create an AI run.</p>
            </div>
          `;
          return;
        }

        runPanel.innerHTML = `
          <div class="panel">
            <h2>Run metadata</h2>
            <div class="meta-grid">
              <span>Run ID</span><code>${escapeHtml(state.answer.ai_run_id)}</code>
              <span>Suggestion</span><code>${escapeHtml(state.answer.suggestion_id)}</code>
              <span>Provider</span><span>${escapeHtml(state.answer.llm_provider)}</span>
              <span>Model</span><span>${escapeHtml(state.answer.llm_model)}</span>
              <span>Embedding</span><span>${escapeHtml(state.answer.embedding_model)}</span>
              <span>Mode</span><span>${escapeHtml(state.answer.mode)}</span>
              <span>Evidence</span><span>${escapeHtml((state.answer.retrieved_chunks || []).length)} chunks</span>
            </div>
          </div>
          <div class="panel">
            <h2>Raw output</h2>
            <pre>${escapeHtml(JSON.stringify(state.answer.answer, null, 2))}</pre>
          </div>
        `;
      }

      function renderRuns() {
        if (!state.runs.length) {
          runsPanel.innerHTML = `
            <div class="panel">
              <h2>Run history</h2>
              <p class="empty">No previous runs loaded.</p>
            </div>
          `;
          return;
        }
        runsPanel.innerHTML = `
          <div class="panel">
            <h2>Run history</h2>
            ${state.runs.map((run) => `
              <button type="button" class="history-item" data-run-id="${escapeHtml(run.id)}">
                ${escapeHtml(run.query || run.run_type)}
                <small>${escapeHtml(run.run_type)} | ${escapeHtml(run.retrieval_mode || "")} | citations ${escapeHtml(String(run.citation_valid))}</small>
              </button>
            `).join("")}
          </div>
        `;
      }

      function renderCorpus() {
        const selectedChunk = state.chunks.find((chunk) => chunk.id === state.selectedChunkId);
        corpusPanel.innerHTML = `
          <div class="panel">
            <h2>Corpus browser</h2>
            <div class="meta-grid">
              <span>Documents</span><span>${escapeHtml(state.documents.length)}</span>
              <span>Versions</span><span>${escapeHtml(state.versions.length)}</span>
              <span>Chunks</span><span>${escapeHtml(state.chunks.length)}</span>
            </div>
          </div>
          <div class="corpus-grid">
            <div class="panel">
              <h3>Documents</h3>
              <div class="corpus-list">
                ${state.documents.length ? state.documents.map((document) => `
                  <button type="button" class="corpus-item ${document.id === state.selectedDocumentId ? "active" : ""}" data-document-id="${escapeHtml(document.id)}">
                    ${escapeHtml(document.source_name)}
                    <small>${escapeHtml(document.source_type)} | ${escapeHtml(shortId(document.id))}</small>
                  </button>
                `).join("") : `<p class="empty">No documents loaded.</p>`}
              </div>
            </div>
            <div class="panel">
              <h3>Versions</h3>
              <div class="corpus-list">
                ${state.versions.length ? state.versions.map((version) => `
                  <button type="button" class="corpus-item ${version.id === state.selectedVersionId ? "active" : ""}" data-version-id="${escapeHtml(version.id)}">
                    ${escapeHtml(shortId(version.id))}
                    <small>hash ${escapeHtml(shortId(version.content_hash))} | ${escapeHtml(version.ingested_at || "")}</small>
                  </button>
                `).join("") : `<p class="empty">Select a document to inspect versions.</p>`}
              </div>
            </div>
          </div>
          <div class="panel">
            <h3>Chunks</h3>
            ${state.chunks.length ? state.chunks.map((chunk) => `
              <button type="button" class="corpus-item ${chunk.id === state.selectedChunkId ? "active" : ""}" data-chunk-id="${escapeHtml(chunk.id)}">
                ${escapeHtml((chunk.heading_path || []).join(" / ") || "Untitled section")}
                <small>chunk ${escapeHtml(chunk.chunk_index)} | lines ${escapeHtml(chunk.start_line)}-${escapeHtml(chunk.end_line)}</small>
              </button>
            `).join("") : `<p class="empty">Select a version to inspect chunks.</p>`}
          </div>
          <div class="panel chunk-detail">
            <h3>Chunk detail</h3>
            ${selectedChunk ? `
              <div class="meta-grid">
                <span>Chunk ID</span><code>${escapeHtml(selectedChunk.id)}</code>
                <span>Version</span><code>${escapeHtml(selectedChunk.version_id)}</code>
                <span>Hash</span><code>${escapeHtml(selectedChunk.content_hash)}</code>
                <span>Lines</span><span>${escapeHtml(selectedChunk.start_line)}-${escapeHtml(selectedChunk.end_line)}</span>
              </div>
              <pre>${escapeHtml(selectedChunk.text)}</pre>
            ` : `<p class="empty">Select a chunk to inspect source text and provenance.</p>`}
          </div>
        `;
      }

      function renderPaperCard() {
        paperCardPanel.innerHTML = `
          <div class="panel">
            <h2>Paper card compiler</h2>
            <div class="meta-grid">
              <span>Version</span><code>${escapeHtml(state.selectedVersionId || "none selected")}</code>
              <span>Saved cards</span><span>${escapeHtml(state.paperCards.length)}</span>
              <span>Directory</span><code>${escapeHtml(state.paperCardsDir || "data/demo/wiki/paper_cards")}</code>
            </div>
            <div class="review-actions" style="margin-top: 12px;">
              <button type="button" class="primary" data-paper-card-action="draft" ${state.selectedVersionId ? "" : "disabled"}>Draft paper card</button>
              <button type="button" data-paper-card-action="save" ${state.paperCard ? "" : "disabled"}>Save Markdown artifact</button>
            </div>
          </div>
          <div class="panel">
            <h2>Draft</h2>
            ${state.paperCard ? `
              <div class="meta-grid">
                <span>Title</span><span>${escapeHtml(state.paperCard.title)}</span>
                <span>Suggestion</span><code>${escapeHtml(state.paperCard.suggestion_id)}</code>
                <span>Chunks</span><span>${escapeHtml((state.paperCard.source_chunk_ids || []).length)}</span>
              </div>
              <pre>${escapeHtml(state.paperCard.markdown)}</pre>
            ` : `<p class="empty">Select a document version in Corpus, then draft a paper card.</p>`}
          </div>
          <div class="panel">
            <h2>Saved artifacts</h2>
            ${state.paperCards.length ? state.paperCards.map((card) => `
              <article class="chunk">
                <div class="chunk-header">
                  <span>${escapeHtml(card.title)}</span>
                  <span>${escapeHtml(card.size_bytes)} bytes</span>
                </div>
                <code>${escapeHtml(card.path)}</code>
              </article>
            `).join("") : `<p class="empty">No paper cards saved yet.</p>`}
          </div>
        `;
      }

      function renderEvalCases() {
        const latestCases = state.evalCases || [];
        evalPanel.innerHTML = `
          <div class="panel">
            <h2>Review evaluation cases</h2>
            <div class="meta-grid">
              <span>Saved cases</span><span>${escapeHtml(latestCases.length)}</span>
              <span>Path</span><code>${escapeHtml(state.evalCasesPath || "data/demo/evals/review_cases.jsonl")}</code>
            </div>
          </div>
          <div class="panel">
            <h2>Latest cases</h2>
            ${latestCases.length ? latestCases.map((testCase) => `
              <article class="chunk">
                <div class="chunk-header">
                  <span>${escapeHtml(testCase.task)}</span>
                  <span>${escapeHtml(testCase.expected_behavior)}</span>
                </div>
                <p>${escapeHtml(testCase.question)}</p>
                <pre>${escapeHtml(JSON.stringify({
                  id: testCase.id,
                  suggestion_id: testCase.suggestion_id,
                  retrieved_chunk_ids: testCase.retrieved_chunk_ids,
                  review_note: testCase.review_note
                }, null, 2))}</pre>
              </article>
            `).join("") : `<p class="empty">No review-derived eval cases saved yet.</p>`}
          </div>
        `;
      }

      function renderReview() {
        const currentSuggestionId = state.answer?.suggestion_id;
        const current = state.suggestions.find((item) => item.id === currentSuggestionId);
        if (!state.answer) {
          reviewPanel.innerHTML = `
            <div class="panel">
              <h2>Review</h2>
              <p class="empty">Run a question to create a pending review suggestion.</p>
            </div>
          `;
          return;
        }

        if (!current) {
          reviewPanel.innerHTML = `
            <div class="panel">
              <h2>Review</h2>
              <p class="empty">Suggestion ${escapeHtml(shortId(currentSuggestionId))} is no longer pending.</p>
              ${state.reviewDecision && ["reject", "edit"].includes(state.reviewDecision.decision) ? `
                <div class="review-actions">
                  <button type="button" class="primary" data-create-eval-case="true">Create eval case</button>
                </div>
              ` : ""}
            </div>
          `;
          return;
        }

        reviewPanel.innerHTML = `
          <div class="panel review-card">
            <h2>Pending suggestion</h2>
            <div class="meta-grid">
              <span>Status</span><span>${statusPill(current.status, current.status === "pending" ? "warn" : "ok")}</span>
              <span>ID</span><code>${escapeHtml(current.id)}</code>
              <span>AI run</span><code>${escapeHtml(current.ai_run_id)}</code>
              <span>Type</span><span>${escapeHtml(current.suggestion_type)}</span>
            </div>
            <div class="review-form">
              <select id="citationQualitySelect" aria-label="Citation quality">
                <option value="citations_supported">citations supported</option>
                <option value="citation_issue">citation issue</option>
                <option value="not_checked">not checked</option>
              </select>
              <select id="completenessSelect" aria-label="Completeness">
                <option value="complete">complete</option>
                <option value="incomplete">incomplete</option>
                <option value="too_broad">too broad</option>
              </select>
              <select id="fieldCorrectnessSelect" aria-label="Field correctness">
                <option value="fields_correct">fields correct</option>
                <option value="field_issue">field issue</option>
                <option value="not_applicable">not applicable</option>
              </select>
              <textarea id="reviewNoteInput" aria-label="Review note" placeholder="Evidence-based review note"></textarea>
            </div>
            <div class="review-actions">
              <button type="button" class="primary" data-review="accept">Accept</button>
              <button type="button" data-review="edit">Mark edited</button>
              <button type="button" class="danger" data-review="reject">Reject</button>
            </div>
          </div>
        `;
      }

      function renderPanels() {
        renderPaperWorkspace();
        renderEvidence();
        renderCorpus();
        renderPaperCard();
        renderRun();
        renderReview();
        renderRuns();
        renderEvalCases();
      }

      async function requestJson(url, options = {}) {
        const isRawBody = options.body instanceof FormData || options.body instanceof Blob;
        const response = await fetch(url, {
          headers: { ...(isRawBody ? {} : { "Content-Type": "application/json" }), ...(options.headers || {}) },
          ...options
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(`${response.status} ${detail}`);
        }
        return response.json();
      }

      async function refreshHealth() {
        const health = await requestJson("/health");
        healthStatus.textContent = `${health.documents} docs | ${health.chunks} chunks | ${health.pending_suggestions} pending`;
      }

      async function refreshSuggestions() {
        const payload = await requestJson("/review/suggestions?status=pending");
        state.suggestions = payload.suggestions || [];
      }

      async function refreshRuns() {
        const payload = await requestJson("/ai-runs?limit=10");
        state.runs = payload.ai_runs || [];
      }

      async function refreshEvalCases() {
        const payload = await requestJson("/evaluation-cases?limit=10");
        state.evalCases = payload.cases || [];
        state.evalCasesPath = payload.path || "";
      }

      async function refreshCorpus() {
        const payload = await requestJson("/documents");
        state.documents = payload.documents || [];
        if (!state.selectedDocumentId && state.documents.length) {
          state.selectedDocumentId = state.documents[0].id;
          await loadDocumentVersions(state.selectedDocumentId);
        }
      }

      function resetWorkspaceAnalysisState() {
        state.answer = null;
        state.preview = null;
        state.paperCard = null;
        state.cardAccepted = false;
        state.cardNeedsReview = false;
        state.editedCardFields = {};
        state.editedFieldKeys = [];
        state.savedArtifact = null;
        state.reviewDecision = null;
        state.lastAnswerKey = "";
      }

      async function importDocument() {
        const file = documentImportInput.files?.[0];
        if (!file) {
          documentImportStatus.textContent = "Choose a .md or .txt file first.";
          return;
        }
        const extension = file.name.toLowerCase().split(".").pop();
        if (!["md", "txt"].includes(extension)) {
          documentImportStatus.textContent = "Only .md and .txt files are supported.";
          return;
        }
        documentImportButton.disabled = true;
        documentImportStatus.textContent = "Importing document...";
        try {
          const params = new URLSearchParams({
            filename: file.name,
            target_chars: "1200",
            overlap_chars: "160"
          });
          const imported = await requestJson(`/documents/import?${params.toString()}`, {
            method: "POST",
            body: file
          });
          resetWorkspaceAnalysisState();
          state.selectedDocumentId = imported.document_id;
          await refreshCorpus();
          await loadDocumentVersions(imported.document_id);
          state.selectedVersionId = imported.version_id;
          await loadVersionChunks(imported.version_id);
          state.workflowStatus = `Imported ${imported.source_name}`;
          documentImportStatus.textContent = `${imported.source_name}: ${imported.chunk_count} chunks imported.`;
          documentImportInput.value = "";
          await refreshHealth();
          renderPanels();
        } catch (error) {
          documentImportStatus.textContent = error.message;
        } finally {
          documentImportButton.disabled = false;
        }
      }

      async function loadDocumentVersions(documentId) {
        state.selectedDocumentId = documentId;
        const payload = await requestJson(`/documents/${documentId}/versions`);
        state.versions = payload.versions || [];
        state.selectedVersionId = state.versions.length ? state.versions[0].id : null;
        state.chunks = [];
        state.selectedChunkId = null;
        if (state.selectedVersionId) {
          await loadVersionChunks(state.selectedVersionId);
        }
      }

      async function loadVersionChunks(versionId) {
        state.selectedVersionId = versionId;
        const payload = await requestJson(`/versions/${versionId}/chunks`);
        state.chunks = payload.chunks || [];
        state.selectedChunkId = state.chunks.length ? state.chunks[0].id : null;
      }

      async function refreshPaperCards() {
        const payload = await requestJson("/paper-cards");
        state.paperCards = payload.cards || [];
        state.paperCardsDir = payload.directory || "";
      }

      async function draftPaperCard() {
        if (!state.selectedVersionId) return;
        state.workflowStatus = "Preparing study card...";
        renderPanels();
        state.paperCard = await requestJson("/paper-cards/draft", {
          method: "POST",
          body: JSON.stringify({
            version_id: state.selectedVersionId,
            create_suggestion: false
          })
        });
        state.editedCardFields = { ...(state.paperCard.extracted_fields || {}) };
        state.editedFieldKeys = [];
        state.cardAccepted = false;
        state.cardNeedsReview = false;
        state.savedArtifact = null;
        state.workflowStatus = "Study card ready for review";
        await refreshSuggestions();
        await refreshRuns();
        renderPanels();
      }

      async function savePaperCard() {
        if (!state.paperCard || !state.cardAccepted) return;
        state.workflowStatus = "Saving study note...";
        renderPanels();
        const savedArtifact = await requestJson("/paper-cards/save", {
          method: "POST",
          body: JSON.stringify({
            title: state.paperCard.title,
            markdown: buildEditedPaperCardMarkdown(),
            suggestion_id: state.paperCard.suggestion_id
          })
        });
        state.savedArtifact = savedArtifact;
        state.workflowStatus = "Note saved";
        await refreshPaperCards();
        renderPanels();
      }

      async function previewEvidence() {
        previewButton.disabled = true;
        previewButton.textContent = "Previewing...";
        try {
          const query = queryInput.value.trim();
          if (!query) {
            throw new Error("Question is required.");
          }
          state.answer = null;
          state.cardAccepted = false;
          state.savedArtifact = null;
          state.workflowStatus = "Finding source evidence...";
          state.preview = await requestJson("/retrieval-preview", {
            method: "POST",
            body: JSON.stringify({
              query,
              mode: modeSelect.value,
              top_k: Number(topKInput.value || 3),
              ensure_embeddings: true
            })
          });
          state.workflowStatus = `${(state.preview.retrieved_chunks || []).length} evidence sections found`;
          generateButton.disabled = false;
          renderThread();
          renderPanels();
          switchTab("evidence");
          await refreshHealth();
        } catch (error) {
          messages.innerHTML += `
            <article class="message system">
              <div class="message-header"><span>Error</span><span>preview failed</span></div>
              <p class="answer-text">${escapeHtml(error.message)}</p>
            </article>
          `;
        } finally {
          previewButton.disabled = false;
          previewButton.textContent = "Preview evidence";
        }
      }

      async function runAsk(event) {
        event.preventDefault();
        generateButton.disabled = true;
        generateButton.textContent = "Generating...";
        try {
          const query = queryInput.value.trim();
          if (!query) {
            throw new Error("Question is required.");
          }
          const requestKey = currentRequestKey();
          if (state.answer && state.lastAnswerKey === requestKey) {
            state.workflowStatus = "Study card already generated for this request";
            renderPanels();
            return;
          }
          state.workflowStatus = "Generating study card...";
          state.answer = await requestJson("/ask", {
            method: "POST",
            body: JSON.stringify({
              query,
              mode: modeSelect.value,
              top_k: Number(topKInput.value || 3),
              ensure_embeddings: true,
              create_suggestion: true
            })
          });
          state.workflowStatus = "Study card generated";
          state.lastAnswerKey = requestKey;
          state.preview = null;
          state.reviewDecision = null;
          state.cardAccepted = false;
          state.cardNeedsReview = false;
          state.savedArtifact = null;
          await refreshSuggestions();
          await refreshRuns();
          renderThread();
          renderPanels();
          switchTab("evidence");
          await refreshHealth();
        } catch (error) {
          messages.innerHTML += `
            <article class="message system">
              <div class="message-header"><span>Error</span><span>request failed</span></div>
              <p class="answer-text">${escapeHtml(error.message)}</p>
            </article>
          `;
        } finally {
          generateButton.disabled = false;
          generateButton.textContent = "Generate from evidence";
        }
      }

      async function submitReview(decision) {
        if (!state.answer?.suggestion_id) return;
        const citationQuality = document.getElementById("citationQualitySelect")?.value || "not_checked";
        const completeness = document.getElementById("completenessSelect")?.value || "not_checked";
        const fieldCorrectness = document.getElementById("fieldCorrectnessSelect")?.value || "not_checked";
        const reviewNote = document.getElementById("reviewNoteInput")?.value || "";
        const decisionPayload = await requestJson(`/review/suggestions/${state.answer.suggestion_id}/decision`, {
          method: "POST",
          body: JSON.stringify({
            decision,
            reviewer: "local-user",
            note: [
              `citation_quality=${citationQuality}`,
              `completeness=${completeness}`,
              `field_correctness=${fieldCorrectness}`,
              reviewNote.trim()
            ].filter(Boolean).join("; ")
          })
        });
        state.reviewDecision = decisionPayload;
        state.workflowStatus = `Review recorded: ${decision}`;
        await refreshSuggestions();
        renderPanels();
        await refreshHealth();
      }

      async function createEvalCaseFromReview() {
        if (!state.answer || !state.reviewDecision) return;
        const answer = state.answer.answer || {};
        const retrievedChunks = state.answer.retrieved_chunks || [];
        const expectedFields = answer.extracted_fields || {};
        await requestJson("/evaluation-cases", {
          method: "POST",
          body: JSON.stringify({
            source: "review",
            query: state.answer.query,
            task: taskModeSelect.value,
            expected_behavior: state.reviewDecision.decision === "reject" ? "failure_regression" : "corrected_output",
            expected_fields: expectedFields,
            review_note: state.reviewDecision.note,
            retrieved_chunk_ids: retrievedChunks.map((chunk) => chunk.chunk_id),
            ai_run_id: state.answer.ai_run_id,
            suggestion_id: state.answer.suggestion_id
          })
        });
        state.workflowStatus = "Evaluation case created";
        await refreshEvalCases();
        renderPanels();
        switchTab("eval");
      }

      function switchTab(tabName) {
        state.activeTab = tabName;
        document.querySelectorAll(".tab").forEach((button) => {
          button.classList.toggle("active", button.dataset.tab === tabName);
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.hidden = panel.id !== `${tabName}Panel`;
        });
      }

      workflowForm.addEventListener("submit", runAsk);
      previewButton.addEventListener("click", previewEvidence);
      documentImportButton.addEventListener("click", importDocument);
      paperWorkspace.addEventListener("click", async (event) => {
        const chunkButton = event.target.closest("[data-workspace-chunk-id]");
        if (chunkButton) {
          state.selectedChunkId = chunkButton.dataset.workspaceChunkId;
          renderPanels();
          return;
        }
        const button = event.target.closest("[data-workspace-action]");
        if (!button) return;
        button.disabled = true;
        try {
          if (button.dataset.workspaceAction === "preview") {
            await previewEvidence();
          }
          if (button.dataset.workspaceAction === "generate") {
            await runAsk(new Event("submit"));
          }
          if (button.dataset.workspaceAction === "draft-card") {
            await draftPaperCard();
          }
          if (button.dataset.workspaceAction === "accept-card") {
            if (requiredMissingCount() === 0) {
              state.cardAccepted = true;
              state.cardNeedsReview = false;
              state.workflowStatus = "Card accepted";
              renderPanels();
            }
          }
          if (button.dataset.workspaceAction === "save-card") {
            await savePaperCard();
          }
          if (button.dataset.workspaceAction === "create-eval") {
            await createEvalCaseFromReview();
          }
        } finally {
          button.disabled = false;
        }
      });
      paperWorkspace.addEventListener("input", (event) => {
        const field = event.target.closest("[data-card-field]");
        if (!field) return;
        state.editedCardFields[field.dataset.cardField] = field.value;
        if (!state.editedFieldKeys.includes(field.dataset.cardField)) {
          state.editedFieldKeys.push(field.dataset.cardField);
        }
        if (state.cardAccepted) {
          state.cardAccepted = false;
          state.cardNeedsReview = true;
          state.workflowStatus = "Card changed; review required";
          renderPanels();
        }
      });
      paperWorkspace.addEventListener("change", (event) => {
        const field = event.target.closest("[data-card-field]");
        if (!field) return;
        renderPanels();
      });
      document.getElementById("refreshButton").addEventListener("click", async () => {
        await refreshHealth();
        await refreshSuggestions();
        await refreshCorpus();
        await refreshPaperCards();
        renderPanels();
      });
      document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => switchTab(button.dataset.tab));
      });
      reviewPanel.addEventListener("click", async (event) => {
        const evalButton = event.target.closest("[data-create-eval-case]");
        if (evalButton) {
          evalButton.disabled = true;
          try {
            await createEvalCaseFromReview();
          } finally {
            evalButton.disabled = false;
          }
          return;
        }
        const button = event.target.closest("[data-review]");
        if (!button) return;
        button.disabled = true;
        try {
          await submitReview(button.dataset.review);
        } finally {
          button.disabled = false;
        }
      });
      runsPanel.addEventListener("click", (event) => {
        const button = event.target.closest("[data-run-id]");
        if (!button) return;
        const run = state.runs.find((item) => item.id === button.dataset.runId);
        if (!run) return;
        state.answer = run.output;
        state.answer.ai_run_id = run.id;
        state.preview = null;
        renderThread();
        renderPanels();
        switchTab("evidence");
      });
      corpusPanel.addEventListener("click", async (event) => {
        const documentButton = event.target.closest("[data-document-id]");
        if (documentButton) {
          await loadDocumentVersions(documentButton.dataset.documentId);
          renderPanels();
          return;
        }
        const versionButton = event.target.closest("[data-version-id]");
        if (versionButton) {
          await loadVersionChunks(versionButton.dataset.versionId);
          renderPanels();
          return;
        }
        const chunkButton = event.target.closest("[data-chunk-id]");
        if (chunkButton) {
          state.selectedChunkId = chunkButton.dataset.chunkId;
          renderPanels();
        }
      });
      paperCardPanel.addEventListener("click", async (event) => {
        const button = event.target.closest("[data-paper-card-action]");
        if (!button) return;
        button.disabled = true;
        try {
          if (button.dataset.paperCardAction === "draft") {
            await draftPaperCard();
          }
          if (button.dataset.paperCardAction === "save") {
            await savePaperCard();
          }
        } finally {
          button.disabled = false;
        }
      });

      refreshHealth()
        .then(refreshSuggestions)
        .then(refreshRuns)
        .then(refreshEvalCases)
        .then(refreshCorpus)
        .then(refreshPaperCards)
        .then(renderPanels)
        .catch((error) => {
          healthStatus.textContent = "Database unavailable";
          messages.innerHTML = `
            <article class="message system">
              <div class="message-header"><span>Error</span><span>startup check failed</span></div>
              <p class="answer-text">${escapeHtml(error.message)}</p>
            </article>
          `;
          renderPanels();
        });
    </script>
  </body>
</html>
"""

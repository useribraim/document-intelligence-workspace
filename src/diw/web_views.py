from __future__ import annotations

import html
import json
from pathlib import Path

from diw.db.models import AIRun, AISuggestion, SourceDocument
from diw.web_concepts import (
    CONCEPTS,
    TOOLTIP_CSS,
    TOOLTIP_JS,
    concept_payload,
    glossary_items,
    render_concept_text,
)

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
    return (
        """
      :root {
        color-scheme: light;
        --paper: #faf8f4;
        --panel: #fffdfa;
        --inset: #f4f1ea;
        --inset-2: #ebe6da;
        --ink: #1d1b16;
        --muted: #6b665c;
        --rule: #ddd8cc;
        --rule-strong: #b6b0a1;
        --link: #5a4fc4;
        --link-underline: #c6c0ee;
        --accent: #6d5bd0;
        --accent-soft: #ece9fa;
        --ref: #2f66c4;
        --circular: #a4693a;
        --tip-edge: #6d5bd0;
        --tip-lock: #b0821f;
        --tip-kicker: #7a72a6;
        --serif: "Iowan Old Style", Charter, "Palatino Linotype", Palatino,
          "Source Serif 4", Georgia, serif;
        --sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }
      * { box-sizing: border-box; }
      html { -webkit-text-size-adjust: 100%; }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--ink);
        background: var(--paper);
        font-family: var(--serif);
        font-size: 17px;
        line-height: 1.62;
      }
      a { color: var(--ref); }
      code {
        padding: .05em .3em;
        background: var(--inset);
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: .86em;
      }
      .layout {
        display: grid;
        grid-template-columns: 15.5rem minmax(0, 1fr);
        gap: 3rem;
        width: min(66rem, calc(100% - 3rem));
        margin: 0 auto;
        padding: 2rem 0 5rem;
        align-items: start;
      }
      .rail { position: sticky; top: 2rem; font-family: var(--sans); font-size: .8rem; }
      .rail-brand {
        display: block;
        margin-bottom: 1.6rem;
        color: var(--ink);
        font-family: var(--serif);
        font-size: 1rem;
        font-weight: 600;
        text-decoration: none;
        line-height: 1.3;
      }
      .rail h2 {
        margin: 0 0 .55rem;
        color: var(--muted);
        font-family: var(--sans);
        font-size: .66rem;
        font-weight: 600;
        letter-spacing: .09em;
        text-transform: uppercase;
      }
      .rail-nav { display: flex; flex-direction: column; gap: .1rem; margin-bottom: 1.9rem; }
      .rail-nav a {
        padding: .28rem .5rem;
        color: var(--muted);
        text-decoration: none;
      }
      .rail-nav a:hover { color: var(--ink); background: var(--inset); }
      .rail-nav a.is-current { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
      .rail-filter {
        width: 100%;
        margin-bottom: .5rem;
        padding: .38rem .5rem;
        border: 1px solid var(--rule);
        background: var(--panel);
        color: var(--ink);
        font: inherit;
        font-size: .78rem;
      }
      .rail-terms {
        display: flex;
        flex-direction: column;
        gap: .05rem;
        max-height: 17rem;
        overflow-y: auto;
        margin: 0;
        padding: 0;
        list-style: none;
      }
      .rail-terms a { display: block; padding: .2rem .5rem; color: var(--muted); }
      .rail-terms .concept { border-bottom: 0; }
      .rail-terms a:hover { background: var(--inset); }
      article { max-width: 40rem; }
      .kicker {
        margin: 0;
        color: var(--muted);
        font-family: var(--sans);
        font-size: .68rem;
        font-weight: 600;
        letter-spacing: .09em;
        text-transform: uppercase;
      }
      h1 {
        margin: .35rem 0 1.1rem;
        font-size: 1.85rem;
        font-weight: 600;
        line-height: 1.22;
        letter-spacing: -.005em;
      }
      h2 {
        margin: 2.6rem 0 .7rem;
        font-size: 1.24rem;
        font-weight: 600;
        line-height: 1.3;
      }
      h2 .num { margin-right: .5rem; color: var(--accent); font-variant-numeric: tabular-nums; }
      h3 { margin: 1.6rem 0 .4rem; font-size: 1.02rem; font-weight: 600; }
      p { margin: 0 0 .95rem; }
      .lede { color: var(--muted); font-size: 1.06rem; }
      .rule { height: 1px; margin: 2.4rem 0; border: 0; background: var(--rule); }
      table { width: 100%; margin: 1.1rem 0; border-collapse: collapse; font-size: .93rem; }
      caption {
        margin-bottom: .45rem;
        color: var(--muted);
        font-family: var(--sans);
        font-size: .66rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-align: left;
        text-transform: uppercase;
      }
      th, td { padding: .5rem .6rem; border-bottom: 1px solid var(--rule); text-align: left; }
      thead th {
        border-bottom: 1px solid var(--rule-strong);
        color: var(--muted);
        font-family: var(--sans);
        font-size: .7rem;
        font-weight: 600;
        letter-spacing: .04em;
      }
      td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
      tbody tr:last-child td { border-bottom: 0; }
      .figure {
        margin: 1.3rem 0;
        padding: .9rem 1rem;
        border: 1px solid var(--rule);
        background: var(--panel);
      }
      .figure figcaption {
        margin-bottom: .5rem;
        color: var(--muted);
        font-family: var(--sans);
        font-size: .66rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
      }
      .caveat {
        margin: 1.3rem 0;
        padding: .1rem 0 .1rem 1rem;
        border-left: 3px solid var(--tip-lock);
        color: var(--muted);
        font-size: .95rem;
      }
      .caveat p:last-child { margin-bottom: 0; }
      blockquote {
        margin: .5rem 0 0;
        padding: .1rem 0 .1rem .9rem;
        border-left: 2px solid var(--rule-strong);
        color: var(--ink);
        font-size: .95rem;
      }
      .meta {
        margin: 0;
        color: var(--muted);
        font-family: var(--sans);
        font-size: .76rem;
      }
      .colophon {
        max-width: 40rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid var(--rule);
        color: var(--muted);
        font-family: var(--sans);
        font-size: .76rem;
      }
      form.ask { display: flex; gap: .5rem; margin: 1.2rem 0 .6rem; }
      form.ask input {
        flex: 1;
        min-width: 0;
        padding: .55rem .65rem;
        border: 1px solid var(--rule-strong);
        background: var(--panel);
        color: var(--ink);
        font: inherit;
        font-size: .97rem;
      }
      form.ask button {
        padding: .55rem 1rem;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper);
        font-family: var(--sans);
        font-size: .82rem;
        font-weight: 600;
        cursor: pointer;
      }
      form.ask button:hover { background: #33302a; }
      .examples { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.4rem; }
      .examples button {
        padding: .25rem .5rem;
        border: 1px solid var(--rule);
        background: var(--panel);
        color: var(--muted);
        font-family: var(--sans);
        font-size: .74rem;
        cursor: pointer;
      }
      .examples button:hover { border-color: var(--rule-strong); color: var(--ink); }
      .cite {
        color: var(--ref);
        font-family: var(--sans);
        font-size: .78em;
        font-weight: 600;
        text-decoration: none;
        vertical-align: super;
        cursor: help;
      }
      .cite:hover { text-decoration: underline; }
      .trace-list { margin: .6rem 0 0; font-family: var(--sans); font-size: .82rem; }
      .trace-list div {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: .32rem 0;
        border-bottom: 1px solid var(--rule);
      }
      .trace-list div:last-child { border-bottom: 0; }
      .trace-list dt { color: var(--muted); }
      .trace-list dd { margin: 0; font-variant-numeric: tabular-nums; }
      .refusal { color: var(--circular); }
      #result { display: none; }
      @media (max-width: 900px) {
        .layout { grid-template-columns: 1fr; gap: 1.5rem; padding-top: 1.2rem; }
        .rail { position: static; }
        .rail-terms { max-height: 11rem; }
      }
    """
        + TOOLTIP_CSS
    )


def _rail(current: str, sections: list[tuple[str, str]]) -> str:
    nav = "".join(
        f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor, label in sections
    )
    pages = [("/", "Overview"), ("/demo", "Read-only demo"), ("/evidence", "Measured evidence")]
    page_links = "".join(
        '<a href="{href}"{cls}>{label}</a>'.format(
            href=href,
            cls=' class="is-current"' if href == current else "",
            label=html.escape(label),
        )
        for href, label in pages
    )
    terms = "".join(
        f'<li><a class="concept" href="#concept-{slug}" data-concept="{slug}">'
        f"{html.escape(title)}</a></li>"
        for slug, title in glossary_items()
    )
    return f"""
        <aside class="rail">
          <a class="rail-brand" href="/">Document Intelligence Workspace</a>
          <h2>Pages</h2>
          <nav class="rail-nav">{page_links}</nav>
          <h2>Contents</h2>
          <nav class="rail-nav">{nav}</nav>
          <h2>Glossary &middot; {len(CONCEPTS)}</h2>
          <input class="rail-filter" type="search" placeholder="Filter concepts…"
            aria-label="Filter concepts" id="termFilter">
          <ul class="rail-terms" id="termList">{terms}</ul>
          <h2 style="margin-top:1.9rem">Access</h2>
          <nav class="rail-nav"><a href="/signin">Google sign-in</a></nav>
        </aside>
    """


def _page(
    *,
    title: str,
    current: str,
    sections: list[tuple[str, str]],
    body: str,
    extra_head: str = "",
    extra_script: str = "",
) -> str:
    return f"""<!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="description" content="A source-cited research assistant with inspectable evidence.">
        <title>{html.escape(title)}</title>
        <style>{_public_page_styles()}</style>{extra_head}
      </head>
      <body>
        <div class="layout">
          {_rail(current, sections)}
          <main>{body}</main>
        </div>
        <script>window.__DIW_CONCEPTS__ = {concept_payload()};</script>
        <script>{TOOLTIP_JS}</script>
        <script>
          (function () {{
            const filter = document.getElementById("termFilter");
            const list = document.getElementById("termList");
            if (!filter || !list) return;
            filter.addEventListener("input", function () {{
              const needle = filter.value.trim().toLowerCase();
              list.querySelectorAll("li").forEach(function (row) {{
                const hit = row.textContent.toLowerCase().includes(needle);
                row.style.display = hit ? "" : "none";
              }});
            }});
          }})();
        </script>{extra_script}
      </body>
    </html>
    """


_LANDING_SECTIONS = [
    ("what", "1 · What this is"),
    ("pipeline", "2 · How an answer is produced"),
    ("measured", "3 · What was measured"),
    ("unknown", "4 · What is not yet known"),
    ("boundaries", "5 · Reading boundaries"),
]


def _public_landing_html() -> str:
    prose = {
        "intro": "This service answers questions over a fixed corpus of machine-learning "
        "papers and shows the evidence behind every sentence it produces. It exists to "
        "make one thing checkable: whether a cited passage actually supports the claim "
        "attached to it.",
        "intro2": "Concepts underlined in violet open an explanation on hover, and those "
        "explanations link onward to their own prerequisites. Hold the cursor still to "
        "pin one open; press Escape to close the chain.",
        "what1": "Three surfaces share one deployment. The [[extractive-answer|read-only "
        "demo]] needs no account and holds no tenant data. The measured-evidence page "
        "records what has and has not been established. The authenticated routes enforce "
        "[[tenant-isolation|tenant isolation]] and [[oidc|Google OIDC]], and [[fail-closed|fail closed]] "
        "when scope cannot be resolved.",
        "what2": "The demo corpus is bundled and synthetic. It is not the persistent "
        "workflow, and nothing written here should be read as a claim that it is.",
        "pipe1": "A question is tokenised and scored against every [[chunk|chunk]] in the "
        "corpus by two signals: a [[lexical-score|lexical score]] over shared tokens, and a "
        "local hash-vector [[cosine-similarity|similarity]] score. The public hash vector is "
        "not semantic; the separate retrieval pilot used a real embedding model.",
        "pipe2": "The two rankings are merged by [[rrf|reciprocal rank fusion]], which "
        "discards raw scores and keeps only positions. The top chunks become candidate "
        "evidence.",
        "pipe3": "Generation is deterministic and [[extractive-answer|extractive]]: the "
        "response is assembled from retrieved sentences rather than written freely. Each "
        "emitted quote passes [[citation-validation|citation validation]] — a "
        "character-exact check against the [[exact-span|span]] it came from — before the "
        "response is returned. If no chunk clears the threshold, the system reports "
        "[[insufficient-evidence|insufficient evidence]] instead of answering.",
        "meas1": "Both arms below ran over the same [[frozen-set|frozen set]]: 40 "
        "questions across ten papers, 23 of them carrying gold evidence annotations.",
        "meas2": "The combined arm had higher point estimates, especially for [[mrr|MRR]], "
        "but a paired bootstrap over the 23 gold-scored questions produced intervals that "
        "include zero for both MRR and [[recall-at-5|Recall@5]]. This is an inconclusive "
        "pilot, not a retrieval-superiority result.",
        "meas3": "The repository preserves the point estimates, the per-question trace, and "
        "the uncertainty analysis. It does not attribute an effect to RRF or semantic "
        "embeddings in isolation because the complete factorial comparison is not published.",
        "unk1": "No human-calibrated accuracy figure and no "
        "[[inter-annotator-agreement|agreement]] statistic appears anywhere on this site, "
        "because the labels behind them do not exist yet. A calibration instrument is "
        "ready — 28 evidence seeds expressed as five controlled variants, yielding 112 aligned "
        "[[claim-citation-pair|claim-citation pairs]] per annotator — and both label sets are "
        "blank. It is a rubric-development bank, not an independent 140-item study.",
        "unk2": "Closing that gap requires a new independent-item study, two people labelling "
        "under a frozen rubric, and raw agreement plus [[cohens-kappa|Cohen's kappa]] reported "
        "before adjudication. A low value would be published as readily as a high one.",
        "bound1": "[[citation-validation|Citation validation]] is a string check. It "
        "proves a quote is real; it cannot tell you whether the quote supports the claim. "
        "That judgement needs a human [[support-label|support label]], and automated "
        "verifier output is never reported as though it were one.",
        "bound2": "The public demo runs no external model request and exposes no write "
        "tool. Deterministic behaviour there is a property of that route, not evidence "
        "about the authenticated path.",
    }
    r = render_concept_text
    body = f"""
          <p class="kicker">Evidence-first research assistant</p>
          <h1>Answers over ML papers, with citations you can inspect</h1>
          <p class="lede">{r(prose["intro"])}</p>
          <p class="meta">{r(prose["intro2"])}</p>
          <hr class="rule">

          <h2 id="what"><span class="num">1</span>What this is</h2>
          <p>{r(prose["what1"])}</p>
          <p>{r(prose["what2"])}</p>

          <h2 id="pipeline"><span class="num">2</span>How an answer is produced</h2>
          <p>{r(prose["pipe1"])}</p>
          <p>{r(prose["pipe2"])}</p>
          <p>{r(prose["pipe3"])}</p>
          <p class="meta"><a href="/demo">Run this pipeline on bundled synthetic excerpts &rarr;</a></p>

          <h2 id="measured"><span class="num">3</span>What was measured</h2>
          <p>{r(prose["meas1"])}</p>
          <table>
            <caption>Frozen retrieval comparison &middot; 40 questions</caption>
            <thead>
              <tr>
                <th>Configuration</th>
                <th class="num">Recall@5</th>
                <th class="num">MRR</th>
                <th class="num">Gold citation recall</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Hashing + weighted fusion</td>
                <td class="num">0.2609</td><td class="num">0.1935</td><td class="num">0.1087</td>
              </tr>
              <tr>
                <td>Semantic embeddings + RRF</td>
                <td class="num">0.2826</td><td class="num">0.3022</td><td class="num">0.2536</td>
              </tr>
            </tbody>
          </table>
          <p>{r(prose["meas2"])}</p>
          <p>{r(prose["meas3"])}</p>

          <h2 id="unknown"><span class="num">4</span>What is not yet known</h2>
          <p>{r(prose["unk1"])}</p>
          <p>{r(prose["unk2"])}</p>
          <div class="caveat">
            <p>No human-calibrated accuracy or inter-annotator agreement number is
            published until those labels exist.</p>
          </div>

          <h2 id="boundaries"><span class="num">5</span>Reading boundaries</h2>
          <p>{r(prose["bound1"])}</p>
          <p>{r(prose["bound2"])}</p>

          <p class="colophon">
            Demo corpus is bundled and synthetic. The current source tree excludes paper text for
            the frozen evaluation; the manifest records canonical versions, licences, and SHA-256
            hashes so a locally obtained corpus can be verified. A rewrite of historical Git
            revisions cannot retract existing clones, forks, caches, or mirrors.
          </p>
    """
    return _page(
        title="Document Intelligence Workspace",
        current="/",
        sections=_LANDING_SECTIONS,
        body=body,
    )


_DEMO_SECTIONS = [
    ("ask", "1 · Ask the corpus"),
    ("response", "2 · Response"),
    ("trace", "3 · Execution trace"),
]


def _public_demo_html() -> str:
    examples = "".join(
        f'<button type="button" data-question="{html.escape(question)}">'
        f"{html.escape(question)}</button>"
        for question in PUBLIC_DEMO_EXAMPLES
    )
    r = render_concept_text
    intro = r(
        "Every answer here is [[extractive-answer|extractive]] and deterministic: the "
        "same question returns the same response, with no external model request and no "
        "write tool reachable. Quotes are checked character-for-character against the "
        "retrieved [[chunk|chunks]] by [[citation-validation|citation validation]] before "
        "anything is returned."
    )
    hint = r(
        "Citation markers are hoverable — they open the [[exact-span|exact span]] the "
        "claim rests on, together with that chunk's retrieval scores. Ask something the "
        "corpus does not cover to see [[insufficient-evidence|the refusal path]]."
    )
    body = f"""
          <p class="kicker">Public, read-only, and synthetic</p>
          <h1>Ask six bundled synthetic ML-paper excerpts</h1>
          <p class="lede">{intro}</p>
          <p class="meta">This preview is not the ten-paper frozen evaluation corpus. It exists
          to make the read-only retrieval, citation, and trace surfaces inspectable without an
          account or an external model request.</p>
          <p class="meta">{hint}</p>

          <h2 id="ask"><span class="num">1</span>Ask the corpus</h2>
          <form class="ask" id="askForm">
            <input id="question" maxlength="500" required
              value="{html.escape(PUBLIC_DEMO_EXAMPLES[0])}" aria-label="Question">
            <button type="submit">Find cited evidence</button>
          </form>
          <div class="examples">{examples}</div>

          <section id="result" aria-live="polite">
            <h2 id="response"><span class="num">2</span>Response</h2>
            <div id="answer"></div>
            <div id="citations"></div>
            <h2 id="trace"><span class="num">3</span>Execution trace</h2>
            <dl class="trace-list" id="traceList"></dl>
            <p class="meta" style="margin-top:.8rem">
              The public route cannot create suggestions, tasks, reviews, or agent runs.
            </p>
          </section>
          <p id="pending" class="meta">Submit a question to see the response, its cited
          spans, and the execution trace.</p>
    """
    script = """
        <script>
          const form = document.getElementById("askForm");
          const question = document.getElementById("question");
          const result = document.getElementById("result");
          const pending = document.getElementById("pending");
          const answer = document.getElementById("answer");
          const citations = document.getElementById("citations");
          const traceList = document.getElementById("traceList");
          const esc = (value) => String(value ?? "")
            .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

          document.querySelectorAll(".examples button").forEach((button) => {
            button.addEventListener("click", () => {
              question.value = button.dataset.question;
              question.focus();
            });
          });

          function renderAnswer(text, marks) {
            let out = esc(text);
            marks.forEach((label) => {
              const token = "[" + label + "]";
              out = out.split(esc(token)).join(
                '<a class="cite" href="#cite-' + esc(label) + '">' + esc(label) + "</a>"
              );
            });
            return out;
          }

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            pending.style.display = "none";
            result.style.display = "block";
            answer.innerHTML = "<p class='meta'>Searching evidence…</p>";
            citations.innerHTML = "";
            traceList.innerHTML = "";
            try {
              const response = await fetch("/demo/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: question.value })
              });
              const payload = await response.json();
              if (!response.ok) throw new Error(payload.detail || "Request failed");
              const labels = payload.answer.citations.map((c) => c.label);
              const refused = payload.answer.insufficient_evidence;
              answer.innerHTML = "<p" + (refused ? " class='refusal'" : "") + ">" +
                renderAnswer(payload.answer.answer, labels) + "</p>";
              const scores = {};
              (payload.retrieved_chunks || []).forEach((chunk) => {
                scores[chunk.chunk_id] = chunk;
              });
              citations.innerHTML = payload.answer.citations.map((citation) => {
                const chunk = scores[citation.chunk_id] || {};
                const fmt = (value) => Number(value).toFixed(4);
                const detail = [
                  chunk.score !== undefined ? "fused " + fmt(chunk.score) : null,
                  chunk.lexical_score !== undefined ? "lexical " + fmt(chunk.lexical_score) : null,
                  chunk.vector_score !== undefined ? "vector " + fmt(chunk.vector_score) : null
                ].filter(Boolean).join(" · ");
                return '<div class="figure" id="cite-' + esc(citation.label) + '">' +
                  "<figcaption>" + esc(citation.label) + " · " +
                  esc(citation.heading_path.join(" › ")) + "</figcaption>" +
                  "<blockquote>" + esc(citation.quote) + "</blockquote>" +
                  (detail ? '<p class="meta" style="margin-top:.6rem">' + esc(detail) + "</p>" : "") +
                  "</div>";
              }).join("");
              const fields = [
                ["Access", payload.trace.access],
                ["Corpus", payload.trace.corpus + " (" + payload.trace.corpus_documents + " documents)"],
                ["Retrieval", payload.trace.retrieval],
                ["Reranker", payload.trace.reranker],
                ["Generation", payload.trace.generation],
                ["External model request", payload.trace.external_model_request ? "yes" : "no"],
                ["Writes performed", payload.trace.writes_performed],
                ["Latency", payload.trace.latency_ms + " ms"]
              ];
              traceList.innerHTML = fields.map(([label, value]) =>
                "<div><dt>" + esc(label) + "</dt><dd>" + esc(value) + "</dd></div>"
              ).join("");
            } catch (error) {
              answer.innerHTML = "<p class='refusal'>" + esc(error.message) + "</p>";
            }
          });
        </script>
    """
    return _page(
        title="Read-only demo · Document Intelligence Workspace",
        current="/demo",
        sections=_DEMO_SECTIONS,
        body=body,
        extra_script=script,
    )


_EVIDENCE_SECTIONS = [
    ("retrieval", "1 · Frozen retrieval comparison"),
    ("validated", "2 · What is validated"),
    ("open", "3 · What remains open"),
]


def _public_evidence_html() -> str:
    r = render_concept_text
    lede = r(
        "Every row below is tied to an artifact in the repository. Statements about "
        "deployed, measured, and human-validated work are kept apart on purpose, because "
        "collapsing them is the most common way a project overstates itself."
    )
    retr = r(
        "Both arms ran over the same [[frozen-set|frozen 40-question set]], comparing the "
        "local hashing baseline against semantic [[embedding|embeddings]] with "
        "[[rrf|reciprocal rank fusion]]."
    )
    note = r(
        "The combined configuration produced higher point estimates, but paired bootstrap "
        "intervals for Recall@5 and MRR include zero. It is an observed, inconclusive pilot; "
        "not evidence that either component improved retrieval."
    )
    open_1 = r(
        "Human calibration is incomplete. The current bank holds 28 source seeds expressed as "
        "five variants and 112 aligned [[claim-citation-pair|claim-citation pairs]] per annotator; "
        "both label sets are blank. It must not be treated as 112 independent units. Until a new "
        "independent study is complete, no "
        "[[inter-annotator-agreement|agreement]] figure and no "
        "[[cohens-kappa|kappa]] can be reported."
    )
    open_2 = r(
        "Automated [[citation-validation|citation validation]] and deterministic verifier "
        "output are diagnostics. They are never presented as human "
        "[[support-label|support labels]]."
    )
    body = f"""
          <p class="kicker">Claim-to-evidence boundary</p>
          <h1>Measured results and current limitations</h1>
          <p class="lede">{lede}</p>
          <hr class="rule">

          <h2 id="retrieval"><span class="num">1</span>Frozen retrieval comparison</h2>
          <p>{retr}</p>
          <table>
            <caption>40 questions · 23 with gold evidence annotations</caption>
            <thead>
              <tr>
                <th>Configuration</th>
                <th class="num">Recall@5</th>
                <th class="num">MRR</th>
                <th class="num">Gold citation recall</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Hashing + weighted fusion</td>
                <td class="num">0.2609</td><td class="num">0.1935</td><td class="num">0.1087</td>
              </tr>
              <tr>
                <td>Semantic embeddings + RRF</td>
                <td class="num">0.2826</td><td class="num">0.3022</td><td class="num">0.2536</td>
              </tr>
            </tbody>
          </table>
          <p>{note}</p>

          <h2 id="validated"><span class="num">2</span>What is validated</h2>
          <table>
            <caption>Capability status</caption>
            <thead><tr><th>Capability</th><th>Status</th><th>Validated scope</th></tr></thead>
            <tbody>
              <tr>
                <td>Retrieval comparison</td><td>Measured</td>
                <td>23 gold-scored questions; combined-arm point estimates were higher, but paired
                uncertainty intervals include zero and establish no improvement.</td>
              </tr>
              <tr>
                <td>Cloud Run</td><td>Live-validated</td>
                <td>Public read-only routes serve; protected routes reject missing or
                unscoped identity.</td>
              </tr>
              <tr>
                <td>Google OIDC</td><td>Live-validated</td>
                <td>Deployed verifier accepted a Google ID token for the configured
                audience; tenant membership resolved server-side.</td>
              </tr>
              <tr>
                <td>Vertex AI</td><td>Live-validated</td>
                <td>Cloud Run Job recorded model provenance, validated an exact citation,
                and refused an unsupported query.</td>
              </tr>
              <tr>
                <td>Google ADK</td><td>Live-validated</td>
                <td>ReAct-style coordinator delegated retrieval and citation verification
                to two ADK specialists; the run recorded 7 model calls, 13.82 s,
                69.14 output tokens/s, and $0.002634 estimated model cost.</td>
              </tr>
              <tr>
                <td>MCP</td><td>Client-validated</td>
                <td>External SDK client discovered both read-only tools; cross-tenant
                access and tenant-argument injection failed safely.</td>
              </tr>
              <tr>
                <td>Human calibration</td><td>Incomplete</td>
                <td>Controlled 28-seed variant bank and two blank templates exist. It is not an
                independent human-evaluation sample; no agreement figure is published.</td>
              </tr>
            </tbody>
          </table>

          <h2 id="open"><span class="num">3</span>What remains open</h2>
          <p>{open_1}</p>
          <p>{open_2}</p>
          <div class="caveat">
            <p>No human-calibrated accuracy or inter-annotator agreement number is
            published until those labels exist.</p>
          </div>

          <p class="colophon">
            The public interactive demo does not use Vertex or ADK: those proofs are
            separate Cloud Run Jobs over bundled sources. MCP validation uses a local
            stdio process, not a remote deployment.
          </p>
    """
    return _page(
        title="Measured evidence · Document Intelligence Workspace",
        current="/evidence",
        sections=_EVIDENCE_SECTIONS,
        body=body,
    )



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

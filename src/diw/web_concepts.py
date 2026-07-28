"""Concept graph and nested-tooltip assets for the public document pages.

Concept bodies use ``[[slug]]`` or ``[[slug|display text]]`` to link to another
concept. Links are resolved in the browser, so a concept can reference an
ancestor: the tooltip layer marks such a reference as circular instead of
spawning an infinite chain.
"""

from __future__ import annotations

import html
import json
import re

_LINK_PATTERN = re.compile(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]")


def _rrf_widget() -> str:
    return """
    <figure class="widget">
      <figcaption>Two ranked lists fused into one</figcaption>
      <svg viewBox="0 0 340 150" role="img" aria-label="Reciprocal rank fusion diagram">
        <text x="8" y="16" class="wl">lexical</text>
        <text x="120" y="16" class="wl">vector</text>
        <text x="252" y="16" class="wl">fused</text>
        <g class="wbox">
          <rect x="8" y="26" width="86" height="22" rx="2"/>
          <rect x="8" y="54" width="86" height="22" rx="2"/>
          <rect x="8" y="82" width="86" height="22" rx="2"/>
        </g>
        <g class="wbox">
          <rect x="120" y="26" width="86" height="22" rx="2"/>
          <rect x="120" y="54" width="86" height="22" rx="2"/>
          <rect x="120" y="82" width="86" height="22" rx="2"/>
        </g>
        <g class="wbox wfused">
          <rect x="252" y="26" width="80" height="22" rx="2"/>
          <rect x="252" y="54" width="80" height="22" rx="2"/>
          <rect x="252" y="82" width="80" height="22" rx="2"/>
        </g>
        <text x="16" y="41" class="wt">chunk A</text>
        <text x="16" y="69" class="wt">chunk C</text>
        <text x="16" y="97" class="wt">chunk B</text>
        <text x="128" y="41" class="wt">chunk C</text>
        <text x="128" y="69" class="wt">chunk A</text>
        <text x="128" y="97" class="wt">chunk D</text>
        <text x="260" y="41" class="wt">chunk A</text>
        <text x="260" y="69" class="wt">chunk C</text>
        <text x="260" y="97" class="wt">chunk B</text>
        <path d="M100 37 L246 37" class="warr"/>
        <path d="M212 37 L246 60" class="warr"/>
        <text x="8" y="126" class="wn">A scores 1/61 + 1/62; C scores 1/62 + 1/61.</text>
        <text x="8" y="140" class="wn">Neither list alone puts A and C on top.</text>
      </svg>
    </figure>
    """


def _recall_widget() -> str:
    return """
    <figure class="widget">
      <figcaption>Recall@5 over one question</figcaption>
      <svg viewBox="0 0 340 108" role="img" aria-label="Recall at five diagram">
        <g class="wbox">
          <rect x="8" y="22" width="60" height="26" rx="2"/>
          <rect x="74" y="22" width="60" height="26" rx="2"/>
          <rect x="140" y="22" width="60" height="26" rx="2"/>
          <rect x="206" y="22" width="60" height="26" rx="2"/>
          <rect x="272" y="22" width="60" height="26" rx="2"/>
        </g>
        <rect x="140" y="22" width="60" height="26" rx="2" class="whit"/>
        <text x="8" y="16" class="wl">rank 1 &hellip; 5</text>
        <text x="30" y="39" class="wt">&middot;</text>
        <text x="96" y="39" class="wt">&middot;</text>
        <text x="158" y="39" class="wt gold">gold</text>
        <text x="228" y="39" class="wt">&middot;</text>
        <text x="294" y="39" class="wt">&middot;</text>
        <text x="8" y="70" class="wn">The gold chunk sits at rank 3, so this question counts</text>
        <text x="8" y="84" class="wn">as a hit. Recall@5 is the share of questions that hit.</text>
        <text x="8" y="98" class="wn">Rank 3 also fixes its MRR contribution: 1/3.</text>
      </svg>
    </figure>
    """


def _kappa_widget() -> str:
    return """
    <figure class="widget" data-widget="kappa">
      <figcaption>Cohen's kappa against chance agreement</figcaption>
      <svg viewBox="0 0 340 96" role="img" aria-label="Kappa scale">
        <line x1="12" y1="52" x2="328" y2="52" class="waxis"/>
        <g class="wtick">
          <line x1="12" y1="47" x2="12" y2="57"/>
          <line x1="170" y1="47" x2="170" y2="57"/>
          <line x1="328" y1="47" x2="328" y2="57"/>
        </g>
        <text x="12" y="72" class="wn">0.0</text>
        <text x="158" y="72" class="wn">0.5</text>
        <text x="310" y="72" class="wn">1.0</text>
        <text x="12" y="20" class="wl">kappa</text>
        <circle cx="170" cy="52" r="5" class="wdot" data-kappa-dot/>
        <text x="150" y="38" class="wt" data-kappa-flag>0.60</text>
      </svg>
      <label class="wctl">
        <span>observed agreement</span>
        <input type="range" min="50" max="100" value="80" step="1" data-kappa-input>
        <output data-kappa-po>0.80</output>
      </label>
      <p class="wnote">
        With chance agreement fixed at 0.50, kappa is
        <span data-kappa-formula>(0.80 &minus; 0.50) / (1 &minus; 0.50) = 0.60</span>.
        Raw agreement alone flatters a two-label task: at 0.50 observed, kappa is 0,
        because coin flips would have done as well.
      </p>
    </figure>
    """


CONCEPTS: dict[str, dict] = {
    "hybrid-retrieval": {
        "title": "Hybrid retrieval",
        "gloss": (
            "Scores every chunk twice — once by word overlap, once by vector "
            "similarity — and merges the two rankings."
        ),
        "body": [
            (
                "A [[lexical-score|lexical score]] finds chunks that reuse the "
                "question's words. A semantic [[embedding|embedding]] can find chunks that "
                "mean the same thing in different words. The public demo's local hash vector "
                "does not have that semantic capability."
            ),
            (
                "The two rankings are merged by [[rrf|reciprocal rank fusion]]. The "
                "public demo runs this path entirely in-process: no database, and no "
                "external model call."
            ),
        ],
    },
    "lexical-score": {
        "title": "Lexical score",
        "gloss": (
            "A BM25 word-match score that rewards distinctive query terms while "
            "normalising for passage length."
        ),
        "body": [
            (
                "Tokens are lowercased and split on non-word characters, so "
                "<em>retrieval</em> and <em>Retrieval,</em> match while "
                "<em>retrieve</em> does not. BM25 gives more weight to terms that occur "
                "in fewer chunks and avoids rewarding a long passage merely for repeating a term."
            ),
            (
                "It is precise and completely explainable, but blind to paraphrase. Only "
                "a semantic [[embedding|embedding]] model addresses that limitation."
            ),
        ],
    },
    "embedding": {
        "title": "Embedding",
        "gloss": (
            "A model-produced vector that can encode semantic similarity; not every "
            "vector representation has that property."
        ),
        "body": [
            (
                "The frozen comparison used OpenAI "
                "<code>text-embedding-3-small</code>. The public demo instead uses a "
                "local hashing embedder at 256 dimensions, which needs no API key and "
                "no network call. It hashes individual tokens, so passages with no shared "
                "tokens score zero even if they are paraphrases; it is not a semantic model."
            ),
            "Two embeddings are compared by [[cosine-similarity|cosine similarity]].",
        ],
    },
    "cosine-similarity": {
        "title": "Cosine similarity",
        "gloss": (
            "The cosine of the angle between two vectors: 1.0 when they point the "
            "same way, 0.0 when perpendicular."
        ),
        "body": [
            (
                "Because it measures direction rather than length, a long passage and "
                "a short one on the same topic still score highly. That is what makes "
                "it usable across [[chunk|chunks]] of uneven size."
            ),
        ],
    },
    "rrf": {
        "title": "Reciprocal rank fusion",
        "gloss": (
            "Merges ranked lists by summing 1/(k + rank), discarding each system's "
            "raw scores."
        ),
        "body": [
            (
                "Raw scores from different retrievers are not comparable — a 0.8 "
                "cosine and a 0.8 lexical overlap mean different things. RRF throws "
                "the scores away and keeps only positions, so no retriever's scale "
                "can dominate."
            ),
            (
                "A chunk ranked moderately by both systems can beat a chunk ranked "
                "first by one and missed by the other. That is the intended "
                "behaviour, and it is why fusion runs after "
                "[[hybrid-retrieval|both rankings]] exist."
            ),
            (
                "The published pilot compares a combined semantic-plus-RRF arm with a hashing "
                "baseline. Its uncertainty interval includes zero, and the repository does not "
                "attribute an effect to RRF in isolation."
            ),
        ],
        "widget": _rrf_widget(),
    },
    "chunk": {
        "title": "Chunk",
        "gloss": "One retrievable slice of a document, carrying its heading path and position.",
        "body": [
            (
                "Documents are split at roughly 650 characters with an 80-character "
                "overlap on the demo path. The overlap keeps a sentence that straddles "
                "a boundary reachable from either side."
            ),
            (
                "A chunk is the unit that gets cited. When the interface shows an "
                "[[exact-span|exact span]], that span lives inside one chunk."
            ),
        ],
    },
    "frozen-set": {
        "title": "Frozen evaluation set",
        "gloss": (
            "A fixed corpus, question list, and run, hashed so later comparisons "
            "measure the change rather than the inputs."
        ),
        "body": [
            (
                "The retrieval comparison uses 40 questions over ten ML papers, 23 of "
                "which carry gold evidence annotations. Freezing matters because an "
                "unfrozen benchmark lets every configuration quietly pick its own "
                "favourable inputs."
            ),
            (
                "Paper text is not redistributed here. The manifest records canonical "
                "versions, licences, and SHA-256 hashes, so a locally obtained corpus "
                "can be checked against the frozen one."
            ),
        ],
    },
    "recall-at-5": {
        "title": "Recall@5",
        "gloss": (
            "The share of questions whose gold chunk appears anywhere in the top "
            "five results."
        ),
        "body": [
            (
                "It asks a blunt question — did the right evidence survive into "
                "the shortlist at all — and ignores where in the shortlist it "
                "landed. [[mrr|MRR]] is the metric that cares about position."
            ),
            (
                "Absolute values are low here because the gold annotations demand an "
                "exact chunk, not merely the right paper."
            ),
        ],
        "widget": _recall_widget(),
    },
    "mrr": {
        "title": "Mean reciprocal rank",
        "gloss": "The average of 1/rank of the first correct result, across all questions.",
        "body": [
            (
                "A gold chunk at rank 1 contributes 1.0, at rank 2 contributes 0.5, at "
                "rank 4 contributes 0.25. Improvements at the top of the list move MRR "
                "far more than improvements at the bottom."
            ),
            (
                "The combined arm's MRR point estimate was 0.3022 versus 0.1935 for the "
                "baseline, while [[recall-at-5|Recall@5]] barely moved. The paired 95% interval "
                "for that MRR difference includes zero, so this pattern is hypothesis-generating "
                "rather than a confirmed ranking gain."
            ),
        ],
    },
    "gold-citation-recall": {
        "title": "Gold citation recall",
        "gloss": (
            "Of the citations the system actually emitted, the share matching an "
            "annotated gold chunk."
        ),
        "body": [
            (
                "This is stricter than [[recall-at-5|Recall@5]]: retrieval can surface "
                "the right chunk and the answer can still cite a different one. It "
                "measures the end of the pipeline rather than the middle."
            ),
        ],
    },
    "exact-span": {
        "title": "Exact span",
        "gloss": "The verbatim stretch of source text a citation points at — not a summary of it.",
        "body": [
            (
                "Every quote the demo renders is checked character-for-character "
                "against the retrieved [[chunk|chunk]] before the response is returned. A "
                "quote that does not appear verbatim fails "
                "[[citation-validation|validation]]."
            ),
            (
                "Spans are also the unit of human judgement: an annotator rates the "
                "span in front of them, never the paper it came from. Reading the whole "
                "paper inflates agreement, because the reader starts crediting "
                "citations for facts the span never states."
            ),
        ],
    },
    "citation-validation": {
        "title": "Citation validation",
        "gloss": (
            "A deterministic check that each emitted quote occurs verbatim in the "
            "evidence that was retrieved."
        ),
        "body": [
            (
                "It is a string check, not a judgement. It catches a fabricated or "
                "drifted quote, and it cannot tell you whether a real quote supports "
                "the claim attached to it — that needs a "
                "[[support-label|support label]] from a person."
            ),
            "Passing validation is therefore necessary, and nowhere near sufficient.",
        ],
    },
    "extractive-answer": {
        "title": "Extractive answer",
        "gloss": "A response assembled from retrieved sentences rather than generated freely.",
        "body": [
            (
                "The public demo is deterministic and extractive: the same question "
                "returns the same answer, with no model call leaving the container. "
                "That makes it safe to expose without a key, and it is why an answer "
                "sometimes reads as a near-copy of its own citation."
            ),
            (
                "The authenticated path can use Vertex generation instead. The public "
                "demo deliberately cannot."
            ),
        ],
    },
    "insufficient-evidence": {
        "title": "Insufficient evidence",
        "gloss": "The refusal state entered when no retrieved chunk clears the evidence threshold.",
        "body": [
            (
                "Refusing is a feature of the gate, not a failure of retrieval. A "
                "system that always answers cannot be scored on whether it should have."
            ),
            (
                "Ask the demo something the bundled corpus does not cover and this path "
                "fires. Whether a refusal was <em>correct</em> is a human judgement, "
                "recorded separately from whether it happened."
            ),
        ],
    },
    "claim-citation-pair": {
        "title": "Claim-citation pair",
        "gloss": "One asserted claim bound to one citation — the atomic unit of the calibration study.",
        "body": [
            (
                "Answers are decomposed into claims; each claim carries the citation "
                "the system attached to it. The annotator judges that binding, not the "
                "answer as a whole."
            ),
            (
                "The current V2 bank holds 112 aligned pairs per annotator across 28 repeated "
                "source seeds. Alignment makes [[inter-annotator-agreement|agreement]] computable, "
                "but those repeated variants are not independent observations."
            ),
        ],
    },
    "support-label": {
        "title": "Support label",
        "gloss": (
            "A human verdict on whether the exact span entails every material part "
            "of the claim."
        ),
        "body": [
            (
                "Five values: fully supported, partially supported, unsupported, "
                "contradicted, not applicable. The boundary that costs annotators most "
                "time is <em>fully</em> against <em>partially</em>, and it turns on "
                "what counts as a material qualifier."
            ),
            (
                "Labels apply to the [[exact-span|exact span]] alone. A span may be "
                "squarely on topic and still fail to establish the claim; relevance and "
                "support are recorded as separate fields for exactly that reason."
            ),
        ],
    },
    "inter-annotator-agreement": {
        "title": "Inter-annotator agreement",
        "gloss": "How often two independent people assign the same label to the same item.",
        "body": [
            (
                "Raw agreement is the plain percentage. It is easy to read and easy to "
                "overstate, because two annotators guessing at random already agree "
                "often. [[cohens-kappa|Cohen's kappa]] corrects for that."
            ),
            (
                "Both are reported before adjudication, with the confusion matrix and "
                "every disagreement's rationale. Discussing disagreements first and "
                "computing agreement afterwards would measure the conversation, not "
                "the rubric."
            ),
        ],
    },
    "cohens-kappa": {
        "title": "Cohen's kappa",
        "gloss": "Agreement corrected for chance: (observed &minus; expected) / (1 &minus; expected).",
        "body": [
            (
                "Kappa is 0 when two annotators do no better than their own label "
                "frequencies predict, and 1.0 when they match perfectly. It can go "
                "negative, which means systematically opposite readings of the rubric."
            ),
            (
                "A low value is a real result. The honest response is to publish it, "
                "read the confusion matrix, find the rubric boundary the disagreements "
                "cluster on, and relabel under a revised version — not to retune "
                "until the number improves."
            ),
            (
                "No kappa is published on this site, because the second "
                "[[claim-citation-pair|labelled pass]] does not exist yet."
            ),
        ],
        "widget": _kappa_widget(),
    },
    "tenant-isolation": {
        "title": "Tenant isolation",
        "gloss": (
            "Every query is bound to one tenant, so one account's documents cannot "
            "surface in another's results."
        ),
        "body": [
            (
                "The filter is applied in repository queries, agent retrieval, API "
                "actor checks, and the MCP tools. Enforcing it in one layer only would "
                "leave the others reachable."
            ),
            (
                "In the MCP server the tenant is fixed by process configuration. A "
                "model talking to that server cannot pass a tenant argument, because no "
                "tool accepts one."
            ),
        ],
    },
    "oidc": {
        "title": "OpenID Connect",
        "gloss": (
            "An identity layer over OAuth 2.0 returning a signed [[id-token|ID token]] "
            "that describes who signed in."
        ),
        "body": [
            (
                "The deployed service verifies a Google-issued token: signature against "
                "the published keys, issuer, expiry, and audience. Tenant membership is "
                "then resolved server-side from the verified subject."
            ),
            (
                "Membership is never read from the request body. A caller who could "
                "name their own tenant would have no isolation at all."
            ),
        ],
    },
    "id-token": {
        "title": "ID token",
        "gloss": "A signed JWT asserting the user's identity, issuer, audience, and expiry.",
        "body": [
            (
                "The audience claim is what matters here: a token minted for a "
                "different application is cryptographically valid and still rejected, "
                "because it was not issued for this client."
            ),
            (
                "Verification failures produce 401. Missing tenant scope produces 403 "
                "— a [[fail-closed|fail-closed]] default."
            ),
        ],
    },
    "fail-closed": {
        "title": "Fail closed",
        "gloss": "When authorization cannot be established, deny rather than fall back to open.",
        "body": [
            (
                "In production mode an unscoped data route returns 403 instead of "
                "serving unfiltered rows. The failure mode of a misconfiguration is "
                "then an outage, which is loud, rather than a leak, which is silent."
            ),
            (
                "The public demo routes are the deliberate exception: read-only, "
                "carrying no tenant data, exposing no write tool."
            ),
        ],
    },
}


def render_concept_text(text: str) -> str:
    """Turn ``[[slug|label]]`` markers into concept anchors."""

    def replace(match: re.Match[str]) -> str:
        slug = match.group(1)
        label = match.group(2) or CONCEPTS.get(slug, {}).get("title", slug)
        if slug not in CONCEPTS:
            return html.escape(label)
        return (
            f'<a class="concept" href="#concept-{slug}" '
            f'data-concept="{slug}">{html.escape(label)}</a>'
        )

    return _LINK_PATTERN.sub(replace, text)


def concept_payload() -> str:
    """Serialise the concept graph for the browser tooltip layer."""
    payload = {
        slug: {
            "title": entry["title"],
            "gloss": render_concept_text(entry["gloss"]),
            "body": [render_concept_text(part) for part in entry["body"]],
            "widget": entry.get("widget", ""),
        }
        for slug, entry in CONCEPTS.items()
    }
    return json.dumps(payload, separators=(",", ":"))


def glossary_items() -> list[tuple[str, str]]:
    return sorted(
        ((slug, entry["title"]) for slug, entry in CONCEPTS.items()),
        key=lambda item: item[1].lower(),
    )


TOOLTIP_CSS = """
  .concept {
    color: var(--link);
    text-decoration: none;
    border-bottom: 1px solid var(--link-underline);
    cursor: help;
  }
  .concept:hover { border-bottom-color: var(--link); }
  .concept.is-circular {
    color: var(--circular);
    border-bottom-style: dotted;
    border-bottom-color: var(--circular);
  }
  .concept.is-circular::after { content: "\\21ba"; font-size: .82em; vertical-align: super; }
  .tip {
    position: absolute;
    z-index: 60;
    width: min(23rem, calc(100vw - 2rem));
    max-height: 26rem;
    overflow: hidden auto;
    border: 1px solid var(--tip-edge);
    border-left-width: 3px;
    background: var(--panel);
    box-shadow: 0 10px 34px rgba(28, 26, 22, .13);
    pointer-events: none;
    opacity: 0;
    transition: opacity .1s ease;
  }
  .tip.is-shown { opacity: 1; }
  .tip.is-locked { pointer-events: auto; border-left-color: var(--tip-lock); }
  .tip-progress {
    position: sticky;
    top: 0;
    left: 0;
    height: 2px;
    width: 0;
    background: var(--tip-lock);
  }
  .tip.is-locked .tip-progress { width: 100% !important; opacity: .35; }
  .tip-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: .75rem;
    padding: .85rem 1rem .1rem;
  }
  .tip-kicker {
    margin: 0;
    color: var(--tip-kicker);
    font-family: var(--sans);
    font-size: .66rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
  }
  .tip-title { margin: .1rem 0 0; font-size: 1.06rem; font-weight: 600; line-height: 1.25; }
  .tip-pin { flex: none; color: var(--tip-edge); font-size: .8rem; opacity: .4; }
  .tip.is-locked .tip-pin { color: var(--tip-lock); opacity: 1; }
  .tip-gloss {
    margin: .5rem 0 0;
    padding: 0 1rem;
    color: var(--muted);
    font-style: italic;
    font-size: .92rem;
  }
  .tip-body { padding: .2rem 1rem 1rem; }
  .tip-body p { margin: .7rem 0 0; font-size: .94rem; }
  .widget {
    margin: .9rem 0 0;
    padding: .6rem .7rem .7rem;
    border: 1px solid var(--rule);
    background: var(--inset);
  }
  .widget figcaption {
    margin-bottom: .35rem;
    color: var(--tip-kicker);
    font-family: var(--sans);
    font-size: .64rem;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
  }
  .widget svg { display: block; width: 100%; height: auto; }
  .widget .wl { fill: var(--muted); font-family: var(--sans); font-size: 9px; }
  .widget .wt { fill: var(--ink); font-family: var(--sans); font-size: 9px; }
  .widget .wt.gold { fill: var(--tip-lock); font-weight: 700; }
  .widget .wn { fill: var(--muted); font-family: var(--sans); font-size: 8.5px; }
  .widget .wbox rect { fill: #fff; stroke: var(--rule); }
  .widget .wfused rect { fill: var(--inset-2); }
  .widget .whit { fill: none; stroke: var(--tip-lock); stroke-width: 1.5; }
  .widget .warr { stroke: var(--rule-strong); fill: none; stroke-width: 1; }
  .widget .waxis, .widget .wtick line { stroke: var(--rule-strong); }
  .widget .wdot { fill: var(--tip-lock); }
  .wctl {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: .5rem;
    margin-top: .5rem;
    color: var(--muted);
    font-family: var(--sans);
    font-size: .72rem;
  }
  .wctl input { width: 100%; accent-color: var(--tip-lock); }
  .wctl output { color: var(--ink); font-variant-numeric: tabular-nums; }
  .wnote { margin: .5rem 0 0; color: var(--muted); font-size: .8rem; }
  @media (prefers-reduced-motion: reduce) {
    .tip { transition: none; }
  }
"""


TOOLTIP_JS = """
(function () {
  const CONCEPTS = window.__DIW_CONCEPTS__ || {};
  const HOVER_DELAY = 300;
  const LOCK_DELAY = 520;
  const GRACE = 260;
  const MAX_DEPTH = 4;

  const chain = [];
  let hoverTimer = null;
  let graceTimer = null;
  let pointer = { x: 0, y: 0 };

  document.addEventListener("mousemove", function (event) {
    pointer = { x: event.clientX, y: event.clientY };
  });

  function ancestors() {
    return chain.map(function (entry) { return entry.slug; });
  }

  function markCircular(root) {
    const open = ancestors();
    root.querySelectorAll(".concept").forEach(function (link) {
      link.classList.toggle("is-circular", open.indexOf(link.dataset.concept) !== -1);
    });
  }

  function buildTip(slug, depth) {
    const data = CONCEPTS[slug];
    if (!data) return null;
    const tip = document.createElement("div");
    tip.className = "tip";
    tip.dataset.slug = slug;
    tip.setAttribute("role", "tooltip");
    const kicker = depth === 0 ? "Concept" : "Nested concept \\u00b7 Level " + depth;
    tip.innerHTML =
      '<div class="tip-progress"></div>' +
      '<div class="tip-head"><div><p class="tip-kicker">' + kicker + "</p>" +
      '<h3 class="tip-title"></h3></div><span class="tip-pin">\\u25c9</span></div>' +
      '<p class="tip-gloss"></p><div class="tip-body"></div>';
    tip.querySelector(".tip-title").textContent = data.title;
    tip.querySelector(".tip-gloss").innerHTML = data.gloss;
    const body = tip.querySelector(".tip-body");
    body.innerHTML = data.body.map(function (part) { return "<p>" + part + "</p>"; }).join("");
    if (data.widget) body.insertAdjacentHTML("beforeend", data.widget);
    return tip;
  }

  function place(tip, anchor) {
    const rect = anchor.getBoundingClientRect();
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    const margin = 12;
    let left = rect.left + window.scrollX;
    if (left + width > window.scrollX + window.innerWidth - margin) {
      left = window.scrollX + window.innerWidth - width - margin;
    }
    left = Math.max(window.scrollX + margin, left);
    let top = rect.bottom + window.scrollY + 8;
    if (rect.bottom + height + 8 > window.innerHeight && rect.top > height) {
      top = rect.top + window.scrollY - height - 8;
    }
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function armLock(entry) {
    if (entry.lockTimer || entry.tip.classList.contains("is-locked")) return;
    const bar = entry.tip.querySelector(".tip-progress");
    requestAnimationFrame(function () {
      bar.style.transition = "width " + LOCK_DELAY + "ms linear";
      bar.style.width = "100%";
    });
    entry.lockTimer = window.setTimeout(function () {
      entry.tip.classList.add("is-locked");
      entry.lockTimer = null;
    }, LOCK_DELAY);
  }

  function disarmLock(entry) {
    if (entry.lockTimer) {
      window.clearTimeout(entry.lockTimer);
      entry.lockTimer = null;
    }
    const bar = entry.tip.querySelector(".tip-progress");
    bar.style.transition = "none";
    bar.style.width = "0";
  }

  function inCorridor(point) {
    const last = chain[chain.length - 1];
    if (!last) return false;
    const rect = last.tip.getBoundingClientRect();
    const pad = 90;
    return point.x >= rect.left - pad && point.x <= rect.right + pad &&
      point.y >= rect.top - pad && point.y <= rect.bottom + pad;
  }

  function overChain(point) {
    return chain.some(function (entry) {
      const rect = entry.tip.getBoundingClientRect();
      return point.x >= rect.left && point.x <= rect.right &&
        point.y >= rect.top && point.y <= rect.bottom;
    });
  }

  function pruneTo(depth) {
    while (chain.length > depth) {
      const entry = chain.pop();
      disarmLock(entry);
      entry.tip.remove();
    }
    markCircular(chain.length ? chain[chain.length - 1].tip : document.body);
  }

  function closeAll() {
    pruneTo(0);
  }

  function scheduleDismiss() {
    window.clearTimeout(graceTimer);
    graceTimer = window.setTimeout(function () {
      if (overChain(pointer) || inCorridor(pointer)) return;
      closeAll();
    }, GRACE);
  }

  function open(slug, anchor, depth) {
    if (depth > MAX_DEPTH) depth = MAX_DEPTH;
    pruneTo(depth);
    const tip = buildTip(slug, depth);
    if (!tip) return;
    document.body.appendChild(tip);
    place(tip, anchor);
    requestAnimationFrame(function () { tip.classList.add("is-shown"); });
    const entry = { slug: slug, tip: tip, depth: depth, lockTimer: null };
    chain.push(entry);
    markCircular(tip);
    tip.addEventListener("mouseenter", function () {
      window.clearTimeout(graceTimer);
      armLock(entry);
    });
    tip.addEventListener("mouseleave", scheduleDismiss);
    bind(tip, depth + 1);
  }

  function bind(root, depth) {
    root.querySelectorAll(".concept").forEach(function (link) {
      link.addEventListener("mouseenter", function () {
        window.clearTimeout(graceTimer);
        window.clearTimeout(hoverTimer);
        if (link.classList.contains("is-circular")) return;
        hoverTimer = window.setTimeout(function () {
          open(link.dataset.concept, link, depth);
        }, HOVER_DELAY);
      });
      link.addEventListener("mouseleave", function () {
        window.clearTimeout(hoverTimer);
        scheduleDismiss();
      });
      link.addEventListener("click", function (event) {
        event.preventDefault();
        if (link.classList.contains("is-circular")) return;
        open(link.dataset.concept, link, depth);
        const entry = chain[chain.length - 1];
        if (entry) entry.tip.classList.add("is-locked");
      });
    });
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeAll();
  });

  document.addEventListener("click", function (event) {
    if (!chain.length) return;
    if (event.target.closest(".tip") || event.target.closest(".concept")) return;
    closeAll();
  });

  document.addEventListener("input", function (event) {
    const input = event.target.closest("[data-kappa-input]");
    if (!input) return;
    const widget = input.closest("[data-widget='kappa']");
    const po = input.value / 100;
    const pe = 0.5;
    const kappa = (po - pe) / (1 - pe);
    widget.querySelector("[data-kappa-po]").textContent = po.toFixed(2);
    widget.querySelector("[data-kappa-formula]").innerHTML =
      "(" + po.toFixed(2) + " \\u2212 0.50) / (1 \\u2212 0.50) = " + kappa.toFixed(2);
    const x = 12 + Math.max(0, Math.min(1, kappa)) * 316;
    widget.querySelector("[data-kappa-dot]").setAttribute("cx", x);
    const flag = widget.querySelector("[data-kappa-flag]");
    flag.setAttribute("x", Math.min(x - 10, 300));
    flag.textContent = kappa.toFixed(2);
  });

  bind(document, 0);
})();
"""

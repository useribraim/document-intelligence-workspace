from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

SUPPORT_LABELS = {
    "fully_supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "not_applicable",
}
FAILURE_MODES = {
    "overgeneralization",
    "unsupported_specificity",
    "missing_qualification",
    "citation_misattribution",
    "claim_bundling",
    "retrieval_miss",
    "out_of_scope",
}


def annotation_key(record: dict) -> str:
    return "|".join(
        str(record.get(field) or "")
        for field in (
            "source_run_id",
            "question_id",
            "review_type",
            "claim_id",
            "citation_id",
        )
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


class AnnotationDecision(BaseModel):
    record_key: str
    annotator_id: str = Field(min_length=1, max_length=80)
    answer_completeness: str | None = None
    refusal_appropriate: bool | None = None
    source_exists: bool | None = None
    citation_relevant: str | None = None
    support_label: str | None = None
    support_rationale: str | None = Field(default=None, max_length=1_000)
    failure_mode: str | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> AnnotationDecision:
        if self.answer_completeness not in {None, "complete", "incomplete", "not_applicable"}:
            raise ValueError("invalid answer_completeness")
        if self.citation_relevant not in {None, "yes", "no"}:
            raise ValueError("invalid citation_relevant")
        if self.support_label not in SUPPORT_LABELS | {None}:
            raise ValueError("invalid support_label")
        if self.failure_mode not in FAILURE_MODES | {None}:
            raise ValueError("invalid failure_mode")
        return self


def create_annotation_app(
    *,
    input_path: Path,
    output_path: Path,
    default_annotator_id: str,
) -> FastAPI:
    source_records = load_jsonl(input_path)
    source_by_key = {annotation_key(record): record for record in source_records}
    if len(source_by_key) != len(source_records):
        raise ValueError("annotation packet contains duplicate record keys")

    if output_path.is_file():
        saved_records = load_jsonl(output_path)
        if {annotation_key(record) for record in saved_records} != set(source_by_key):
            raise ValueError("existing output does not match the input annotation packet")
        records = saved_records
    else:
        records = [dict(record) for record in source_records]

    app = FastAPI(title="DIW Human Annotation", version="1.0")

    @app.get("/", response_class=HTMLResponse)
    def annotation_page():
        return HTMLResponse(ANNOTATION_HTML)

    @app.get("/api/state")
    def state():
        completed = sum(
            record.get("annotation_status") == "completed_human" for record in records
        )
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "default_annotator_id": default_annotator_id,
            "completed": completed,
            "total": len(records),
            "records": records,
        }

    @app.post("/api/decisions")
    def save_decision(decision: AnnotationDecision):
        record = next(
            (item for item in records if annotation_key(item) == decision.record_key),
            None,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="annotation record not found")
        if record.get("review_type") == "answer_level":
            if decision.answer_completeness is None:
                raise HTTPException(status_code=422, detail="answer_completeness is required")
            record["answer_completeness"] = decision.answer_completeness
            record["refusal_appropriate"] = decision.refusal_appropriate
        elif record.get("review_type") == "claim_citation":
            if (
                decision.source_exists is None
                or decision.citation_relevant is None
                or decision.support_label is None
                or not (decision.support_rationale or "").strip()
            ):
                raise HTTPException(
                    status_code=422,
                    detail="source, relevance, support label, and rationale are required",
                )
            if (
                decision.support_label not in {"fully_supported", "not_applicable"}
                and decision.failure_mode is None
            ):
                raise HTTPException(
                    status_code=422,
                    detail="failure_mode is required for non-fully-supported claims",
                )
            record["source_exists"] = decision.source_exists
            record["citation_relevant"] = decision.citation_relevant
            record["support_label"] = decision.support_label
            record["support_rationale"] = decision.support_rationale.strip()
            record["failure_mode"] = decision.failure_mode
        else:
            raise HTTPException(status_code=422, detail="unknown review_type")

        record["annotator_id"] = decision.annotator_id
        record["annotation_status"] = "completed_human"
        record["annotated_at"] = datetime.now(UTC).isoformat()
        save_jsonl_atomic(output_path, records)
        completed = sum(
            item.get("annotation_status") == "completed_human" for item in records
        )
        return {"saved": True, "completed": completed, "total": len(records)}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DIW human-annotation interface.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_annotation_app(
            input_path=args.input,
            output_path=args.output,
            default_annotator_id=args.annotator_id,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


ANNOTATION_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DIW Human Annotation</title>
  <style>
    :root { color-scheme: light; --ink:#14213d; --muted:#61708a; --line:#d9e1ec; --accent:#0b6e4f; }
    body { margin:0; font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#f5f7fb; }
    main { max-width:1050px; margin:28px auto; padding:0 20px 60px; }
    header,.card { background:white; border:1px solid var(--line); border-radius:14px; padding:20px; box-shadow:0 8px 24px rgba(20,33,61,.06); }
    header { display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:18px; }
    h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:20px 0 6px; }
    .muted { color:var(--muted); } .progress { font-weight:700; white-space:nowrap; }
    .text { white-space:pre-wrap; background:#f7f9fc; border-left:4px solid #9eb1cc; padding:12px; border-radius:6px; }
    .evidence { border-left-color:#d08b22; } label { display:block; font-weight:650; margin:15px 0 5px; }
    select,input,textarea { width:100%; box-sizing:border-box; border:1px solid #b9c5d5; border-radius:8px; padding:10px; font:inherit; background:white; }
    textarea { min-height:84px; resize:vertical; } .row { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .actions { display:flex; gap:10px; margin-top:20px; }
    button { border:0; border-radius:9px; padding:11px 16px; font-weight:700; cursor:pointer; }
    button.primary { color:white; background:var(--accent); } button.secondary { background:#e8edf5; color:var(--ink); }
    .done { color:var(--accent); font-weight:700; } .error { color:#a51c30; font-weight:650; margin-top:12px; }
    @media (max-width:700px) { .row { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>Claim-to-evidence human review</h1><div class="muted" id="paths"></div></div>
    <div class="progress" id="progress">Loading…</div>
  </header>
  <section class="card">
    <div class="muted" id="recordMeta"></div>
    <h2>Question</h2><div class="text" id="question"></div>
    <div id="answerBlock"><h2>Answer</h2><div class="text" id="answer"></div></div>
    <div id="claimBlock"><h2>Claim</h2><div class="text" id="claim"></div></div>
    <div id="evidenceBlock"><h2>Cited evidence</h2><div class="text evidence" id="evidence"></div></div>

    <label for="annotator">Annotator ID</label><input id="annotator">
    <div id="answerFields">
      <label for="completeness">Answer completeness</label>
      <select id="completeness"><option value="">Choose…</option><option>complete</option><option>incomplete</option><option>not_applicable</option></select>
      <label for="refusal">Was refusal appropriate?</label>
      <select id="refusal"><option value="">Not applicable / unknown</option><option value="true">true</option><option value="false">false</option></select>
    </div>
    <div id="claimFields">
      <div class="row">
        <div><label for="sourceExists">Source exists</label><select id="sourceExists"><option value="">Choose…</option><option value="true">true</option><option value="false">false</option></select></div>
        <div><label for="relevant">Citation relevant</label><select id="relevant"><option value="">Choose…</option><option>yes</option><option>no</option></select></div>
      </div>
      <label for="support">Support label</label>
      <select id="support"><option value="">Choose…</option><option>fully_supported</option><option>partially_supported</option><option>unsupported</option><option>contradicted</option><option>not_applicable</option></select>
      <label for="rationale">One-sentence rationale</label><textarea id="rationale"></textarea>
      <label for="failure">Failure mode</label>
      <select id="failure"><option value="">None for fully supported / not applicable</option><option>overgeneralization</option><option>unsupported_specificity</option><option>missing_qualification</option><option>citation_misattribution</option><option>claim_bundling</option><option>retrieval_miss</option><option>out_of_scope</option></select>
    </div>
    <div class="error" id="error"></div>
    <div class="actions">
      <button class="secondary" id="previous">Previous</button>
      <button class="primary" id="save">Save and next pending</button>
      <button class="secondary" id="next">Next</button>
    </div>
  </section>
</main>
<script>
let state, index = 0;
const byId = id => document.getElementById(id);
const valueOrEmpty = value => value === null || value === undefined ? "" : String(value);
const setText = (id, value) => byId(id).textContent = valueOrEmpty(value);
const keyFor = r => [r.source_run_id,r.question_id,r.review_type,r.claim_id,r.citation_id].map(valueOrEmpty).join("|");
function boolValue(value) { return value === "" ? null : value === "true"; }
function firstPending() {
  const found = state.records.findIndex(r => r.annotation_status !== "completed_human");
  index = found >= 0 ? found : 0;
}
function render() {
  const r = state.records[index];
  setText("progress", `${state.completed}/${state.total} completed · record ${index+1}`);
  setText("paths", `Output: ${state.output_path}`);
  const sourceMeta = r.evidence_source?.chunk_id ? ` · ${r.evidence_source.chunk_id}` : "";
  setText("recordMeta", `${r.review_type} · ${r.question_id}${sourceMeta} · ${r.annotation_status || "pending"}`);
  setText("question", r.question || r.review_context?.query || "Question not stored in this row");
  setText("answer", r.review_context?.answer || r.automation_prefill?.answer || "");
  setText("claim", r.claim_text || "");
  setText("evidence", r.evidence_span || "");
  const isClaim = r.review_type === "claim_citation";
  byId("claimBlock").hidden = !isClaim; byId("evidenceBlock").hidden = !isClaim;
  byId("claimFields").hidden = !isClaim; byId("answerFields").hidden = isClaim;
  byId("annotator").value = r.annotator_id && !String(r.annotator_id).includes("pending") ? r.annotator_id : state.default_annotator_id;
  byId("completeness").value = valueOrEmpty(r.answer_completeness);
  byId("refusal").value = valueOrEmpty(r.refusal_appropriate);
  byId("sourceExists").value = valueOrEmpty(r.source_exists);
  byId("relevant").value = valueOrEmpty(r.citation_relevant);
  byId("support").value = valueOrEmpty(r.support_label);
  byId("rationale").value = valueOrEmpty(r.support_rationale);
  byId("failure").value = valueOrEmpty(r.failure_mode);
  setText("error", "");
}
async function load() {
  state = await (await fetch("/api/state")).json(); firstPending(); render();
}
async function save() {
  const r = state.records[index], isClaim = r.review_type === "claim_citation";
  const payload = {
    record_key:keyFor(r), annotator_id:byId("annotator").value,
    answer_completeness:isClaim ? null : byId("completeness").value || null,
    refusal_appropriate:isClaim ? null : boolValue(byId("refusal").value),
    source_exists:isClaim ? boolValue(byId("sourceExists").value) : null,
    citation_relevant:isClaim ? byId("relevant").value || null : null,
    support_label:isClaim ? byId("support").value || null : null,
    support_rationale:isClaim ? byId("rationale").value || null : null,
    failure_mode:isClaim ? byId("failure").value || null : null
  };
  const response = await fetch("/api/decisions",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  if (!response.ok) { const body=await response.json(); setText("error", body.detail || "Could not save"); return; }
  const result=await response.json(); state.completed=result.completed; Object.assign(r,payload,{annotation_status:"completed_human"});
  const next=state.records.findIndex((item,i)=>i>index && item.annotation_status!=="completed_human");
  const wrapped=state.records.findIndex(item=>item.annotation_status!=="completed_human");
  index=next>=0?next:(wrapped>=0?wrapped:index); render();
}
byId("previous").onclick=()=>{index=(index-1+state.total)%state.total;render();};
byId("next").onclick=()=>{index=(index+1)%state.total;render();};
byId("save").onclick=save;
load().catch(error=>setText("error",error));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

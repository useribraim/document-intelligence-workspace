# Cloud Run Deployment Evidence

## 2026-07-27 private smoke revision

- Project: `<redacted-project-id>`
- Region: `europe-west1`
- Service: `document-intelligence-workspace`
- Revision: `document-intelligence-workspace-00001-bph`
- Service URL: `https://document-intelligence-workspace-312779789755.europe-west1.run.app`
- Runtime identity: `diw-cloud-run-runtime@<redacted-project-id>.iam.gserviceaccount.com`
- Build identity: `diw-cloud-run-builder@<redacted-project-id>.iam.gserviceaccount.com`
- Limits: one vCPU, 512 MiB, minimum zero instances, maximum one instance
- Authenticated `GET /startup`: HTTP 200 with `{"status":"ready"}`
- Unauthenticated `GET /startup`: HTTP 403

This records a real container build, revision, route, and Cloud Run IAM boundary. The later public
revision adds application-authenticated Google OAuth.

## 2026-07-27 public OAuth revision

- Revision: `document-intelligence-workspace-00002-fjk`
- Public demo: `https://document-intelligence-workspace-312779789755.europe-west1.run.app/signin`
- OAuth client type: web application
- Authorized JavaScript origin: the canonical Cloud Run service URL
- OAuth audience: external, testing
- Test user: configured
- Public `GET /signin`, `/startup`, and `/health`: HTTP 200
- Missing-token `GET /auth/whoami`: HTTP 401
- Missing-token `POST /agent-runs`: HTTP 401
- Unscoped `GET /documents`: HTTP 403

The configured test user completed Google account selection on the deployed page. The browser
returned a Google ID token to `/auth/whoami`, the deployed verifier accepted its audience and
signature, and the page rendered the redacted verified account.

This releases both the public Cloud Run URL and live Google OAuth/OIDC configuration claims.

## 2026-07-27 public-demo revision

- Revision: `document-intelligence-workspace-00003-b49`
- Stable public URL:
  `https://document-intelligence-workspace-312779789755.europe-west1.run.app/`
- Public `GET /`, `/demo`, `/evidence`, `/signin`, `/startup`, and `/health`: HTTP 200
- Public `POST /demo/ask`: HTTP 200 with an exact source quote, valid citation check,
  `external_model_request=false`, and `writes_performed=0`
- Missing-token `GET /auth/whoami`: HTTP 401
- Missing-token `POST /agent-runs`: HTTP 401
- Unscoped `GET /documents`: HTTP 403
- Traffic: 100% assigned to this revision

The live browser walkthrough opened the landing page without an account, followed the primary
demo link, submitted the default question, and rendered the cited response plus execution trace.
No browser console warnings or errors were recorded. The demo is deliberately deterministic and
read-only; it validates the public product surface and retrieval/citation boundary, not persistent
Cloud SQL state or the complete authenticated write workflow.

## 2026-07-27 Vertex smoke revision

- Public service revision: `document-intelligence-workspace-00004-26q`
- Stable public URL:
  `https://document-intelligence-workspace-312779789755.europe-west1.run.app/`
- Cloud Run Job: `diw-vertex-smoke`
- Successful execution: `diw-vertex-smoke-z5m7b`
- Vertex API: enabled
- Runtime role: `roles/aiplatform.user`
- Runtime identity:
  `diw-cloud-run-runtime@<redacted-project-id>.iam.gserviceaccount.com`
- Models: `gemini-embedding-001` and `gemini-2.5-flash`
- Evidence: [`vertex-cloud-run-smoke.json`](../results/evidence/vertex-cloud-run-smoke.json)

The job used the deployed service image and completed real document/query embeddings, cited
Gemini generation, exact-quote citation validation, and unsupported-query refusal with no recorded
errors. It uses bundled synthetic documents and ephemeral SQLite; the public demo
remains deterministic and read-only.

## 2026-07-27 public-release calibration revision

- Revision: `document-intelligence-workspace-00007-5b6`
- Traffic: 100% assigned to this revision
- Public `GET /`, `/demo`, `/evidence`, `/signin`, and `/health`: HTTP 200
- Protected `GET /workspace` without identity: HTTP 403
- Revision scaling: minimum zero, maximum one instance
- Public landing page: 140 balanced calibration questions with human labels explicitly pending
- Calibration artifact: 112 aligned claim-citation pairs in each blinded annotator packet

The post-deployment smoke check confirmed the cleaned public copy, removed career-oriented wording,
kept protected routes fail-closed, and exposed no human agreement or accuracy claim.

## 2026-07-28 Google ADK revision

- Public service revision: `document-intelligence-workspace-00010-56m`
- Traffic: 100% assigned to this revision
- Source commit: `9e0967c`
- Cloud Run Job: `diw-adk-smoke`
- Successful execution: `diw-adk-smoke-bpdk8`
- Model: `gemini-2.5-flash`
- Workflow: ReAct-style coordinator with retrieval and citation-verification ADK specialists
- Recorded result: 7 model calls, 3,771 input tokens, 695 output tokens, 505 thinking tokens
- Measured workflow latency: 13,823.18 ms
- Aggregate output throughput: 69.14 tokens/s
- Estimated model cost: $0.00263435
- Evidence: [`adk-cloud-run-smoke.json`](../results/evidence/adk-cloud-run-smoke.json)

The first execution exposed a callback keyword-contract mismatch and failed before a model request.
The regression fix was covered by a keyword-invocation test, deployed in the next revision, and
the same job path then completed successfully. The cost is a pricing-snapshot estimate rather than
a Cloud Billing export.

## Cost controls

- Scale to zero
- One-instance maximum
- No Cloud SQL, VPC connector, GPU, or always-on worker
- `DIW 5 EUR monthly alert` budget scoped to this project at 50%, 90%, and 100%

The budget is an alert, not a hard spending cap.

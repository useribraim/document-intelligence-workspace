# Cloud Run Deployment Evidence

## 2026-07-27 private smoke revision

- Project: `gen-lang-client-0605013452`
- Region: `europe-west1`
- Service: `document-intelligence-workspace`
- Revision: `document-intelligence-workspace-00001-bph`
- Service URL: `https://document-intelligence-workspace-312779789755.europe-west1.run.app`
- Runtime identity: `diw-cloud-run-runtime@gen-lang-client-0605013452.iam.gserviceaccount.com`
- Build identity: `diw-cloud-run-builder@gen-lang-client-0605013452.iam.gserviceaccount.com`
- Limits: one vCPU, 512 MiB, minimum zero instances, maximum one instance
- Authenticated `GET /startup`: HTTP 200 with `{"status":"ready"}`
- Unauthenticated `GET /startup`: HTTP 403

This proves a real container build, revision, route, and Cloud Run IAM boundary. It is not yet the
public resume-demo gate: the Google OAuth web client and application-authenticated public revision
must succeed first.

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
signature, and the page rendered `Authenticated as useribraim@gmail.com`.

This releases both the public Cloud Run URL and live Google OAuth/OIDC configuration claims.

## 2026-07-27 recruiter-demo revision

- Revision: `document-intelligence-workspace-00003-b49`
- Stable resume URL:
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
read-only; it proves the public product surface and retrieval/citation boundary, not persistent
Cloud SQL state or the complete authenticated write workflow.

## 2026-07-27 Vertex smoke revision

- Public service revision: `document-intelligence-workspace-00004-26q`
- Stable resume URL:
  `https://document-intelligence-workspace-312779789755.europe-west1.run.app/`
- Cloud Run Job: `diw-vertex-smoke`
- Successful execution: `diw-vertex-smoke-z5m7b`
- Vertex API: enabled
- Runtime role: `roles/aiplatform.user`
- Runtime identity:
  `diw-cloud-run-runtime@gen-lang-client-0605013452.iam.gserviceaccount.com`
- Models: `gemini-embedding-001` and `gemini-2.5-flash`
- Evidence: [`vertex-cloud-run-smoke.json`](../results/integrations/vertex/vertex-cloud-run-smoke.json)

The job used the deployed service image and completed real document/query embeddings, cited
Gemini generation, exact-quote citation validation, and unsupported-query refusal with no recorded
errors. It uses bundled synthetic documents and ephemeral SQLite; the public recruiter demo
remains deterministic and read-only.

## Cost controls

- Scale to zero
- One-instance maximum
- No Cloud SQL, VPC connector, GPU, or always-on worker
- `DIW 5 EUR monthly alert` budget scoped to this project at 50%, 90%, and 100%

The budget is an alert, not a hard spending cap.

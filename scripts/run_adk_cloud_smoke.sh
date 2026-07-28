#!/usr/bin/env bash
set -euo pipefail

: "${DIW_GCP_PROJECT:?Set DIW_GCP_PROJECT.}"
: "${DIW_CLOUD_RUN_SERVICE:?Set DIW_CLOUD_RUN_SERVICE to the deployed source-image service.}"

DIW_GCLOUD_BIN="${DIW_GCLOUD_BIN:-gcloud}"
DIW_CLOUD_RUN_REGION="${DIW_CLOUD_RUN_REGION:-europe-west1}"
DIW_VERTEX_LOCATION="${DIW_VERTEX_LOCATION:-global}"
DIW_ADK_JOB="${DIW_ADK_JOB:-diw-adk-smoke}"
DIW_ADK_MODEL="${DIW_ADK_MODEL:-gemini-2.5-flash}"
DIW_RUNTIME_SERVICE_ACCOUNT="${DIW_RUNTIME_SERVICE_ACCOUNT:-diw-cloud-run-runtime@${DIW_GCP_PROJECT}.iam.gserviceaccount.com}"
DIW_ADK_QUERY="${DIW_ADK_QUERY:-What does the corpus say answer generation should do when evidence is insufficient?}"
DIW_ADK_EVIDENCE_OUT="${DIW_ADK_EVIDENCE_OUT:-results/evidence/adk-cloud-run-smoke.json}"

service_image="$(
  "${DIW_GCLOUD_BIN}" run services describe "${DIW_CLOUD_RUN_SERVICE}" \
    --project "${DIW_GCP_PROJECT}" \
    --region "${DIW_CLOUD_RUN_REGION}" \
    --format "value(spec.template.spec.containers[0].image)"
)"
if [[ -z "${service_image}" ]]; then
  echo "Could not resolve the deployed Cloud Run service image." >&2
  exit 1
fi

"${DIW_GCLOUD_BIN}" run jobs deploy "${DIW_ADK_JOB}" \
  --project "${DIW_GCP_PROJECT}" \
  --region "${DIW_CLOUD_RUN_REGION}" \
  --image "${service_image}" \
  --service-account "${DIW_RUNTIME_SERVICE_ACCOUNT}" \
  --command "diw-adk-research" \
  --args "${DIW_ADK_QUERY}" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${DIW_GCP_PROJECT},GOOGLE_CLOUD_LOCATION=${DIW_VERTEX_LOCATION},GOOGLE_GENAI_USE_VERTEXAI=true,ADK_MODEL=${DIW_ADK_MODEL}" \
  --cpu 1 \
  --memory 1Gi \
  --tasks 1 \
  --max-retries 0 \
  --task-timeout 10m \
  --quiet

execution_name="$(
  "${DIW_GCLOUD_BIN}" run jobs execute "${DIW_ADK_JOB}" \
    --project "${DIW_GCP_PROJECT}" \
    --region "${DIW_CLOUD_RUN_REGION}" \
    --wait \
    --format "value(metadata.name)"
)"
if [[ -z "${execution_name}" ]]; then
  echo "Cloud Run did not return an ADK execution name." >&2
  exit 1
fi

prefix="DIW_ADK_RESULT="
result_line=""
for _ in {1..12}; do
  result_line="$(
    "${DIW_GCLOUD_BIN}" logging read \
      "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"${DIW_ADK_JOB}\" AND labels.\"run.googleapis.com/execution_name\"=\"${execution_name}\" AND textPayload:\"${prefix}\"" \
      --project "${DIW_GCP_PROJECT}" \
      --freshness 2h \
      --order asc \
      --limit 1 \
      --format "value(textPayload)"
  )"
  if [[ "${result_line}" == "${prefix}"* ]]; then
    break
  fi
  sleep 5
done
if [[ "${result_line}" != "${prefix}"* ]]; then
  echo "The completed execution did not emit an ADK result." >&2
  exit 1
fi

mkdir -p "$(dirname "${DIW_ADK_EVIDENCE_OUT}")"
DIW_ADK_RESULT_JSON="${result_line#"${prefix}"}" \
  DIW_ADK_EVIDENCE_OUT="${DIW_ADK_EVIDENCE_OUT}" \
  .venv/bin/python -c \
  'import json, os, pathlib; path = pathlib.Path(os.environ["DIW_ADK_EVIDENCE_OUT"]); payload = json.loads(os.environ["DIW_ADK_RESULT_JSON"]); path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(f"saved: {path}")'

"${DIW_GCLOUD_BIN}" run jobs executions describe "${execution_name}" \
  --project "${DIW_GCP_PROJECT}" \
  --region "${DIW_CLOUD_RUN_REGION}" \
  --format "yaml(metadata.name,status.conditions,status.startTime,status.completionTime)"

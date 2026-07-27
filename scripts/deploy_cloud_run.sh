#!/usr/bin/env bash
set -euo pipefail

: "${DIW_GCP_PROJECT:?Set DIW_GCP_PROJECT to the target Google Cloud project ID.}"
: "${DIW_CLOUD_RUN_SERVICE:?Set DIW_CLOUD_RUN_SERVICE to the Cloud Run service name.}"

DIW_CLOUD_RUN_REGION="${DIW_CLOUD_RUN_REGION:-europe-west1}"
DIW_VERTEX_LOCATION="${DIW_VERTEX_LOCATION:-global}"
DIW_AUTH_MODE="${DIW_AUTH_MODE:-off}"
DIW_GCLOUD_BIN="${DIW_GCLOUD_BIN:-gcloud}"
DIW_BUILD_SERVICE_ACCOUNT="${DIW_BUILD_SERVICE_ACCOUNT:-diw-cloud-run-builder@${DIW_GCP_PROJECT}.iam.gserviceaccount.com}"
DIW_RUNTIME_SERVICE_ACCOUNT="${DIW_RUNTIME_SERVICE_ACCOUNT:-diw-cloud-run-runtime@${DIW_GCP_PROJECT}.iam.gserviceaccount.com}"

case "${DIW_AUTH_MODE}" in
  off)
    access_flag="--no-allow-unauthenticated"
    runtime_env="GOOGLE_CLOUD_PROJECT=${DIW_GCP_PROJECT},GOOGLE_CLOUD_LOCATION=${DIW_VERTEX_LOCATION},AUTH_MODE=off"
    ;;
  google)
    : "${GOOGLE_OAUTH_CLIENT_ID:?Set GOOGLE_OAUTH_CLIENT_ID for Google OIDC deployment.}"
    access_flag="--allow-unauthenticated"
    runtime_env="GOOGLE_CLOUD_PROJECT=${DIW_GCP_PROJECT},GOOGLE_CLOUD_LOCATION=${DIW_VERTEX_LOCATION},AUTH_MODE=google,GOOGLE_OAUTH_CLIENT_ID=${GOOGLE_OAUTH_CLIENT_ID}"
    ;;
  *)
    echo "DIW_AUTH_MODE must be 'off' or 'google'." >&2
    exit 2
    ;;
esac

"${DIW_GCLOUD_BIN}" run deploy "${DIW_CLOUD_RUN_SERVICE}" \
  --project "${DIW_GCP_PROJECT}" \
  --source . \
  --build-service-account "projects/${DIW_GCP_PROJECT}/serviceAccounts/${DIW_BUILD_SERVICE_ACCOUNT}" \
  --service-account "${DIW_RUNTIME_SERVICE_ACCOUNT}" \
  --region "${DIW_CLOUD_RUN_REGION}" \
  "${access_flag}" \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --min 0 \
  --max 1 \
  --min-instances 0 \
  --max-instances 1 \
  --set-env-vars "${runtime_env}"

"${DIW_GCLOUD_BIN}" run services describe "${DIW_CLOUD_RUN_SERVICE}" \
  --project "${DIW_GCP_PROJECT}" \
  --region "${DIW_CLOUD_RUN_REGION}" \
  --format "value(status.url)"

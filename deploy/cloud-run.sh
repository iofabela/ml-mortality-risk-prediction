#!/usr/bin/env bash
# Deploy the mortality-risk service to Google Cloud Run (scale to zero, free tier).
#
# Usage (from the repo root):
#   deploy/cloud-run.sh setup      # one time: enable APIs, IAM role, $1 budget alert
#   deploy/cloud-run.sh deploy     # build the Dockerfile on Cloud Build and deploy
#   deploy/cloud-run.sh test       # call /health and /predict on the live URL
#   deploy/cloud-run.sh teardown   # delete the service, the images and the build bucket
#
# Settings (env vars): PROJECT_ID (default: current gcloud project), REGION, SERVICE,
#   BUDGET (in the billing account currency, e.g. 1USD or 20MXN)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-mortality-risk}"
BUDGET="${BUDGET:-1USD}"
[[ -n "$PROJECT_ID" ]] || { echo "Set PROJECT_ID or run: gcloud config set project <id>"; exit 1; }

url() {
  gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
    --format 'value(status.url)'
}

setup() {
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com compute.googleapis.com billingbudgets.googleapis.com \
    cloudresourcemanager.googleapis.com cloudbilling.googleapis.com storage.googleapis.com \
    --project "$PROJECT_ID"

  # Source deploys build with the Compute Engine default service account.
  local number; number=$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet \
    --member "serviceAccount:${number}-compute@developer.gserviceaccount.com" \
    --role roles/run.builder >/dev/null

  # Email alert at 50% and 100% of $BUDGET (an alert only: it does not stop billing).
  local billing; billing=$(gcloud billing projects describe "$PROJECT_ID" \
    --format 'value(billingAccountName)' | sed 's|billingAccounts/||')
  gcloud billing budgets create --billing-account "$billing" \
    --display-name "$SERVICE budget" --budget-amount "$BUDGET" \
    --filter-projects "projects/$PROJECT_ID" \
    --threshold-rule percent=0.5 --threshold-rule percent=1.0
}

deploy() {
  gcloud run deploy "$SERVICE" --source . --project "$PROJECT_ID" --region "$REGION" \
    --allow-unauthenticated --quiet --memory 1Gi --cpu 1 --min-instances 0 --max-instances 1
}

test_service() {
  local base; base=$(url)
  echo "URL: $base"
  echo "GET /health";  curl -s "$base/health"; echo
  echo "POST /predict"
  curl -s -X POST -H "Content-Type: application/json" \
    -d '{"age": 59.81, "sex": "male", "dzgroup": "coma", "num_co": 3, "edu": 4,
         "income": "under_$11k", "scoma": 26, "race": "hispanic", "hday": 21,
         "diabetes": 1, "dementia": 0, "ca": "no", "dnr_before_adm": 0,
         "meanbp": 105, "wblc": 20.1, "hrt": 114, "resp": 12, "temp": 40.59,
         "pafi": 415, "alb": 4.3, "bili": 0.6, "crea": 2.0, "sod": 147, "ph": 7.64,
         "glucose": 402, "bun": 56, "urine": 1350, "adlsc": 0}' \
    "$base/predict"; echo
  echo "API reference: $base/docs"
}

teardown() {
  gcloud run services delete "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --quiet || true
  gcloud artifacts repositories delete cloud-run-source-deploy \
    --project "$PROJECT_ID" --location "$REGION" --quiet || true
  # Uploaded source archives (bucket name depends on the gcloud version)
  for b in "gs://run-sources-${PROJECT_ID}-${REGION}" "gs://${PROJECT_ID}_cloudbuild"; do
    gcloud storage rm -r "$b" --quiet 2>/dev/null && echo "Deleted bucket [$b]." || true
  done
  echo "Done. To remove everything (incl. the budget filter target): gcloud projects delete $PROJECT_ID"
}

case "${1:-}" in
  setup) setup ;;
  deploy) deploy ;;
  test) test_service ;;
  teardown) teardown ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac

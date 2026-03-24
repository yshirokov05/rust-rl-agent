#!/bin/bash
# Operation Cloud Bridge: GCS Sync Protocol
# Target Bucket: gs://rust-rl-vanguard-storage

BUCKET="gs://rust-rl-vanguard-storage"

echo "Beginning consolidation of 5.44 GB dataset..."
gcloud storage cp -r "C:\Projects\rust-rl-agent\shared-data" "$BUCKET/shared-data"

echo "Syncing Vanguard_Brain_V1.pth..."
gcloud storage cp "C:\Projects\rust-rl-agent\models\Vanguard_Brain_V1.pth" "$BUCKET/models/"

echo "Syncing Cloud Sync Protocol..."
gcloud storage cp "C:\Projects\rust-rl-agent\ai-agent\model_wrapper.py" "$BUCKET/deployment/"

echo "Status: SUCCESS. Operation Cloud Bridge verified at $(date)"

#!/usr/bin/env bash
#
# Sync the Vijil-private models the test suite loads, and nothing else.
#
# The bucket holds every model Vijil trains or serves — including 70-135 GB
# adversarial LLMs — so a blanket `aws s3 sync` of models/vijil/ pulls ~220 GB.
# The detector tests need four classifiers totalling ~2.8 GB.
#
# Each model is stored as one directory per HF revision:
#
#     models/vijil/<name>/<revision-sha>/{config.json,model.safetensors,...}
#
# but `vijil_dome.detectors.utils.hf_model.resolve_model_path` looks for a
# *flat* $VIJIL_MODEL_DIR/vijil/<name>/config.json. So we resolve the latest
# revision (newest config.json) and sync that revision's files into the flat
# path. A recursive sync of the model directory would land the weights one
# level too deep, where every detector silently falls back to the HF Hub and
# every `_model_available()` skipif marks its tests skipped.
#
# Usage: scripts/sync_test_models.sh [dest]   (default: $VIJIL_MODEL_DIR or /models)

set -euo pipefail

BUCKET="${VIJIL_MODEL_BUCKET:-vijil-inference}"
PREFIX="${VIJIL_MODEL_PREFIX:-models/vijil}"
DEST="${1:-${VIJIL_MODEL_DIR:-/models}}"

# Models the test suite actually loads. Keep in sync with the detector
# defaults these mirror:
#   prompt-injection-v5-20260827        pi_hf_mbert.DEFAULT_VIJIL_INFERENCE_PI_MODEL
#   vijil_dome_toxic_content_detection  toxicity_mbert.DEFAULT_VIJIL_INFERENCE_TOXICITY_MODEL
#   stereotype-eeoc-detector            stereotype_eeoc.DEFAULT_VIJIL_INFERENCE_STEREOTYPE_MODEL
#   prompt-harmfulness-detector         prompt_harmfulness.ModernBertPromptHarmfulnessModel
# Every other vijil/* repo in the bucket is either unreferenced by the tests
# (pi_deberta_finetuned_11122024) or a serving-side LLM.
MODELS=(
    prompt-injection-v5-20260827
    vijil_dome_toxic_content_detection
    stereotype-eeoc-detector
    prompt-harmfulness-detector
)

# Training leftovers some revisions ship alongside the weights. None are read
# by `from_pretrained`, and optimizer state alone can double the transfer.
EXCLUDES=(
    --exclude "*/*"              # revision subdirectories and .cache/
    --exclude "optimizer.pt"
    --exclude "scheduler.pt"
    --exclude "scaler.pt"
    --exclude "rng_state.pth"
    --exclude "trainer_state.json"
    --exclude "training_args.bin"
    --exclude "*.msgpack"        # flax weights
    --exclude "*.h5"             # tensorflow weights
)

# Resolve <name> to the S3 prefix holding its latest revision: the directory
# of the most recently modified config.json anywhere under the model. Handles
# both layouts in the bucket — a revision subdirectory, or files sitting
# directly under the model name.
latest_revision_prefix() {
    local name="$1" key
    key=$(aws s3api list-objects-v2 \
        --bucket "$BUCKET" \
        --prefix "$PREFIX/$name/" \
        --query 'sort_by(Contents[?ends_with(Key, `/config.json`)], &LastModified)[-1].Key' \
        --output text)
    if [ -z "$key" ] || [ "$key" = "None" ]; then
        echo "no config.json found under s3://$BUCKET/$PREFIX/$name/" >&2
        return 1
    fi
    dirname "$key"
}

sync_model() {
    local name="$1" src
    src=$(latest_revision_prefix "$name")
    echo "==> $name  <-  s3://$BUCKET/$src/"
    aws s3 sync "s3://$BUCKET/$src/" "$DEST/vijil/$name/" "${EXCLUDES[@]}" --only-show-errors
}

# A model that lands in the wrong shape is worse than a missing one: the
# detectors fall back to the Hub and the tests skip green. Fail the step here.
verify_model() {
    local name="$1"
    local dir="$DEST/vijil/$name"
    if [ ! -f "$dir/config.json" ]; then
        echo "FAIL: $dir/config.json missing after sync" >&2
        return 1
    fi
    if ! compgen -G "$dir/*.safetensors" > /dev/null && ! compgen -G "$dir/*.bin" > /dev/null; then
        echo "FAIL: no model weights in $dir" >&2
        return 1
    fi
}

mkdir -p "$DEST/vijil"

# Each model is one large safetensors file, so the wall clock is dominated by
# a single multipart download per model — overlap them.
pids=()
for model in "${MODELS[@]}"; do
    sync_model "$model" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ] || { echo "one or more model syncs failed" >&2; exit 1; }

for model in "${MODELS[@]}"; do
    verify_model "$model"
done

echo "Models synced to $DEST/vijil:"
du -sh "$DEST"/vijil/*

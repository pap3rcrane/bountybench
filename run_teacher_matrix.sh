#!/usr/bin/env bash

set -uo pipefail

# Public flags:
#   --modes MODE[,MODE...]
#       Select one or more teacher modes. Valid modes are observe, steer, and
#       objective_rewrite. Modes run sequentially in the order supplied.
#       Default: observe,steer,objective_rewrite.
#   --jobs NUMBER
#       Run up to NUMBER repositories concurrently. Each job uses a dedicated
#       backend container, repository filesystem, and Docker-in-Docker volume.
#       Valid range: 1-5. Default: 5.
#   --prompt-placement user|system|both
#       Select where custom teacher prompts are sent. "user" uses the existing
#       prepend placement, "system" uses Gemini's API system instruction, and
#       "both" runs both variants separately. Default: both.
#   --skip-configurations-from STATUS_JSONL
#       May be repeated. Skip only finalized top-level configurations recorded
#       with status "success" in a previous run_status.jsonl. In-progress
#       student runs and failed configurations are never skipped.
#
# Resume source for the currently interrupted observe matrix. This intentionally
# selects its 143 finalized successful configuration records, not student runs
# that were still in progress when the launcher stopped:
#   ./run_teacher_matrix.sh --modes observe \
#     --skip-configurations-from runs/observe/2026-09-14_22-44-14-695542-0400/run_status.jsonl
#   --dry-run
#       Print and record every planned configuration without starting Docker
#       environments or calling either model.
#   --prune-dind-between-repositories
#       After each repository, remove unused containers, networks, volumes,
#       dangling images, and build cache from Docker-in-Docker. Tagged images
#       such as cybench/bountyagent:latest are preserved.
#   --verbose
#       Stream complete child workflow logs to the terminal in addition to
#       saving them under batch_logs/.
#   --no-progress
#       Disable tqdm progress bars and use plain terminal output instead.
#   -h, --help
#       Print the flag summary and exit.
#
# Optional environment variables (these are not command-line flags):
#   PYTHON_EXECUTABLE  Python interpreter used to launch the matrix runner.
#   RUNS_ROOT         Directory for compact run_status.jsonl files.
#   BATCH_LOG_ROOT    Directory for detailed per-configuration console logs.

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPOSITORY_ROOT"

usage() {
  printf '%s\n' \
    "Usage: ./run_teacher_matrix.sh [OPTIONS]" \
    "" \
    "Options:" \
    "  --modes MODE[,MODE...]              Modes to run; default is all three." \
    "  --jobs NUMBER                        Concurrent isolated jobs; default 5." \
    "  --prompt-placement PLACEMENT         user, system, or both; default both." \
    "  --skip-configurations-from FILE      Skip prior successful configurations." \
    "  --dry-run                           Plan and record without executing." \
    "  --prune-dind-between-repositories   Reclaim unused inner Docker storage." \
    "  --verbose                           Stream complete workflow output." \
    "  --no-progress                       Disable tqdm progress bars." \
    "  -h, --help                          Show this help." \
    "" \
    "Valid modes: observe, steer, objective_rewrite"
}

ORIGINAL_ARGUMENTS=("$@")
SELECTED_MODES_CSV="observe,steer,objective_rewrite"
SELECTED_JOBS=5
SELECTED_PROMPT_PLACEMENT="both"
SKIP_CONFIGURATION_SOURCES=()
DRY_RUN=false
FORWARD_ARGUMENTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modes)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --modes requires a comma-separated value.\n' >&2
        exit 2
      fi
      SELECTED_MODES_CSV="$2"
      shift 2
      ;;
    --modes=*)
      SELECTED_MODES_CSV="${1#*=}"
      if [[ -z "$SELECTED_MODES_CSV" ]]; then
        printf 'ERROR: --modes requires a comma-separated value.\n' >&2
        exit 2
      fi
      shift
      ;;
    --jobs)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --jobs requires a number from 1 through 5.\n' >&2
        exit 2
      fi
      SELECTED_JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      SELECTED_JOBS="${1#*=}"
      shift
      ;;
    --prompt-placement)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --prompt-placement requires user, system, or both.\n' >&2
        exit 2
      fi
      SELECTED_PROMPT_PLACEMENT="$2"
      shift 2
      ;;
    --prompt-placement=*)
      SELECTED_PROMPT_PLACEMENT="${1#*=}"
      shift
      ;;
    --skip-configurations-from)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --skip-configurations-from requires a run_status.jsonl file.\n' >&2
        exit 2
      fi
      SKIP_CONFIGURATION_SOURCES+=("$2")
      shift 2
      ;;
    --skip-configurations-from=*)
      SKIP_CONFIGURATION_SOURCES+=("${1#*=}")
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      FORWARD_ARGUMENTS+=("$1")
      shift
      ;;
    --prune-dind-between-repositories|--verbose|--no-progress)
      FORWARD_ARGUMENTS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$SELECTED_JOBS" =~ ^[1-5]$ ]]; then
  printf 'ERROR: --jobs must be a number from 1 through 5.\n' >&2
  exit 2
fi

case "$SELECTED_PROMPT_PLACEMENT" in
  user|system|both) ;;
  *)
    printf 'ERROR: --prompt-placement must be user, system, or both.\n' >&2
    exit 2
    ;;
esac

IFS=',' read -r -a SELECTED_MODES <<< "$SELECTED_MODES_CSV"
if [[ ${#SELECTED_MODES[@]} -eq 0 ]]; then
  printf 'ERROR: at least one mode is required.\n' >&2
  exit 2
fi

for index in "${!SELECTED_MODES[@]}"; do
  SELECTED_MODES[$index]="${SELECTED_MODES[$index]//[[:space:]]/}"
  case "${SELECTED_MODES[$index]}" in
    observe|steer|objective_rewrite) ;;
    *)
      printf 'ERROR: invalid teacher mode: %s\n' "${SELECTED_MODES[$index]}" >&2
      printf 'Valid modes: observe, steer, objective_rewrite\n' >&2
      exit 2
      ;;
  esac
done

ACTIVE_WORKER_SERVICES=()
ALL_WORKER_SERVICES=(
  backend-worker-1
  backend-worker-2
  backend-worker-3
  backend-worker-4
  backend-worker-5
)
WORKERS_STARTED=false
WORKER_LOCK_DIRECTORY="/tmp/bountybench-teacher-matrix-workers.lock"
WORKER_LOCK_ACQUIRED=false

stop_workers() {
  if [[ "$WORKERS_STARTED" == true ]]; then
    printf 'Stopping isolated matrix workers...\n'
    docker compose --profile matrix-workers stop "${ACTIVE_WORKER_SERVICES[@]}" >/dev/null
  fi
  if [[ "$WORKER_LOCK_ACQUIRED" == true ]]; then
    rmdir "$WORKER_LOCK_DIRECTORY" 2>/dev/null || true
  fi
}

start_workers() {
  if [[ "$SELECTED_JOBS" -eq 1 || "$DRY_RUN" == true ]]; then
    return
  fi

  if ! mkdir "$WORKER_LOCK_DIRECTORY" 2>/dev/null; then
    printf '%s\n' \
      "ERROR: another concurrent matrix launcher is already using the isolated workers." \
      "Wait for it to finish before starting another multi-job launcher." >&2
    exit 1
  fi
  WORKER_LOCK_ACQUIRED=true

  if ! docker image inspect bountybench-backend >/dev/null 2>&1; then
    printf '%s\n' \
      "ERROR: image 'bountybench-backend' is missing." \
      "Build it first with: docker compose up -d --build --force-recreate backend" >&2
    exit 1
  fi

  local job
  for ((job = 1; job <= SELECTED_JOBS; job++)); do
    ACTIVE_WORKER_SERVICES+=("backend-worker-$job")
  done

  printf 'Starting %s isolated matrix workers...\n' "$SELECTED_JOBS"
  docker compose --profile matrix-workers stop "${ALL_WORKER_SERVICES[@]}" \
    >/dev/null 2>&1 || true
  WORKERS_STARTED=true
  if ! docker compose --profile matrix-workers up -d --no-build --force-recreate \
    "${ACTIVE_WORKER_SERVICES[@]}"; then
    printf 'ERROR: failed to start the isolated matrix workers.\n' >&2
    exit 1
  fi
}

if [[ -n "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_COMMAND="$PYTHON_EXECUTABLE"
elif [[ -x "$REPOSITORY_ROOT/venv/bin/python" ]]; then
  PYTHON_COMMAND="$REPOSITORY_ROOT/venv/bin/python"
else
  PYTHON_COMMAND="python3"
fi

configure_mode() {
  local mode="$1"
  MATRIX_NAME="$mode"
  TEACHER_MODE="$mode"
  PROMPT_FILES=()
  ENVIRONMENTS=()

  case "$mode" in
    observe)
      PROMPT_FILES=(
        "prompts/system_prompts/single_error_correction_one_alternative.txt"
        "prompts/system_prompts/single_error_correction_multiple_alternatives.txt"
        "prompts/system_prompts/optimizer.txt"
        "prompts/system_prompts/principle_extraction.txt"
        "prompts/system_prompts/student_nudging.txt"
      )
      ENVIRONMENTS=(
        "LibreChat|0|exploit_workflow"
        "InvokeAI|0|exploit_workflow"
        "agentscope|0|exploit_workflow"
        "astropy|0|detect_workflow"
        "django|0|detect_workflow"
        "gpt_academic|0|detect_workflow"
        "kedro|0|patch_workflow"
        "scikit-learn|0|patch_workflow"
        "yaml|0|patch_workflow"
      )
      ;;
    steer)
      PROMPT_FILES=(
        "prompts/system_prompts/outcome_evaluation.txt"
        "prompts/system_prompts/teacher_answer_comparison.txt"
        "prompts/system_prompts/comparative_ranking.txt"
        "prompts/system_prompts/reasoning_quality_fidelity.txt"
        "prompts/system_prompts/rubric_based_evaluation.txt"
      )
      ENVIRONMENTS=(
        "mlflow|0|exploit_workflow"
        "lunary|0|exploit_workflow"
        "fastapi|0|exploit_workflow"
        "curl|0|detect_workflow"
        "gluon-cv|0|detect_workflow"
        "gunicorn|0|detect_workflow"
        "llama_index|0|patch_workflow"
        "setuptools|0|patch_workflow"
        "zipp|0|patch_workflow"
      )
      ;;
    objective_rewrite)
      PROMPT_FILES=(
        "prompts/system_prompts/increased_difficulty_task.txt"
        "prompts/system_prompts/weakness_targeting_task.txt"
        "prompts/system_prompts/strategy_limiter_task.txt"
        "prompts/system_prompts/justification_gap_task.txt"
        "prompts/system_prompts/oversight_task.txt"
      )
      ENVIRONMENTS=(
        "gradio|0|exploit_workflow"
        "composio|0|exploit_workflow"
        "node|0|exploit_workflow"
        "bentoml|0|detect_workflow"
        "langchain|0|detect_workflow"
        "pytorch-lightning|0|detect_workflow"
        "paddle|0|patch_workflow"
        "parse-url|0|patch_workflow"
        "undici|0|patch_workflow"
      )
      ;;
  esac
}

run_mode() {
  local mode="$1"
  configure_mode "$mode"

  local command=(
    "$PYTHON_COMMAND"
    "$REPOSITORY_ROOT/scripts/run_teacher_prompt_matrix.py"
    --matrix-name "$MATRIX_NAME"
    --teacher-mode "$TEACHER_MODE"
    --jobs "$SELECTED_JOBS"
    --prompt-placement "$SELECTED_PROMPT_PLACEMENT"
    --launcher "$0"
  )

  local prompt_file environment argument skip_source
  for prompt_file in "${PROMPT_FILES[@]}"; do
    command+=(--prompt-file "$prompt_file")
  done
  for environment in "${ENVIRONMENTS[@]}"; do
    command+=(--environment "$environment")
  done
  for skip_source in "${SKIP_CONFIGURATION_SOURCES[@]}"; do
    command+=(--skip-configurations-from "$skip_source")
  done
  for argument in "${ORIGINAL_ARGUMENTS[@]}"; do
    command+=("--launcher-argument=$argument")
  done

  printf 'Running teacher matrix mode: %s\n' "$mode"
  "${command[@]}" "${FORWARD_ARGUMENTS[@]}"
}

trap stop_workers EXIT
start_workers

OVERALL_EXIT_CODE=0
for mode in "${SELECTED_MODES[@]}"; do
  run_mode "$mode"
  MODE_EXIT_CODE=$?
  if [[ "$MODE_EXIT_CODE" -eq 130 ]]; then
    exit 130
  fi
  if [[ "$MODE_EXIT_CODE" -ne 0 ]]; then
    OVERALL_EXIT_CODE=1
  fi
done

exit "$OVERALL_EXIT_CODE"

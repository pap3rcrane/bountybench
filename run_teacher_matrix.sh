#!/usr/bin/env bash

set -uo pipefail

# Public flags:
#   --modes MODE[,MODE...]
#       Select one or more teacher modes. Valid modes are observe, steer, and
#       objective_rewrite. Selected modes run in order and reuse the same global
#       worker pool. Default: observe,steer,objective_rewrite.
#   --jobs NUMBER|all
#       Run up to NUMBER configurations concurrently. Configurations remain ordered
#       by environment, so all work for one environment enters the worker pool before
#       work for the next environment. Each job uses a dedicated container, network,
#       artifact directory, and Docker-in-Docker volume. Valid range: 1-30. "all"
#       means 30. Default: all.
#   --setup-jobs NUMBER
#       Limit concurrent repository setup/build scripts while retaining all selected
#       configuration workers. Valid range: 1-30. Default: 5.
#   --prompt-placement user|system|both
#       Select where custom teacher prompts are sent. "user" uses the existing
#       prepend placement, "system" uses Gemini's API system instruction, and
#       "both" runs both variants separately. Default: both.
#   --only-prompt NAME
#       Run only the named custom teacher prompt and omit the no-custom-prompt
#       baseline. Example: --modes steer --only-prompt comparative_ranking.
#   --teacher-max-input-tokens NUMBER
#       Maximum Gemini teacher input context. Default: 1048576, the model maximum.
#   --teacher-max-output-tokens NUMBER
#       Maximum Gemini teacher response length. Default: 65536, the model maximum.
#   --phase-iterations NUMBER
#       Maximum student iterations in each workflow phase. In observe and steer,
#       this also bounds teacher evaluations because the teacher runs after each
#       completed student turn. Default: 300.
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
#       When a worker moves to another environment, remove its unused containers,
#       networks, volumes, dangling images, and build cache from Docker-in-Docker.
#       Tagged images such as cybench/bountyagent:latest are preserved.
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
#   MATRIX_WORKER_CPU_LIMIT
#                       Optional CPU quota per worker, for example 2. Omitted by default.
#   MATRIX_WORKER_LOCK_DIRECTORY
#                       Override the single-launcher lock path (primarily for tests).
#   MATRIX_REGISTRY_CACHE_ENABLED
#                       Share Docker Hub downloads through one pull-through cache while
#                       retaining a separate Docker daemon per worker. Default: true.
#   MATRIX_REGISTRY_CACHE_IMAGE
#                       Registry image used for the cache. Default:
#                       mirror.gcr.io/library/registry:2.
#   DOCKERHUB_USERNAME Docker Hub username for authenticated upstream pulls (optional).
#   DOCKERHUB_TOKEN    Docker Hub access token; required with DOCKERHUB_USERNAME.
#   BOUNTY_AGENT_IMAGE Kali agent image used inside each worker. Default:
#                       cybench/bountyagent:latest.

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPOSITORY_ROOT"

usage() {
  printf '%s\n' \
    "Usage: ./run_teacher_matrix.sh [OPTIONS]" \
    "" \
    "Options:" \
    "  --modes MODE[,MODE...]              Modes to run; default is all three." \
    "  --jobs NUMBER|all                    Global configuration workers; all means 30." \
    "  --setup-jobs NUMBER                  Concurrent repository setups; default 5." \
    "  --prompt-placement PLACEMENT         user, system, or both; default both." \
    "  --only-prompt NAME                   Run only one named prompt; omit baseline." \
    "  --teacher-max-input-tokens NUMBER    Gemini teacher input limit; default 1048576." \
    "  --teacher-max-output-tokens NUMBER   Gemini teacher output limit; default 65536." \
    "  --phase-iterations NUMBER            Student phase limit; default 300." \
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
SELECTED_JOBS="all"
SELECTED_SETUP_JOBS=5
SELECTED_PROMPT_PLACEMENT="both"
SELECTED_ONLY_PROMPT=""
SELECTED_TEACHER_MAX_INPUT_TOKENS=1048576
SELECTED_TEACHER_MAX_OUTPUT_TOKENS=65536
SELECTED_PHASE_ITERATIONS=300
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
        printf 'ERROR: --jobs requires all or a number from 1 through 30.\n' >&2
        exit 2
      fi
      SELECTED_JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      SELECTED_JOBS="${1#*=}"
      shift
      ;;
    --setup-jobs)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --setup-jobs requires a number from 1 through 30.\n' >&2
        exit 2
      fi
      SELECTED_SETUP_JOBS="$2"
      shift 2
      ;;
    --setup-jobs=*)
      SELECTED_SETUP_JOBS="${1#*=}"
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
    --only-prompt)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --only-prompt requires a prompt filename stem.\n' >&2
        exit 2
      fi
      SELECTED_ONLY_PROMPT="$2"
      shift 2
      ;;
    --only-prompt=*)
      SELECTED_ONLY_PROMPT="${1#*=}"
      if [[ -z "$SELECTED_ONLY_PROMPT" ]]; then
        printf 'ERROR: --only-prompt requires a prompt filename stem.\n' >&2
        exit 2
      fi
      shift
      ;;
    --teacher-max-input-tokens)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --teacher-max-input-tokens requires a positive integer.\n' >&2
        exit 2
      fi
      SELECTED_TEACHER_MAX_INPUT_TOKENS="$2"
      shift 2
      ;;
    --teacher-max-input-tokens=*)
      SELECTED_TEACHER_MAX_INPUT_TOKENS="${1#*=}"
      shift
      ;;
    --teacher-max-output-tokens)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --teacher-max-output-tokens requires a positive integer.\n' >&2
        exit 2
      fi
      SELECTED_TEACHER_MAX_OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --teacher-max-output-tokens=*)
      SELECTED_TEACHER_MAX_OUTPUT_TOKENS="${1#*=}"
      shift
      ;;
    --phase-iterations)
      if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'ERROR: --phase-iterations requires a positive integer.\n' >&2
        exit 2
      fi
      SELECTED_PHASE_ITERATIONS="$2"
      shift 2
      ;;
    --phase-iterations=*)
      SELECTED_PHASE_ITERATIONS="${1#*=}"
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

if [[ "$SELECTED_JOBS" != "all" && ! "$SELECTED_JOBS" =~ ^([1-9]|[12][0-9]|30)$ ]]; then
  printf 'ERROR: --jobs must be all or a number from 1 through 30.\n' >&2
  exit 2
fi

if [[ ! "$SELECTED_SETUP_JOBS" =~ ^([1-9]|[12][0-9]|30)$ ]]; then
  printf 'ERROR: --setup-jobs must be a number from 1 through 30.\n' >&2
  exit 2
fi

if [[ ! "$SELECTED_TEACHER_MAX_INPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: --teacher-max-input-tokens must be a positive integer.\n' >&2
  exit 2
fi

if [[ ! "$SELECTED_TEACHER_MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: --teacher-max-output-tokens must be a positive integer.\n' >&2
  exit 2
fi

if [[ ! "$SELECTED_PHASE_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: --phase-iterations must be a positive integer.\n' >&2
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

MAX_MATRIX_WORKERS=30
MATRIX_WORKER_CPU_LIMIT="${MATRIX_WORKER_CPU_LIMIT:-}"
MATRIX_REGISTRY_CACHE_ENABLED="${MATRIX_REGISTRY_CACHE_ENABLED:-true}"
MATRIX_REGISTRY_CACHE_IMAGE="${MATRIX_REGISTRY_CACHE_IMAGE:-mirror.gcr.io/library/registry:2}"
MATRIX_REGISTRY_CACHE_CONTAINER="bountybench-matrix-registry-cache"
MATRIX_REGISTRY_CACHE_VOLUME="bountybench-matrix-registry-cache"
MATRIX_REGISTRY_CACHE_ALIAS="matrix-registry-cache"
MATRIX_REGISTRY_CACHE_URL="http://$MATRIX_REGISTRY_CACHE_ALIAS:5000"
MATRIX_LAUNCH_ID="$(date +%Y%m%d-%H%M%S)-$$"
MATRIX_SETUP_GATE_VOLUME="bb-matrix-$MATRIX_LAUNCH_ID-setup-gate"
MATRIX_SETUP_GATE_PATH="/matrix-setup-gate"
MATRIX_WORKER_COUNT=0
MATRIX_SETUP_JOB_COUNT=0
ACTIVE_WORKER_SERVICES=()
ACTIVE_WORKER_NETWORKS=()
ACTIVE_WORKER_VOLUMES=()
WORKERS_STARTED=false
WORKER_LOCK_DIRECTORY="${MATRIX_WORKER_LOCK_DIRECTORY:-/tmp/bountybench-teacher-matrix-workers.lock}"
WORKER_LOCK_ACQUIRED=false

case "$MATRIX_REGISTRY_CACHE_ENABLED" in
  true|false) ;;
  *)
    printf 'ERROR: MATRIX_REGISTRY_CACHE_ENABLED must be true or false.\n' >&2
    exit 2
    ;;
esac

if [[ -n "${DOCKERHUB_USERNAME:-}" || -n "${DOCKERHUB_TOKEN:-}" ]]; then
  if [[ -z "${DOCKERHUB_USERNAME:-}" || -z "${DOCKERHUB_TOKEN:-}" ]]; then
    printf 'ERROR: DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be set together.\n' >&2
    exit 2
  fi
fi

if [[ "${BATCH_LOG_ROOT:-$REPOSITORY_ROOT/batch_logs}" = /* ]]; then
  MATRIX_BATCH_LOG_ROOT="${BATCH_LOG_ROOT:-$REPOSITORY_ROOT/batch_logs}"
else
  MATRIX_BATCH_LOG_ROOT="$REPOSITORY_ROOT/${BATCH_LOG_ROOT}"
fi
WORKER_ARTIFACT_ROOT="$MATRIX_BATCH_LOG_ROOT/matrix_workers/$MATRIX_LAUNCH_ID"

stop_workers() {
  if [[ "$WORKERS_STARTED" == true ]]; then
    printf 'Removing isolated matrix workers, registry cache, and DinD volumes...\n'
    docker rm -f "${ACTIVE_WORKER_SERVICES[@]}" >/dev/null 2>&1 || true
    if [[ "$MATRIX_REGISTRY_CACHE_ENABLED" == true ]]; then
      docker rm -f "$MATRIX_REGISTRY_CACHE_CONTAINER" >/dev/null 2>&1 || true
    fi
    docker network rm "${ACTIVE_WORKER_NETWORKS[@]}" >/dev/null 2>&1 || true
    docker volume rm "${ACTIVE_WORKER_VOLUMES[@]}" >/dev/null 2>&1 || true
    docker volume rm "$MATRIX_SETUP_GATE_VOLUME" >/dev/null 2>&1 || true
    if [[ "$MATRIX_REGISTRY_CACHE_ENABLED" == true ]]; then
      docker volume rm "$MATRIX_REGISTRY_CACHE_VOLUME" >/dev/null 2>&1 || true
    fi
  fi
  if [[ "$WORKER_LOCK_ACQUIRED" == true ]]; then
    rmdir "$WORKER_LOCK_DIRECTORY" 2>/dev/null || true
  fi
}

start_workers() {
  local job worker_name network_name volume_name worker_root host_cpu_count
  local network_index stale_resource

  mkdir -p "$WORKER_ARTIFACT_ROOT"
  for ((job = 1; job <= MATRIX_WORKER_COUNT; job++)); do
    worker_name="backend-worker-$job"
    ACTIVE_WORKER_SERVICES+=("$worker_name")
    worker_root="$WORKER_ARTIFACT_ROOT/$worker_name"
    mkdir -p "$worker_root/logs" "$worker_root/full_logs" \
      "$worker_root/generated_objectives"
  done

  if [[ "$DRY_RUN" == true ]]; then
    return
  fi

  host_cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if [[ "$host_cpu_count" =~ ^[0-9]+$ && "$MATRIX_WORKER_COUNT" -gt "$host_cpu_count" ]]; then
    printf 'WARNING: %s workers exceed the %s detected CPU cores; setup phases may contend.\n' \
      "$MATRIX_WORKER_COUNT" "$host_cpu_count" >&2
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

  # The global launcher lock guarantees these can only be stale workers from an
  # interrupted prior launch, never containers owned by another active matrix.
  docker rm -f "${ACTIVE_WORKER_SERVICES[@]}" >/dev/null 2>&1 || true
  docker rm -f "$MATRIX_REGISTRY_CACHE_CONTAINER" >/dev/null 2>&1 || true
  while IFS= read -r stale_resource; do
    case "$stale_resource" in
      bb-matrix-*) docker network rm "$stale_resource" >/dev/null 2>&1 || true ;;
    esac
  done < <(docker network ls --format '{{.Name}}')
  while IFS= read -r stale_resource; do
    case "$stale_resource" in
      bb-matrix-*-dind|bb-matrix-*-setup-gate)
        docker volume rm "$stale_resource" >/dev/null 2>&1 || true
        ;;
    esac
  done < <(docker volume ls --format '{{.Name}}')
  docker volume rm "$MATRIX_REGISTRY_CACHE_VOLUME" >/dev/null 2>&1 || true

  printf 'Starting %s fully isolated matrix workers...\n' "$MATRIX_WORKER_COUNT"
  WORKERS_STARTED=true
  for ((job = 1; job <= MATRIX_WORKER_COUNT; job++)); do
    network_name="bb-matrix-$MATRIX_LAUNCH_ID-worker-$job"
    volume_name="bb-matrix-$MATRIX_LAUNCH_ID-worker-$job-dind"
    ACTIVE_WORKER_NETWORKS+=("$network_name")
    ACTIVE_WORKER_VOLUMES+=("$volume_name")

    if ! docker network create \
      --subnet 0.0.0.0/24 \
      "$network_name" >/dev/null; then
      printf 'ERROR: failed to create isolated network %s.\n' \
        "$network_name" >&2
      exit 1
    fi
    if ! docker volume create "$volume_name" >/dev/null; then
      printf 'ERROR: failed to create DinD volume %s.\n' "$volume_name" >&2
      exit 1
    fi
  done

  if ! docker volume create "$MATRIX_SETUP_GATE_VOLUME" >/dev/null; then
    printf 'ERROR: failed to create repository setup gate volume %s.\n' \
      "$MATRIX_SETUP_GATE_VOLUME" >&2
    exit 1
  fi

  if [[ "$MATRIX_REGISTRY_CACHE_ENABLED" == true ]]; then
    if ! docker image inspect "$MATRIX_REGISTRY_CACHE_IMAGE" >/dev/null 2>&1; then
      printf 'Pulling shared matrix registry cache image: %s\n' \
        "$MATRIX_REGISTRY_CACHE_IMAGE"
      if ! docker pull "$MATRIX_REGISTRY_CACHE_IMAGE"; then
        printf 'ERROR: unable to pull registry cache image %s.\n' \
          "$MATRIX_REGISTRY_CACHE_IMAGE" >&2
        exit 1
      fi
    fi

    if ! docker volume create "$MATRIX_REGISTRY_CACHE_VOLUME" >/dev/null; then
      printf 'ERROR: failed to create the shared registry cache volume.\n' >&2
      exit 1
    fi
    local cache_command=(
      docker run -d
      --name "$MATRIX_REGISTRY_CACHE_CONTAINER"
      --network "${ACTIVE_WORKER_NETWORKS[0]}"
      --network-alias "$MATRIX_REGISTRY_CACHE_ALIAS"
      --volume "$MATRIX_REGISTRY_CACHE_VOLUME:/var/lib/registry"
      --env REGISTRY_PROXY_REMOTEURL=https://registry-1.docker.io
    )
    if [[ -n "${DOCKERHUB_USERNAME:-}" ]]; then
      cache_command+=(
        --env "REGISTRY_PROXY_USERNAME=$DOCKERHUB_USERNAME"
        --env "REGISTRY_PROXY_PASSWORD=$DOCKERHUB_TOKEN"
      )
    fi
    cache_command+=("$MATRIX_REGISTRY_CACHE_IMAGE")
    if ! "${cache_command[@]}" >/dev/null; then
      printf 'ERROR: failed to start the shared Docker registry cache.\n' >&2
      exit 1
    fi

    for ((network_index = 1; network_index < ${#ACTIVE_WORKER_NETWORKS[@]}; network_index++)); do
      network_name="${ACTIVE_WORKER_NETWORKS[$network_index]}"
      if ! docker network connect \
        --alias "$MATRIX_REGISTRY_CACHE_ALIAS" \
        "$network_name" \
        "$MATRIX_REGISTRY_CACHE_CONTAINER"; then
        printf 'ERROR: failed to connect the registry cache to %s.\n' \
          "$network_name" >&2
        exit 1
      fi
    done
  fi

  for ((job = 1; job <= MATRIX_WORKER_COUNT; job++)); do
    worker_name="backend-worker-$job"
    network_name="${ACTIVE_WORKER_NETWORKS[$((job - 1))]}"
    volume_name="${ACTIVE_WORKER_VOLUMES[$((job - 1))]}"
    worker_root="$WORKER_ARTIFACT_ROOT/$worker_name"

    local docker_command=(
      docker run -d
      --name "$worker_name"
      --hostname "$worker_name"
      --privileged
      --network "$network_name"
      --volume "$volume_name:/var/lib/docker"
      --volume "$worker_root/logs:/app/logs"
      --volume "$worker_root/full_logs:/app/full_logs"
      --volume "$worker_root/generated_objectives:/app/generated_objectives"
      --volume "$MATRIX_SETUP_GATE_VOLUME:$MATRIX_SETUP_GATE_PATH"
      --env "BOUNTYBENCH_REPO_SETUP_GATE_DIR=$MATRIX_SETUP_GATE_PATH"
      --env "BOUNTYBENCH_REPO_SETUP_CONCURRENCY=$MATRIX_SETUP_JOB_COUNT"
      --volume "$REPOSITORY_ROOT/prompts/teacher_agent_system_prompt.txt:/app/prompts/teacher_agent_system_prompt.txt:ro"
      --volume "$REPOSITORY_ROOT/prompts/student_prompts:/app/prompts/student_prompts:ro"
      --volume "$REPOSITORY_ROOT/prompts/system_prompts:/app/prompts/system_prompts:ro"
    )
    if [[ "$MATRIX_REGISTRY_CACHE_ENABLED" == true ]]; then
      docker_command+=(--env "DOCKER_REGISTRY_MIRROR=$MATRIX_REGISTRY_CACHE_URL")
    fi
    if [[ -f "$REPOSITORY_ROOT/.env" ]]; then
      docker_command+=(--env-file "$REPOSITORY_ROOT/.env")
    fi
    docker_command+=(
      --env "BOUNTYBENCH_DISABLE_KALI_DOCKER=${BOUNTYBENCH_DISABLE_KALI_DOCKER:-1}"
    )
    local environment_variable
    for environment_variable in \
      HELM_API_KEY \
      OPENAI_API_KEY \
      OPENROUTER_API_KEY \
      AZURE_OPENAI_API_KEY \
      AZURE_OPENAI_ENDPOINT \
      ANTHROPIC_API_KEY \
      GOOGLE_API_KEY \
      TOGETHER_API_KEY \
      BOUNTY_AGENT_IMAGE \
      DOCKERHUB_USERNAME \
      DOCKERHUB_TOKEN; do
      if [[ -n "${!environment_variable:-}" ]]; then
        docker_command+=(--env "$environment_variable")
      fi
    done
    if [[ -n "$MATRIX_WORKER_CPU_LIMIT" ]]; then
      docker_command+=(--cpus "$MATRIX_WORKER_CPU_LIMIT")
    fi
    docker_command+=(bountybench-backend tail -f /dev/null)
    if ! "${docker_command[@]}" >/dev/null; then
      printf 'ERROR: failed to start isolated worker %s.\n' "$worker_name" >&2
      exit 1
    fi
  done
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

  if [[ -n "$SELECTED_ONLY_PROMPT" ]]; then
    local matching_prompts=()
    local candidate_prompt
    for candidate_prompt in "${PROMPT_FILES[@]}"; do
      if [[ "$(basename "$candidate_prompt" .txt)" == "$SELECTED_ONLY_PROMPT" ]]; then
        matching_prompts+=("$candidate_prompt")
      fi
    done
    if [[ ${#matching_prompts[@]} -eq 0 ]]; then
      printf 'ERROR: prompt %s is not configured for mode %s.\n' \
        "$SELECTED_ONLY_PROMPT" "$mode" >&2
      exit 2
    fi
    PROMPT_FILES=("${matching_prompts[@]}")
  fi
}

resolve_worker_allocation() {
  if [[ "$SELECTED_JOBS" == "all" ]]; then
    MATRIX_WORKER_COUNT=$MAX_MATRIX_WORKERS
  else
    MATRIX_WORKER_COUNT=$SELECTED_JOBS
  fi
  if [[ "$SELECTED_SETUP_JOBS" -lt "$MATRIX_WORKER_COUNT" ]]; then
    MATRIX_SETUP_JOB_COUNT=$SELECTED_SETUP_JOBS
  else
    MATRIX_SETUP_JOB_COUNT=$MATRIX_WORKER_COUNT
  fi
}

run_mode() {
  local mode="$1"
  configure_mode "$mode"

  local command=(
    "$PYTHON_COMMAND"
    "$REPOSITORY_ROOT/scripts/run_teacher_prompt_matrix.py"
    --matrix-name "$MATRIX_NAME"
    --teacher-mode "$TEACHER_MODE"
    --jobs "$MATRIX_WORKER_COUNT"
    --setup-jobs "$MATRIX_SETUP_JOB_COUNT"
    --prompt-placement "$SELECTED_PROMPT_PLACEMENT"
    --teacher-max-input-tokens "$SELECTED_TEACHER_MAX_INPUT_TOKENS"
    --teacher-max-output-tokens "$SELECTED_TEACHER_MAX_OUTPUT_TOKENS"
    --phase-iterations "$SELECTED_PHASE_ITERATIONS"
    --launcher "$0"
  )

  if [[ -n "$SELECTED_ONLY_PROMPT" ]]; then
    command+=(--exclude-baseline)
  fi

  local prompt_file environment argument skip_source worker_number worker_name
  for ((worker_number = 1; worker_number <= MATRIX_WORKER_COUNT; worker_number++)); do
    worker_name="backend-worker-$worker_number"
    command+=(--backend-container "$worker_name")
    command+=(
      --backend-log-root
      "$worker_name=$WORKER_ARTIFACT_ROOT/$worker_name"
    )
  done
  for prompt_file in "${PROMPT_FILES[@]}"; do
    command+=(--prompt-file "$prompt_file")
  done
  for environment in "${ENVIRONMENTS[@]}"; do
    command+=(--environment "$environment")
  done
  for skip_source in "${SKIP_CONFIGURATION_SOURCES[@]+"${SKIP_CONFIGURATION_SOURCES[@]}"}"; do
    command+=(--skip-configurations-from "$skip_source")
  done
  for argument in "${ORIGINAL_ARGUMENTS[@]+"${ORIGINAL_ARGUMENTS[@]}"}"; do
    command+=("--launcher-argument=$argument")
  done
  printf 'Running teacher matrix mode: %s with %s configuration workers and %s setup slots\n' \
    "$mode" "$MATRIX_WORKER_COUNT" "$MATRIX_SETUP_JOB_COUNT"
  "${command[@]}" "${FORWARD_ARGUMENTS[@]+"${FORWARD_ARGUMENTS[@]}"}"
}

cleanup() {
  stop_workers
}

trap cleanup EXIT
trap 'exit 130' INT TERM
resolve_worker_allocation
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

docker compose build backend
docker compose up -d --force-recreate backend


run_and_export() {
  task="$1"
  bounty="$2"

  echo "Running $task bounty $bounty"

  docker exec -i backend-service \
    python -m workflows.runner \
    --workflow-type detect_workflow \
    --task_dir "bountytasks/$task" \
    --bounty_number "$bounty" \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --phase_iterations 100 \
    --max_input_tokens 8192 \
    --max_output_tokens 8192 \
    --logging_level DEBUG

  workflow_status=$?

  python3 environment_timelines/export_latest_agent_timeline.py
  timeline_status=$?

  echo "$task bounty $bounty: workflow=$workflow_status timeline=$timeline_status"
}

run_and_export mlflow 2
run_and_export node 0

unset -f run_and_export
# BountyBench

## Table of Contents

- [bountybench](#bountybench)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Environment Setup](#environment-setup)
    - [1. Ensure Python 3.11 is Installed](#1-ensure-python-311-is-installed)
    - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
    - [3. Activate and Set Up the Environment](#3-activate-and-set-up-the-environment)
    - [4. Configure the .env File](#4-configure-the-env-file)
    - [5. Setup Docker Desktop App](#5-setup-docker-desktop-app)
  - [Usage](#usage)
    - [Running Workflows](#running-workflows)
    - [Running the Workflows through Web Interface](#running-the-workflows-through-web-interface)
    - [Dockerize run](#dockerize-run)
    - [Sample Run](#sample-run)
    - [Troubleshooting](#troubleshooting)


## Installation

### Environment Setup

You can quickly set up the dev environment by running the following command:

```bash
./setup.sh
source venv/bin/activate
```

To initialize all submodules, run:

```bash
./setup.sh --all
source venv/bin/activate
```

### Alternative Manual Setup

If you prefer to set up the environment manually, follow these steps:

#### 1. Ensure Python 3.11 is Installed

Verify that Python 3.11 is available on your system:

```bash
python3.11 --version
```

#### 2. Create a Virtual Environment

Set up a virtual environment to isolate dependencies:

```bash
python3.11 -m venv venv
```

#### 3. Activate and Set Up the Environment

Activate the virtual environment, install required dependencies (may take several minutes to tens of minutes to complete, please leave time for this installation):

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Initialize submodules (may take a few minutes to complete):

```bash
git submodule update --init
git submodule update --remote
cd bountytasks
git submodule update --init
```

Additionally, please install `tree`:

macOS (using Homebrew):

```bash
brew install tree
```

or Debian/Ubuntu (using APT):

```bash
sudo apt-get install tree
```

#### 4. Configure the .env File

Create and populate an .env file in `bountybench/` with the following keys:

```bash
ANTHROPIC_API_KEY={ANTHROPIC_API_KEY}
AZURE_OPENAI_API_KEY={AZURE_OPENAI_API_KEY}
AZURE_OPENAI_ENDPOINT={AZURE_OPENAI_ENDPOINT}
GOOGLE_API_KEY={GOOGLE_API_KEY}
HELM_API_KEY={HELM_API_KEY}
OPENAI_API_KEY={OPENAI_API_KEY}
OPENROUTER_API_KEY={OPENROUTER_API_KEY}
TOGETHER_API_KEY={TOGETHER_API_KEY}
XAI_API_KEY={XAI_API_KEY}
```

Replace {KEY_NAME} with your actual API key values (make sure you don't include {} when adding the key, e.g. KEY=sk-proj...). You only need to fill in whichever keys you will use.

#### 5. Setup Docker Desktop App

Make sure that you have started up your Docker Desktop App before proceeding with running a workflow.

##### Docker Setup

To get started with Docker, follow these installation instructions based on your operating system:

- **[Docker Desktop Installation for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)**
- **[Docker Desktop Installation for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)**

 **Verify Installation**  

- Open a terminal or command prompt and run the following command:  

     ```bash
     docker --version
     ```  

- Ensure Docker is installed and the version is displayed.

###### Ensure your Docker Desktop has proper sharing permissions

You want to ensure that Docker Desktop has mounting permissions for your current working directory. Run:
`docker run --rm -v "$(pwd)":/test alpine ls /test`
It should list the contents of your current working directory. If you encounter a mounting issue, please follow [Docker Mount Issue](#docker-mount-issue) for next steps.

## Usage

### Running Workflows

Make sure your Docker Desktop app is running.

Running workflows from CLI should use `runner.py` module. Each runnable workflow defines required and optional arguments. Important parameter interactions:

- `--model` and `--use_mock_model` are mutually exclusive. You cannot specify both simultaneously.
- If `--use_mock_model` is True, then `--use_helm` parameter is ignored
- The `--use_helm` parameter determines whether to use Helm as the model provider

Teacher support is optional. A teacher-enabled command must provide `--teacher_mode` and
`--teacher_system_prompt_placement`. The teacher never receives an environment resource
and therefore cannot run commands or submit work.

The system prompt placement must be selected explicitly:

- `none`: omits the custom teacher prompt. The original benchmark task remains ahead of
  the trace in the user message. For `objective_rewrite`, the API system instruction
  contains only the required JSON response format.
- `prepend`: prepends the TXT file contents and original benchmark task to the teacher's
  user message, followed by the trace.
- `system`: passes the TXT file contents and original benchmark task to Gemini together
  as its API-level `system_instruction`; only the trace remains in the user message.
  This placement requires a direct `google/...` Gemini teacher model.

`--teacher_system_prompt_file` is required for `prepend` and `system`, and must be
omitted for `none`.

For true Gemini system-prompt transport, use, for example,
`--teacher_model google/gemini-2.5-flash` together with
`--teacher_system_prompt_placement system`. Gemini models routed through OpenRouter do
not use this placement.

The available modes are:

- `observe`: runs after the evaluator on every student turn. It sees the original
  benchmark task followed by only student model output, commands, and environment
  responses. Its response is logged but hidden from the student, and prior teacher
  observations are not included in later teacher inputs.
- `steer`: receives the same original-task and student-only trace as `observe`, then
  places its response in the student's context for the next turn.
- `objective_rewrite`: consumes exactly three task-and-student-trace blocks from three
  teacher-free source runs, writes a strict
  `{"objective": "..."}` file, and immediately launches one teacher-free student run in
  which that objective replaces the applicable `DETECT_DESCRIPTION`,
  `EXPLOIT_DESCRIPTION`, or `PATCH_DESCRIPTION`. Combined workflows use one objective
  for both phases. Generated source runs use the same workflow arguments and clean
  environment setup; no seed is passed. Invalid or extra teacher JSON fails immediately.
  Its five `*_task.txt` teacher prompts each contain the required JSON response format;
  the no-custom-prompt baseline uses only that format as its API system instruction.

Example of per-turn steering:

```bash
python -m workflows.runner --workflow-type detect_workflow \
    --task_dir bountytasks/lunary \
    --bounty_number 0 \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --teacher_model openrouter/deepseek/deepseek-v4-pro \
    --teacher_system_prompt_file prompts/teacher_agent_system_prompt.txt \
    --teacher_system_prompt_placement prepend \
    --teacher_mode steer \
    --phase_iterations 30
```

Generate three source runs, rewrite the objective, and launch the next student run:

```bash
python -m workflows.runner --workflow-type detect_workflow \
    --task_dir bountytasks/lunary \
    --bounty_number 0 \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --teacher_model openrouter/deepseek/deepseek-v4-pro \
    --teacher_system_prompt_file prompts/teacher_agent_system_prompt.txt \
    --teacher_system_prompt_placement prepend \
    --teacher_mode objective_rewrite \
    --generate_source_runs \
    --phase_iterations 30
```

To use existing runs instead, replace `--generate_source_runs` with
`--source_logs LOG_1 LOG_2 LOG_3`. Objective files are written to
`generated_objectives/` by default; change this with `--objective_output_dir`.
Teacher responses are recorded as `teacher_agent` actions in normal workflow JSON logs.
The automatic objective-rewrite run records its teacher model, system prompt file,
placement, three source logs, and rewritten objective in the final workflow log metadata.
Commands that omit all teacher flags run without a teacher.

The `run_teacher_matrix.sh` batch launcher exercises the uploaded prompt groups over
three disjoint environment sets, one for each teacher mode.

Each launcher contains nine `(repository, bounty, workflow)` environments split evenly
across detect, exploit, and patch. Each environment runs its five mode-specific prompts
with both `prepend` and `system` placement, five times per configuration, followed by five
teacher-active `none` baseline runs. That is 495 top-level runs per launcher. Every
`objective_rewrite` run also generates three fresh teacher-free source runs before its
rewritten-objective student run.

Select only the user-message (`prepend`) variant, only Gemini's API system-instruction
variant, or both variants with `--prompt-placement user`, `system`, or `both`. The
default is `both`, and the five teacher-active `none` baseline runs remain included for
all three selections. A single-placement matrix contains 270 top-level runs.

Rebuild the backend once so it contains the runner changes and mounts the prompt folder:

```bash
docker compose up -d --build --force-recreate backend
```

Run a matrix with a progress-focused terminal:

```bash
./run_teacher_matrix.sh
```

The launcher runs five repositories concurrently by default. Each job receives its own
`backend-worker-N` container, repository filesystem, and Docker-in-Docker volume, while
all configurations for the same repository remain sequential. This prevents concurrent
runs from resetting the same Git checkout or Docker environment. Use `--jobs 1` through
`--jobs 5` to change the limit:

```bash
./run_teacher_matrix.sh --jobs 3
```

The isolated workers are stopped when the launcher exits. Their named Docker volumes are
retained so later runs can reuse downloaded images; use
`--prune-dind-between-repositories` when storage is more important than cache reuse.

Run only selected modes with a comma-separated list:

```bash
./run_teacher_matrix.sh --modes observe,steer
```

Run only the API-level system-prompt configurations (plus the baseline):

```bash
./run_teacher_matrix.sh --prompt-placement system
```

Resume from one or more earlier status indexes without rerunning configurations that
already completed successfully:

```bash
./run_teacher_matrix.sh \
    --modes observe \
    --skip-configurations-from runs/observe/TIMESTAMP/run_status.jsonl
```

The skip loader deliberately reads only top-level `configuration` records whose status
is `success`. It does not skip failures or a configuration whose student run completed
but whose top-level configuration record had not yet finalized. Skipped configurations
are written into the new status index with status `skipped` for complete accounting.

With no `--modes` flag, the launcher runs `observe`, `steer`, and
`objective_rewrite` sequentially. Run `./run_teacher_matrix.sh --help` for all flags.

The outer `tqdm` bar tracks completed configurations and estimates the remaining time
from their complete elapsed durations, including environment setup, model calls,
evaluation, and cleanup. Up to five job bars track each active workflow's completed
iterations out of 300. Objective rewriting labels `source_1`, `source_2`, `source_3`,
the teacher rewrite, and the final `rewritten_objective` student separately.

To reclaim Docker-in-Docker storage after each repository finishes, add:

```bash
./run_teacher_matrix.sh --prune-dind-between-repositories
```

This removes unused inner containers, networks, volumes, dangling images, and build
cache. It deliberately preserves tagged images such as `cybench/bountyagent:latest`.
Cleanup failures are reported as warnings and do not stop later repository runs.

Detailed child output is saved under `batch_logs/<teacher-mode>/<timestamp>/` instead of
flooding the terminal. Add `--verbose` to stream those lines too, or `--no-progress` for
plain CI/redirected output.

Every invocation also writes a compact status index to
`runs/<teacher-mode>/<timestamp>/run_status.jsonl`. It contains batch details, one status
record per top-level configuration, and one status record for every student workflow.
The student roles are `normal`, or `source_1` through `source_3` plus
`rewritten_objective`. Records contain identifiers, repository, bounty, workflow, system
prompt name (or `none`), repetition 1–5, teacher type/model, timestamps, duration, exit
code, status, assigned backend worker, concise error, and pointers to the detailed logs.
They do not duplicate prompts, model responses, commands, or environment output. A
student that never starts because an earlier stage failed is recorded as `skipped`.

Set `RUNS_ROOT` or `BATCH_LOG_ROOT` to relocate either output tree.

Preview a matrix without calling either model:

```bash
BATCH_LOG_ROOT=/tmp/bountybench-matrix ./run_teacher_matrix.sh \
    --modes observe --dry-run
```

Print the complete Gemini prompt payload for every detect/exploit/patch workflow,
uploaded system prompt, and `prepend`/`system` placement pair:

```bash
python scripts/print_teacher_prompts.py \
    --output teacher_prompt_previews/all_teacher_prompts.txt
```

This is a dry test: it uses deterministic representative traces and the production
prompt-building code, but does not call Gemini or start Docker. Each section prints the
exact `system_instruction`, `contents`, and generation configuration that would be sent
for that fixture. Use `--format json` for structured output, or filter with repeatable
`--workflow`, `--prompt`, and `--placement` flags.

Then launch any group directly. Failures are recorded in `run_status.jsonl` and the
individual configuration log, and the remaining configurations continue running.

```bash
python -m workflows.runner --workflow-type WORKFLOW_TYPE [OPTIONS]
```

Available workflow types:
- `detect_workflow`:
- `exploit_workflow`:
- `patch_workflow`:
- `detect_patch_workflow`:
- `exploit_patch_workflow`:

Required flags vary by workflow type.

Examples:

1. **Detect Workflow**:
```bash
python -m workflows.runner --workflow-type detect_workflow \
    --task_dir bountytasks/lunary \
    --bounty_number 0 \
    --model anthropic/claude-3-5-sonnet-20241022 \
    --phase_iterations 3
```

2. **Exploit Workflow**:
```bash
python -m workflows.runner --workflow-type exploit_workflow \
    --task_dir bountytasks/django \
    --bounty_number 0 \
    --model anthropic/claude-3-sonnet-20240229 \
    --phase_iterations 2 \
    --use_helm
```

3. **Patch Workflow**:
```bash
python -m workflows.runner --workflow-type patch_workflow \
    --task_dir bountytasks/mlflow \
    --bounty_number 1 \
    --use_mock_model \
    --phase_iterations 5
```

Please be aware that there may be a brief delay between initiating the workflow and observing the first log outputs (typically a few seconds). This initial pause is primarily due to the time required for importing necessary Python packages and initializing the environment.

### Running the Workflows through Web Application

1. In the root directory run:

```bash
npm install
npm start
```

This will launch the development server for the frontend and start the backend. You may need to refresh as the backend takes a second to run.

Once both the backend and frontend are running, you can access the application through your web browser (default `localhost:3000`)

### Dockerize run

1. Open the Docker Desktop app and ensure it's running.

2. Create a Docker volume for DinD data

   ```bash
   docker volume create dind-data
   ```

3. Navigate to the `bountybench` directory and run:

   ```bash
   docker compose up --build -d
   ```

Once built, the frontend will be running at http://localhost:3000/, and everything should be the same as in non-dockerized versions.

To stop the containers, run
```
docker compose down
```

To start the containers without rebuilding, run:
```
docker compose up -d
```
If docker still attempts to rebuild, try cancelling the build using `control+c` and adding the `--no-build` flag (assuming no images are missing).

To exec into the container, run:
```
docker exec -it backend-service bash
```

Then follow [Running Workflows](#running-workflows).


### Troubleshooting

#### Docker Mount Issue

**Error Message:**
Internal Server Error ("Mounts denied: The path *** is not shared from the host and is not known to Docker. You can configure shared paths from Docker -> Preferences... -> Resources -> File Sharing.")

**Solution:**
To resolve this issue, add the absolute path of your `bountybench` directory to Docker's shared paths. Follow these steps:

1. **Determine the Absolute Path:**
   - Open your terminal.
   - Navigate to the root directory of your project.
   - Retrieve the absolute path using the `pwd` command.
   - **Example Output:**

     ```bash
     /Users/yourusername/projects/bountybench
     ```

2. **Add the Path to Docker's Shared Paths:**
   - Open **Docker Desktop** on your machine.
   - Click on the **Settings** (gear) icon.
   - Navigate to **Resources** > **File Sharing**.
   - Paste the absolute path you obtained earlier (e.g., `/Users/yourusername/projects/bountybench`).
   - Click the **`+`** button to add the new shared path.
   - Also add `/tmp` using the **`+`** button.
   - Click **Apply & Restart** to save the changes.

3. **Verify the Configuration:**
   - After Docker restarts, try running your `bountybench` workflow again.
   - The error should be resolved, allowing Docker to access the necessary directories.

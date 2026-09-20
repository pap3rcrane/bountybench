#!/usr/bin/env python3
"""Prepare four VM input bundles for the one-repeat teacher split."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


PROMPTS = {
    "observe": (
        "single_error_correction_one_alternative",
        "single_error_correction_multiple_alternatives",
        "optimizer",
        "principle_extraction",
        "student_nudging",
    ),
    "steer": (
        "outcome_evaluation",
        "teacher_answer_comparison",
        "comparative_ranking",
        "reasoning_quality_fidelity",
        "rubric_based_evaluation",
    ),
    "objective_rewrite": (
        "increased_difficulty_task",
        "weakness_targeting_task",
        "strategy_limiter_task",
        "justification_gap_task",
        "oversight_task",
    ),
}

ENVIRONMENTS = {
    "observe": (
        ("LibreChat", "0", "exploit_workflow"),
        ("InvokeAI", "0", "exploit_workflow"),
        ("agentscope", "0", "exploit_workflow"),
        ("astropy", "0", "detect_workflow"),
        ("django", "0", "detect_workflow"),
        ("gpt_academic", "0", "detect_workflow"),
        ("kedro", "0", "patch_workflow"),
        ("scikit-learn", "0", "patch_workflow"),
        ("yaml", "0", "patch_workflow"),
    ),
    "steer": (
        ("mlflow", "0", "exploit_workflow"),
        ("lunary", "0", "exploit_workflow"),
        ("fastapi", "0", "exploit_workflow"),
        ("curl", "0", "detect_workflow"),
        ("gluon-cv", "0", "detect_workflow"),
        ("gunicorn", "0", "detect_workflow"),
        ("llama_index", "0", "patch_workflow"),
        ("setuptools", "0", "patch_workflow"),
        ("zipp", "0", "patch_workflow"),
    ),
}

SPLITS = {
    "vm1": {
        "observe": {"LibreChat", "agentscope", "astropy", "django"},
        "steer": {"mlflow", "lunary", "fastapi", "curl"},
        "objective_rewrite": {
            "gradio",
            "composio",
            "node",
            "bentoml",
            "langchain",
        },
    },
    "vm2": {
        "observe": {"gpt_academic", "kedro", "scikit-learn", "yaml"},
        "steer": {"gluon-cv", "gunicorn", "setuptools", "zipp"},
        "objective_rewrite": {
            "pytorch-lightning",
            "paddle",
            "parse-url",
            "undici",
        },
    },
}

TASK_WORKFLOWS = {
    "gradio": "exploit_workflow",
    "composio": "exploit_workflow",
    "node": "exploit_workflow",
    "bentoml": "detect_workflow",
    "langchain": "detect_workflow",
    "pytorch-lightning": "detect_workflow",
    "paddle": "patch_workflow",
    "parse-url": "patch_workflow",
    "undici": "patch_workflow",
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def exclusion_records(split: str, placement: str) -> list[dict]:
    records = []
    saved_placement = "prepend" if placement == "user" else "system"
    for mode in ("observe", "steer"):
        selected = SPLITS[split][mode]
        for repository, bounty, workflow in ENVIRONMENTS[mode]:
            for prompt in PROMPTS[mode]:
                for repetition in range(1, 6):
                    if repository in selected and repetition == 1:
                        continue
                    records.append(
                        {
                            "record_type": "configuration",
                            "teacher_type": mode,
                            "repo_name": repository,
                            "bounty_number": bounty,
                            "workflow_type": workflow,
                            "system_prompt_name": prompt,
                            "system_prompt_placement": saved_placement,
                            "run_number": repetition,
                        }
                    )
            for repetition in range(1, 6):
                records.append(
                    {
                        "record_type": "configuration",
                        "teacher_type": mode,
                        "repo_name": repository,
                        "bounty_number": bounty,
                        "workflow_type": workflow,
                        "system_prompt_name": "none",
                        "system_prompt_placement": "none",
                        "run_number": repetition,
                    }
                )
    return records


def task_plan(
    *,
    source_root: Path,
    bundle_root: Path,
    bundle_name: str,
    split: str,
    placement: str,
) -> list[dict]:
    records = []
    saved_placement = "prepend" if placement == "user" else "system"
    for repository in sorted(SPLITS[split]["objective_rewrite"]):
        workflow = TASK_WORKFLOWS[repository]
        environment_name = f"{repository}_bounty_0_{workflow}"
        for prompt in PROMPTS["objective_rewrite"]:
            configuration_name = (
                f"{environment_name}__{placement}__task_designer__{prompt}"
            )
            source_directory = source_root / environment_name / configuration_name
            destination_directory = (
                bundle_root / "sources" / environment_name / configuration_name
            )
            destination_directory.mkdir(parents=True, exist_ok=True)
            relative_logs = []
            for source_number in range(1, 4):
                source = source_directory / f"run1_source_{source_number}.json"
                if not source.is_file():
                    raise FileNotFoundError(f"Missing source trace: {source}")
                destination = destination_directory / source.name
                shutil.copy2(source, destination)
                relative_logs.append(
                    str(
                        Path("matrix_inputs")
                        / bundle_name
                        / "sources"
                        / environment_name
                        / configuration_name
                        / source.name
                    )
                )
            records.append(
                {
                    "record_type": "configuration",
                    "teacher_type": "objective_rewrite",
                    "repo_name": repository,
                    "bounty_number": "0",
                    "workflow_type": workflow,
                    "system_prompt_name": prompt,
                    "system_prompt_placement": saved_placement,
                    "run_number": 1,
                    "system_prompt_file": f"prompts/system_prompts/{prompt}.txt",
                    "source_logs": relative_logs,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "Desktop" / "bountybench",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("matrix_manifests/four_vm_once"),
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    staging_root = output_root / "staging" / "matrix_inputs"
    bundles_root = output_root / "bundles"
    shutil.rmtree(output_root / "staging", ignore_errors=True)
    bundles_root.mkdir(parents=True, exist_ok=True)

    definitions = (
        ("system_vm1", "vm1", "system"),
        ("system_vm2", "vm2", "system"),
        ("user_vm3", "vm1", "user"),
        ("user_vm4", "vm2", "user"),
    )
    for bundle_name, split, placement in definitions:
        bundle_root = staging_root / bundle_name
        bundle_root.mkdir(parents=True, exist_ok=True)
        retry_helper = Path(__file__).resolve().parent / "retry_objective_rewrites_from_plan.py"
        if not retry_helper.is_file():
            raise FileNotFoundError(f"Missing Task Designer retry helper: {retry_helper}")
        shutil.copy2(
            retry_helper,
            bundle_root / "run_task_designer_from_saved_sources.py",
        )
        exclusions = exclusion_records(split, placement)
        plan = task_plan(
            source_root=source_root,
            bundle_root=bundle_root,
            bundle_name=bundle_name,
            split=split,
            placement=placement,
        )
        write_jsonl(bundle_root / "exclude.jsonl", exclusions)
        write_jsonl(bundle_root / "task_designer_plan.jsonl", plan)

        archive = bundles_root / f"{bundle_name}_inputs.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            tar.add(bundle_root, arcname=f"matrix_inputs/{bundle_name}")
        print(
            f"{archive}: {len(exclusions)} exclusions, "
            f"{len(plan)} Task Designer configurations"
        )

    shutil.rmtree(output_root / "staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

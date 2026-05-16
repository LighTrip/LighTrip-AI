from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_GEMMA_ENV_DEFAULTS = {
    "GEMMA_MODEL_PATH": "models/gemma-4-E2B-it-Q4_K_S.gguf",
    "GEMMA_MMPROJ_PATH": "models/mmproj-F16.gguf",
    "GEMMA_PROMPT_PATH": "configs/draft_prompt_boundary_v2.txt",
    "GEMMA_N_CTX": "2048",
    "GEMMA_MAX_TOKENS": "160",
    "GEMMA_TEMPERATURE": "0.2",
    "GEMMA_TOP_P": "0.9",
    "GEMMA_TOP_K": "40",
    "GEMMA_REPEAT_PENALTY": "1.1",
    "GEMMA_STOP_TOKENS": "<end_of_turn>",
    "GEMMA_N_GPU_LAYERS": "-1",
    "GEMMA_MAIN_GPU": "0",
    "GEMMA_OFFLOAD_KQV": "true",
    "GEMMA_MMPROJ_USE_GPU": "true",
    "GEMMA_VERBOSE": "false",
    "GGML_CUDA_DISABLE_GRAPHS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "경계 보강 프롬프트(configs/draft_prompt_boundary_v2.txt)로 Places365 초안을 재생성하고 "
            "TF-IDF + Linear SVM을 재학습하는 실험 runner입니다."
        )
    )
    parser.add_argument(
        "--draft-output",
        type=Path,
        default=Path("data/category_classifier/places365_v1/interim/places365_service_prompt_drafts.jsonl"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/category_classifier/places365_v1/processed_service_prompt"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("experiments/gemma_category_compare/artifacts/service_prompt_classifier"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("experiments/gemma_category_compare/reports/service_prompt_classifier"),
    )
    parser.add_argument(
        "--compare-output-dir",
        type=Path,
        default=Path("experiments/gemma_category_compare/results/service_prompt_vs_direct"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument(
        "--categories",
        default="",
        help="생성할 카테고리 필터입니다. 예: 카페,식당,공원",
    )
    parser.add_argument("--overwrite-drafts", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--run-compare",
        action="store_true",
        help="재학습 artifact로 Gemma direct 및 split pipeline 비교까지 실행합니다.",
    )
    parser.add_argument(
        "--compare-limit",
        type=int,
        default=0,
        help="--run-compare에서 앞 N개만 비교합니다. 0이면 전체 test split입니다.",
    )
    parser.add_argument(
        "--safe-hybrid",
        action="store_true",
        help="GPU OOM 방지를 위해 일부 layer만 GPU에 올리고 mmproj/KQV는 CPU로 둡니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="명령어만 출력하고 실행하지 않습니다.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    defaults = dict(SERVICE_GEMMA_ENV_DEFAULTS)
    if args.safe_hybrid:
        defaults.update(
            {
                "GEMMA_N_GPU_LAYERS": "8",
                "GEMMA_OFFLOAD_KQV": "false",
                "GEMMA_MMPROJ_USE_GPU": "false",
            }
        )

    for name, value in defaults.items():
        env.setdefault(name, value)
    return env


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_step(
    name: str,
    command: list[str],
    *,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    print(f"\n=== {name} ===")
    print(command_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def generate_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/dataset/generate_places365_drafts.py",
        "--output",
        str(args.draft_output),
        "--no-category-hint",
        "--include-metadata",
        "--include-prompt",
        "--full-gpu",
    ]
    if args.overwrite_drafts:
        command.append("--overwrite")
    if args.limit_total:
        command.extend(["--limit-total", str(args.limit_total)])
    if args.limit_per_category:
        command.extend(["--limit-per-category", str(args.limit_per_category)])
    if args.categories:
        command.extend(["--categories", args.categories])
    return command


def split_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/dataset/split_places365_dataset.py",
        "--input",
        str(args.draft_output),
        "--output-dir",
        str(args.processed_dir),
        "--seed",
        str(args.seed),
        "--valid-ratio",
        str(args.valid_ratio),
        "--test-ratio",
        str(args.test_ratio),
    ]


def train_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "experiments/category_classifier/train.py",
        "--model",
        "linear_svm",
        "--train",
        str(args.processed_dir / "train.jsonl"),
        "--valid",
        str(args.processed_dir / "valid.jsonl"),
        "--test",
        str(args.processed_dir / "test.jsonl"),
        "--artifact-dir",
        str(args.artifact_dir),
        "--report-dir",
        str(args.report_dir),
    ]


def compare_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "experiments/gemma_category_compare/run_compare.py",
        "--input-jsonl",
        str(args.processed_dir / "test.jsonl"),
        "--classifier-artifact",
        str(args.artifact_dir / "linear_svm_tfidf.joblib"),
        "--gemma-prompt",
        "configs/draft_prompt_boundary_v2.txt",
        "--run-gemma-direct",
        "--run-split-pipeline",
        "--output-dir",
        str(args.compare_output_dir),
    ]
    if args.compare_limit:
        command.extend(["--limit", str(args.compare_limit)])
    return command


def print_summary(args: argparse.Namespace) -> None:
    summary = {
        "draft_output": str(resolve(args.draft_output)),
        "processed_dir": str(resolve(args.processed_dir)),
        "artifact": str(resolve(args.artifact_dir / "linear_svm_tfidf.joblib")),
        "train_metrics": str(resolve(args.report_dir / "linear_svm_metrics.json")),
        "compare_output_dir": str(resolve(args.compare_output_dir)),
    }
    print("\n=== outputs ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    env = build_env(args)

    if not args.skip_generate:
        run_step("1. generate service-prompt drafts", generate_command(args), env=env, dry_run=args.dry_run)
    if not args.skip_split:
        run_step("2. stratified split", split_command(args), env=env, dry_run=args.dry_run)
    if not args.skip_train:
        run_step("3. train linear SVM", train_command(args), env=env, dry_run=args.dry_run)
    if args.run_compare:
        run_step("4. compare retrained SVM vs Gemma direct", compare_command(args), env=env, dry_run=args.dry_run)

    print_summary(args)


if __name__ == "__main__":
    main()

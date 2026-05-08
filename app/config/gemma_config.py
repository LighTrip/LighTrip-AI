from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TRUE_VALUES: Final[set[str]] = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GemmaPaths:
    model_path: str
    mmproj_path: str
    prompt_path: str


@dataclass(frozen=True)
class GemmaModelConfig:
    n_ctx: int
    n_gpu_layers: int
    main_gpu: int
    offload_kqv: bool
    mmproj_use_gpu: bool


@dataclass(frozen=True)
class GemmaGenerationConfig:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    stop_tokens: list[str]

    def as_chat_completion_kwargs(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stop": list(self.stop_tokens),
        }


@dataclass(frozen=True)
class GemmaConfig:
    paths: GemmaPaths
    model: GemmaModelConfig
    generation: GemmaGenerationConfig


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"필수 환경변수가 설정되지 않았습니다: {name}")
    return value.strip()


def required_path_env(name: str, project_root: Path = PROJECT_ROOT) -> str:
    path = Path(required_env(name))
    if path.is_absolute():
        return str(path)
    return str(project_root / path)


def required_int_env(name: str) -> int:
    return int(required_env(name))


def required_float_env(name: str) -> float:
    return float(required_env(name))


def required_bool_env(name: str) -> bool:
    return required_env(name).lower() in TRUE_VALUES


def required_list_env(name: str) -> list[str]:
    return [
        item.strip()
        for item in required_env(name).split(",")
        if item.strip()
    ]


def load_gemma_config(project_root: Path = PROJECT_ROOT) -> GemmaConfig:
    return GemmaConfig(
        paths=GemmaPaths(
            model_path=required_path_env("GEMMA_MODEL_PATH", project_root),
            mmproj_path=required_path_env("GEMMA_MMPROJ_PATH", project_root),
            prompt_path=required_path_env("GEMMA_PROMPT_PATH", project_root),
        ),
        model=GemmaModelConfig(
            n_ctx=required_int_env("GEMMA_N_CTX"),
            n_gpu_layers=required_int_env("GEMMA_N_GPU_LAYERS"),
            main_gpu=required_int_env("GEMMA_MAIN_GPU"),
            offload_kqv=required_bool_env("GEMMA_OFFLOAD_KQV"),
            mmproj_use_gpu=required_bool_env("GEMMA_MMPROJ_USE_GPU"),
        ),
        generation=GemmaGenerationConfig(
            max_tokens=required_int_env("GEMMA_MAX_TOKENS"),
            temperature=required_float_env("GEMMA_TEMPERATURE"),
            top_p=required_float_env("GEMMA_TOP_P"),
            top_k=required_int_env("GEMMA_TOP_K"),
            repeat_penalty=required_float_env("GEMMA_REPEAT_PENALTY"),
            stop_tokens=required_list_env("GEMMA_STOP_TOKENS"),
        ),
    )


GEMMA_CONFIG: Final[GemmaConfig] = load_gemma_config()

from __future__ import annotations

from typing import Final, Optional

import numpy as np
from llama_cpp import Llama

from app.config.gemma_config import GEMMA_CONFIG


# 임베딩 인스턴스는 텍스트만 다루므로 mmproj(비전) 핸들러가 필요 없습니다.
# 같은 GGUF 를 재사용하되, 더 가볍게 가려면 MODEL_PATH 를 EmbeddingGemma GGUF 로
# 바꾸기만 하면 아래 코드는 그대로 동작합니다.
MODEL_PATH: Final[str] = GEMMA_CONFIG.paths.model_path
N_CTX: Final[int] = GEMMA_CONFIG.model.n_ctx
N_GPU_LAYERS: Final[int] = GEMMA_CONFIG.model.n_gpu_layers
MAIN_GPU: Final[int] = GEMMA_CONFIG.model.main_gpu
OFFLOAD_KQV: Final[bool] = GEMMA_CONFIG.model.offload_kqv

__all__ = [
    "embed_texts",
    "get_embedding_llm",
    "is_embedding_model_loaded",
    "load_embedding_model",
    "unload_embedding_model",
]

_embed_llm: Optional[Llama] = None


def get_embedding_llm() -> Optional[Llama]:
    return _embed_llm


def is_embedding_model_loaded() -> bool:
    return _embed_llm is not None


def unload_embedding_model() -> None:
    global _embed_llm
    _embed_llm = None


def load_embedding_model(verbose: bool = True) -> None:
    global _embed_llm

    if _embed_llm is not None:
        return

    # 생성용 _llm 과 별개 인스턴스. embedding=True 가 핵심.
    _embed_llm = Llama(
        model_path=MODEL_PATH,
        embedding=True,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        main_gpu=MAIN_GPU,
        offload_kqv=OFFLOAD_KQV,
        verbose=verbose,
    )


def _pool_and_normalize(token_vectors: list) -> list[float]:
    # 생성형 모델은 토큰별 벡터(2차원)를 주므로 mean 으로 직접 풀링 후 L2 정규화.
    arr = np.asarray(token_vectors, dtype=np.float32)   # (토큰수, 차원)
    pooled = arr.mean(axis=0)                            # (차원,)
    norm = float(np.linalg.norm(pooled))
    if norm > 0.0:
        pooled = pooled / norm
    return pooled.tolist()


def embed_texts(llm: Llama, texts: list[str]) -> list[list[float]]:
    out = llm.create_embedding(texts)

    vectors: list[list[float]] = []
    for item in out["data"]:
        emb = item["embedding"]
        # 토큰별 벡터(2차원)면 직접 풀링
        if emb and isinstance(emb[0], list):
            vectors.append(_pool_and_normalize(emb))
        # 이미 풀링된 1차원 벡터면 정규화만
        else:
            v = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(v))
            vectors.append((v / n if n > 0.0 else v).tolist())

    return vectors

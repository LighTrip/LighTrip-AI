"""FastAPI 앱 진입점.

서버 시작/종료 시 모델 생명주기를 관리하고 실제 API 라우터를 등록한다.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.embedding import router as embedding_router
from app.services.embedding_service import (
    is_embedding_model_loaded,
    load_embedding_model,
    unload_embedding_model,
)

from app.api.pipeline import router as pipeline_router

from app.services.category_service import (
    is_category_model_loaded,
    load_category_model,
    unload_category_model,
)

from app.services.gemma_service import load_model, unload_model, is_model_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델은 요청마다 다시 로드하지 않고 앱 시작 시 한 번만 올린다.
    load_model()
    load_category_model()
    load_embedding_model()
    yield
    # 종료 시 전역 모델 참조를 비워 재시작/테스트 환경에서 상태가 섞이지 않게 한다.
    unload_embedding_model()
    unload_category_model()
    unload_model()


app = FastAPI(
    title="LighTrip AI API",
    description="이미지 기반 블로그 초안 및 카테고리 생성 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(pipeline_router)
app.include_router(embedding_router)    

@app.get("/")
async def root():
    return {"message": "LighTrip AI Pipeline API is running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemma_model_loaded": is_model_loaded(),
        "category_model_loaded": is_category_model_loaded(),
        "embedding_model_loaded": is_embedding_model_loaded(),
    }

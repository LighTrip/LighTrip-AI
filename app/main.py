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
    load_model()
    load_category_model()
    load_embedding_model()
    yield
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

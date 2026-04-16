from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.gemma import router as gemma_router
from app.services.gemma_service import load_model, unload_model, is_model_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


app = FastAPI(
    title="Gemma4 Vision API",
    description="이미지 + 사용자 프롬프트 기반 블로그 초안 생성 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(gemma_router)


@app.get("/")
async def root():
    return {"message": "Gemma4 Vision API is running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
    }
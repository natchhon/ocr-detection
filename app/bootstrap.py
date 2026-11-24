from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.infrastructure.ocr.factory import OCRProviderFactory
from app.application.services.id_card_service import IDCardService


@asynccontextmanager
async def lifespan(app: FastAPI):

    settings = get_settings()

    provider = OCRProviderFactory.create(settings)

    await run_in_threadpool(provider.load)

    app.state.ocr_provider = provider

    app.state.id_card_service = IDCardService(
        ocr_provider=provider
    )

    try:
        yield

    finally:
        await run_in_threadpool(provider.close)

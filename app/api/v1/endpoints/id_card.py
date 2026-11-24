import io

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    HTTPException
)

from PIL import Image, ImageOps, UnidentifiedImageError

from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.api.dependencies import get_id_card_service
from app.application.services.id_card_service import IDCardService


router = APIRouter(
    prefix="/id-card",
    tags=["ID Card"]
)


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    service: IDCardService = Depends(
        get_id_card_service
    )
):

    settings = get_settings()

    content = await file.read(
        settings.MAX_FILE_SIZE + 1
    )

    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")

    try:

        with Image.open(io.BytesIO(content)) as source:

            if source.format not in {"JPEG", "PNG"}:
                raise HTTPException(415, "Unsupported image")

            if source.width * source.height > 20_000_000:
                raise HTTPException(413, "Image too large")

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

        image.thumbnail((2400, 2400))

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError
    ):
        raise HTTPException(400, "Invalid image")

    result = await run_in_threadpool(
        service.detect,
        image
    )

    return {
        "success": True,
        "data": result
    }

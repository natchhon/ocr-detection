import hashlib
import io
import warnings

from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def load_image(content: bytes, max_pixels: int) -> tuple[Image.Image, dict]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG"} or getattr(source, "n_frames", 1) != 1:
                    raise ImageValidationError(415, "Unsupported image")
                if source.width * source.height > max_pixels:
                    raise ImageValidationError(413, "Image too large")
                diagnostics = {
                    "width": source.width,
                    "height": source.height,
                    "format": source.format,
                    "file_size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                source.load()
                image = source.copy()
                return image, diagnostics
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(413, "Image too large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError(400, "Invalid image") from exc

from app.domain.ports.image_processor import ImageProcessor
from app.infrastructure.preprocessing.original_processor import OriginalProcessor


def create_processor(name: str, max_dimension: int = 2400) -> ImageProcessor:
    if name == "original":
        return OriginalProcessor(max_dimension)
    if name == "grayscale":
        from app.infrastructure.preprocessing.grayscale_processor import GrayscaleProcessor

        return GrayscaleProcessor(max_dimension)
    if name == "clahe":
        from app.infrastructure.preprocessing.clahe_processor import CLAHEProcessor

        return CLAHEProcessor(max_dimension)
    if name == "khmer":
        from app.infrastructure.preprocessing.khmer_processor import KhmerProcessor

        return KhmerProcessor(max_dimension)
    raise ValueError(f"Unsupported image processor: {name}")

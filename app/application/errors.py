class TemplateMatchError(ValueError):
    """The selected layout cannot be applied safely; OCR has not run."""

    def __init__(self, reason: str, template: str | None, card_detected: bool):
        super().__init__(
            "Could not locate a card matching the selected template. "
            "Use the original card image with visible borders and the correct layout template."
        )
        self.reason = reason
        self.template = template
        self.card_detected = card_detected

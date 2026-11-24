from fastapi import Request

from app.application.services.id_card_service import IDCardService


def get_id_card_service(request: Request) -> IDCardService:
    return request.app.state.id_card_service

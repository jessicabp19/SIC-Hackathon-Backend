"""
Rutas de la API para el chatbot educativo.
"""
from fastapi import APIRouter
from typing import List

from app.models.schemas import ChatbotRequest, ChatbotResponse
from app.services.chatbot_service import chatbot_service

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


@router.post(
    "/message",
    response_model=ChatbotResponse
)
async def send_message(request: ChatbotRequest):
    """
    Envía un mensaje al chatbot educativo.

    El chatbot puede responder preguntas sobre:
    - Conceptos básicos de inversión
    - Riesgo y volatilidad
    - Diversificación
    - Métricas (Sharpe, rendimiento)
    - Cómo funciona esta herramienta
    - Estrategias de inversión

    **Nota:** El chatbot es educativo y NO proporciona asesoría financiera.
    """
    result = chatbot_service.responder(request.message)

    return ChatbotResponse(
        response=result["response"],
        categoria=result["categoria"]
    )


@router.get("/suggestions")
async def get_suggestions() -> List[str]:
    """
    Obtiene preguntas sugeridas para el usuario.

    Útil para mostrar en el frontend como opciones rápidas.
    """
    return chatbot_service.obtener_preguntas_sugeridas()


@router.get("/categories")
async def get_categories() -> List[str]:
    """
    Obtiene las categorías de conocimiento disponibles.
    """
    return chatbot_service.obtener_categorias()


@router.get("/health")
async def health_check():
    """Verifica que el servicio de chatbot esté funcionando."""
    return {"status": "ok", "service": "chatbot"}

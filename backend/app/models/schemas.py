"""
Schemas Pydantic para validación de requests y responses.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# === REQUEST SCHEMAS ===

class AnalyzeRequest(BaseModel):
    """Request para analizar y optimizar portafolio."""
    tickers: List[str] = Field(
        ...,
        min_length=2,
        description="Lista de tickers o nombres de empresas (mínimo 2)",
        examples=[["AAPL", "MSFT", "TSLA"]]
    )


class ChatbotRequest(BaseModel):
    """Request para el chatbot educativo."""
    message: str = Field(
        ...,
        min_length=1,
        description="Mensaje del usuario",
        examples=["¿Qué es el riesgo?"]
    )


# === RESPONSE SCHEMAS ===

class TickerMatch(BaseModel):
    """Un resultado de búsqueda de ticker."""
    nombre: str
    ticker: str


class SearchResponse(BaseModel):
    """Response de búsqueda de tickers."""
    matches: List[TickerMatch]


class ParametrosProyectados(BaseModel):
    """Parámetros proyectados por activo."""
    ticker: str
    drift_anual: float = Field(..., description="Drift (μ) anualizado en porcentaje")
    volatilidad_anual: float = Field(..., description="Volatilidad (σ) anualizada en porcentaje")


class MetricasValidacion(BaseModel):
    """Métricas de validación del modelo."""
    rmse_modelo: float
    rmse_baseline: float
    ganancia_vs_buy_hold: float = Field(..., description="Ganancia vs Buy&Hold en porcentaje")


class AnalyzeResponse(BaseModel):
    """Response del análisis de portafolio."""
    success: bool
    pesos_optimos: Dict[str, float] = Field(
        ...,
        description="Distribución óptima de pesos por ticker (0-1)"
    )
    sharpe_ratio: float
    parametros_proyectados: List[ParametrosProyectados]
    metricas_validacion: MetricasValidacion
    tiempo_ejecucion: float = Field(..., description="Tiempo de ejecución en segundos")
    mensaje: Optional[str] = None


class ChatbotResponse(BaseModel):
    """Response del chatbot educativo."""
    response: str
    categoria: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response de error."""
    success: bool = False
    error: str
    detalle: Optional[str] = None

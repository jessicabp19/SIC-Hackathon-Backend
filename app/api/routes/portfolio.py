"""
Rutas de la API para optimización de portafolios.
"""
import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    SearchResponse,
    TickerMatch,
    ParametrosProyectados,
    MetricasValidacion,
    ErrorResponse
)
from app.services.data_service import data_service
from app.services.lstm_service import lstm_service
from app.services.montecarlo_service import montecarlo_service
from app.services.optimizer_service import optimizer_service

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def analyze_portfolio(request: AnalyzeRequest):
    """
    Analiza y optimiza un portafolio de inversión.

    Proceso:
    1. Resuelve tickers (nombres → símbolos)
    2. Descarga datos históricos de Yahoo Finance
    3. Entrena/usa modelo LSTM para proyección
    4. Ejecuta simulación Monte Carlo
    5. Optimiza pesos del portafolio (Sharpe Ratio)
    6. Valida resultados

    **Nota:** Primera ejecución puede tardar ~1-2 min por entrenamiento LSTM.
    Ejecuciones posteriores son más rápidas si el modelo es compatible.
    """
    start_time = time.time()

    try:
        # 1. Resolver tickers
        tickers = [
            data_service.resolver_ticker(t.strip())
            for t in request.tickers
        ]

        # 2. Descargar datos
        df_precios, df_rendimientos = data_service.descargar_datos(tickers)

        if df_precios is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudieron obtener datos válidos. Verifica los tickers o intenta con otros activos."
            )

        # 3. Preparar datos y entrenar/cargar modelo
        datos = lstm_service.preparar_datos(df_rendimientos)
        modelo = lstm_service.entrenar_modelo(datos)

        if modelo is None:
            raise HTTPException(
                status_code=500,
                detail="Error al cargar o entrenar el modelo LSTM."
            )

        # 4. Simulación Monte Carlo
        ultimos_precios = df_precios.iloc[-1].values
        rend_inv, precios_sim, mu, sigma = montecarlo_service.proyectar_y_simular(
            modelo, datos, ultimos_precios
        )

        # 5. Optimización
        sharpe, pesos_optimos = optimizer_service.optimizar_portafolio(
            precios_sim, datos["names"]
        )

        # 6. Validación
        metricas = lstm_service.calcular_metricas_validacion(datos, pesos_optimos)

        # 7. Formatear respuesta
        parametros = montecarlo_service.obtener_parametros_proyectados(
            datos["names"], mu, sigma
        )

        tiempo_total = time.time() - start_time

        return AnalyzeResponse(
            success=True,
            pesos_optimos=pesos_optimos,
            sharpe_ratio=float(sharpe),
            parametros_proyectados=[
                ParametrosProyectados(**p) for p in parametros
            ],
            metricas_validacion=MetricasValidacion(
                rmse_modelo=metricas["rmse_modelo"],
                rmse_baseline=metricas["rmse_baseline"],
                ganancia_vs_buy_hold=metricas["ganancia_vs_buy_hold"]
            ),
            tiempo_ejecucion=tiempo_total,
            mensaje=f"Análisis completado para {len(datos['names'])} activos."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el análisis: {str(e)}"
        )


@router.get(
    "/search",
    response_model=SearchResponse
)
async def search_tickers(query: str, limit: int = 5):
    """
    Busca empresas del S&P 500 por nombre.

    Usa fuzzy matching para encontrar coincidencias parciales.
    Útil cuando el usuario no conoce el ticker exacto.
    """
    if not query or len(query) < 2:
        raise HTTPException(
            status_code=400,
            detail="La búsqueda debe tener al menos 2 caracteres."
        )

    matches = data_service.buscar_ticker(query, limit=limit)

    return SearchResponse(
        matches=[
            TickerMatch(nombre=m["nombre"], ticker=m["ticker"])
            for m in matches
        ]
    )


@router.get("/health")
async def health_check():
    """Verifica que el servicio esté funcionando."""
    return {"status": "ok", "service": "portfolio"}

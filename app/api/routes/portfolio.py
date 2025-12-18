"""
Rutas de la API para optimización de portafolios.
"""
import time
import numpy as np
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
from app.services.montecarlo_service import montecarlo_service

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
        # 1. Resolver tickers - Convertir nombres a tickers
        if isinstance(request.tickers, str):
            tickers = data_service.buscar_tickers(request.tickers)
        else:
            tickers = [
                data_service.resolver_ticker(t.strip())
                for t in request.tickers
            ]

        if len(tickers) < 2:
            raise HTTPException(
                status_code=400,
                detail="Se requieren al menos 2 activos válidos."
            )

        # 2. Descargar datos
        df_precios, df_rendimientos = data_service.descargar_datos(tickers)

        if df_precios is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudieron obtener datos válidos. Verifica los tickers o intenta con otros activos."
            )

        # 3. Preparar datos y entrenar/cargar modelo
        X, Y = data_service.preparar_secuencias(df_rendimientos)
        
        # Dividir datos en train/test
        div = int(len(X) * 0.8)
        X_train, X_test = X[:div], X[div:]
        Y_train, Y_test = Y[:div], Y[div:]

        n_features = df_rendimientos.shape[1]
        
        # Inicializar servicio LSTM con número de features
        from app.services.lstm_service import LSTMService
        lstm_srv = LSTMService(n_features=n_features)

        # Preparar datos en diccionario
        datos = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test,
            "scaler": data_service.scaler,
            "n_features": n_features,
            "names": df_rendimientos.columns.tolist(),
            "df_rend": df_rendimientos,
            "index_test": len(df_rendimientos) - len(X) + div
        }

        modelo = lstm_srv.entrenar_modelo(datos)

        if modelo is None:
            raise HTTPException(
                status_code=500,
                detail="Error al cargar o entrenar el modelo LSTM."
            )

        # 4. Predicción del próximo rendimiento esperado
        prediccion_scaled = lstm_srv.predecir_futuro(X_test[-1])
        prediccion_real = data_service.scaler.inverse_transform(
            prediccion_scaled.reshape(1, -1)
        )[0]

        # 5. Optimización usando scipy.optimize
        ultimos_precios = df_precios.iloc[-1].values
        sharpe, pesos_optimos = montecarlo_service.optimizar_cartera(
            df_rendimientos, prediccion_real, datos["names"]
        )

        # 6. Stress test Monte Carlo con VaR
        rend_inv, precios_sim, mu, sigma = montecarlo_service.proyectar_y_simular(
            modelo, datos, ultimos_precios
        )
        var_95 = montecarlo_service.stress_test_monte_carlo(
            precios_sim, np.array(list(pesos_optimos.values()))
        )

        # 7. Validación del modelo
        metricas = lstm_srv.calcular_metricas_validacion(datos, pesos_optimos)

        # 8. Preparar parámetros proyectados
        parametros = [
            {
                "ticker": datos["names"][i],
                "drift_anual": float(mu[i] * 100),
                "volatilidad_anual": float(sigma[i] * 100)
            }
            for i in range(len(datos["names"]))
        ]

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
            mensaje=f"Análisis completado para {len(datos['names'])} activos. VaR 95%: {var_95*100:.2f}%"
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

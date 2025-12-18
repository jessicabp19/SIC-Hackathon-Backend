"""
Servicio Monte Carlo: simulación de trayectorias de precios y optimización de cartera.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from scipy.optimize import minimize

from tensorflow.keras.models import Sequential

from app.config import (
    DIAS_TRADING_ANUALES,
    TAMANO_VENTANA,
    N_RUTAS_MC,
    DIAS_PROYECCION
)


class MonteCarloService:
    """Servicio para simulación Monte Carlo de precios."""

    def proyectar_y_simular(
        self,
        modelo: Sequential,
        datos: Dict[str, Any],
        ultimos_precios: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Proyecta rendimientos con LSTM y simula trayectorias con Monte Carlo.

        Args:
            modelo: Modelo LSTM entrenado
            datos: Diccionario con datos preparados (incluye scaler y X_test)
            ultimos_precios: Array con los últimos precios de cada activo

        Returns:
            Tuple de (rendimientos_proyectados, precios_simulados, mu, sigma)
        """
        if modelo is None:
            raise ValueError("Modelo LSTM no disponible")

        scaler = datos["scaler"]
        ventana = datos["X_test"][-1].copy()

        # Proyección LSTM iterativa
        forecast_scaled = []
        for _ in range(DIAS_PROYECCION):
            pred = modelo.predict(
                ventana.reshape(1, TAMANO_VENTANA, -1),
                verbose=0
            )[0]
            forecast_scaled.append(pred)
            ventana = np.vstack([ventana[1:], pred])

        # Invertir normalización
        rend_inv = scaler.inverse_transform(np.array(forecast_scaled))

        # Calcular parámetros anualizados
        mu = rend_inv.mean(axis=0) * DIAS_TRADING_ANUALES
        sigma = rend_inv.std(axis=0) * np.sqrt(DIAS_TRADING_ANUALES)

        # Simulación Monte Carlo con GBM
        precios = self._simular_gbm(ultimos_precios, mu, sigma)

        return rend_inv, precios, mu, sigma

    def _simular_gbm(
        self,
        precios_iniciales: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        Simula trayectorias de precios usando Geometric Brownian Motion.

        Args:
            precios_iniciales: Precios iniciales de cada activo
            mu: Drift anualizado de cada activo
            sigma: Volatilidad anualizada de cada activo

        Returns:
            Array de forma (dias+1, n_rutas, n_activos) con precios simulados
        """
        n_activos = len(precios_iniciales)
        dt = 1 / DIAS_TRADING_ANUALES

        # Inicializar array de precios
        precios = np.zeros((DIAS_PROYECCION + 1, N_RUTAS_MC, n_activos))
        precios[0] = precios_iniciales

        # Generar ruido aleatorio
        Z = np.random.normal(size=(DIAS_PROYECCION, N_RUTAS_MC, n_activos))

        # Parámetros del GBM
        drift = (mu - 0.5 * sigma**2) * dt
        stdev = sigma * np.sqrt(dt)

        # Simular trayectorias
        for t in range(1, DIAS_PROYECCION + 1):
            precios[t] = precios[t - 1] * np.exp(drift + stdev * Z[t - 1])

        return precios

    def calcular_matriz_covarianza(self, df_rendimientos: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la matriz de covarianza anualizada de los rendimientos históricos.

        Args:
            df_rendimientos: DataFrame con rendimientos logarítmicos

        Returns:
            Matriz de covarianza anualizada
        """
        return df_rendimientos.cov() * DIAS_TRADING_ANUALES

    def optimizar_cartera(
        self,
        df_rendimientos: pd.DataFrame,
        predicciones_ia: np.ndarray,
        nombres: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Encuentra los pesos óptimos del portafolio maximizando el Sharpe Ratio.
        Utiliza scipy.optimize.minimize para optimización determinista.

        Args:
            df_rendimientos: DataFrame con rendimientos históricos
            predicciones_ia: Array con rendimientos predichos por LSTM
            nombres: Lista de nombres/tickers de activos

        Returns:
            Tuple de (mejor_sharpe_ratio, diccionario_pesos_optimos)
        """
        cov_matrix = self.calcular_matriz_covarianza(df_rendimientos).values
        rend_esperados = predicciones_ia * DIAS_TRADING_ANUALES
        n_activos = len(nombres)

        def objetivo(weights):
            retorno_p = np.sum(rend_esperados * weights)
            vol_p = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = retorno_p / (vol_p + 1e-9)
            # Para distribución más suave
            penalizacion_estabilidad = 0.001 * np.sum(weights**2)
            return -(sharpe - penalizacion_estabilidad)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.02, 0.45) for _ in range(n_activos))
        iniciales = np.array([1. / n_activos] * n_activos)

        resultado = minimize(
            objetivo,
            iniciales,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-12
        )

        pesos_optimos = resultado.x if resultado.success else iniciales

        # Convertir a diccionario
        pesos_dict = {
            nombre: float(peso)
            for nombre, peso in zip(nombres, pesos_optimos)
        }

        # Calcular Sharpe ratio final
        retorno_final = np.sum(rend_esperados * pesos_optimos)
        vol_final = np.sqrt(np.dot(pesos_optimos.T, np.dot(cov_matrix, pesos_optimos)))
        sharpe_final = retorno_final / (vol_final + 1e-9)

        return sharpe_final, pesos_dict

    def stress_test_monte_carlo(
        self,
        precios_simulados: np.ndarray,
        pesos_optimos: np.ndarray
    ) -> float:
        """
        Calcula el Valor en Riesgo (VaR) 95% de la cartera.

        Args:
            precios_simulados: Array (dias, rutas, activos) con precios simulados
            pesos_optimos: Array con pesos de la cartera

        Returns:
            VaR 95% como porcentaje (ej: -0.05 para pérdida del 5%)
        """
        precios_iniciales = precios_simulados[0]
        precios_finales = precios_simulados[-1]

        # Rendimientos de la cartera total en cada simulación
        valor_inicial_cartera = np.dot(precios_iniciales, pesos_optimos)
        valor_final_cartera = np.dot(precios_finales, pesos_optimos)
        
        rendimientos_simulados = (valor_final_cartera / valor_inicial_cartera) - 1

        # Cálculo de Valor en Riesgo (VaR) 95%
        var_95 = np.percentile(rendimientos_simulados, 5)

        return var_95


# Instancia singleton del servicio
montecarlo_service = MonteCarloService()

"""
Servicio Monte Carlo: simulación de trayectorias de precios.
"""
import numpy as np
from typing import Dict, Any, Tuple, List

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

        # Proyección LSTM
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

    def obtener_parametros_proyectados(
        self,
        nombres: List[str],
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Formatea los parámetros proyectados para la respuesta.

        Args:
            nombres: Lista de nombres/tickers de activos
            mu: Drift anualizado
            sigma: Volatilidad anualizada

        Returns:
            Lista de diccionarios con parámetros por activo
        """
        return [
            {
                "ticker": nombres[i],
                "drift_anual": float(mu[i] * 100),
                "volatilidad_anual": float(sigma[i] * 100)
            }
            for i in range(len(nombres))
        ]


# Instancia singleton del servicio
montecarlo_service = MonteCarloService()

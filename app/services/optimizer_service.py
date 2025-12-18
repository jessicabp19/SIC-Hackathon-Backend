"""
Servicio de Optimización: búsqueda de pesos óptimos de cartera.
Utiliza simulación de Monte Carlo o scipy.optimize para encontrar pesos óptimos.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize

from app.config import N_SIMULACIONES, DIAS_TRADING_ANUALES


class OptimizerService:
    """Servicio para optimización de portafolios usando Sharpe Ratio."""

    def optimizar_portafolio(
        self,
        precios_simulados: np.ndarray,
        nombres: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Encuentra los pesos óptimos del portafolio maximizando el Sharpe Ratio.

        Usa simulación aleatoria de N_SIMULACIONES combinaciones de pesos
        evaluadas contra los escenarios de precios de Monte Carlo.

        Args:
            precios_simulados: Array (dias, rutas, activos) con precios simulados
            nombres: Lista de nombres/tickers de los activos

        Returns:
            Tuple de (mejor_sharpe_ratio, diccionario_pesos_optimos)
        """
        precios_final = precios_simulados[-1]  # Precios al final de la proyección
        precios_inicial = precios_simulados[0]  # Precios iniciales
        n_activos = precios_final.shape[1]

        mejor_sharpe = -999
        mejor_pesos = None

        for _ in range(N_SIMULACIONES):
            # Generar pesos aleatorios que sumen 1
            w = np.random.random(n_activos)
            w /= w.sum()

            # Calcular rendimiento y volatilidad del portafolio
            valor_final = precios_final @ w
            valor_inicial = precios_inicial @ w

            rendimiento = valor_final.mean() / valor_inicial.mean() - 1
            volatilidad = np.std(valor_final)

            # Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
            sharpe = rendimiento / volatilidad if volatilidad > 1e-6 else 0

            if sharpe > mejor_sharpe:
                mejor_sharpe = sharpe
                mejor_pesos = w

        # Convertir a diccionario
        pesos_dict = {
            nombre: float(peso)
            for nombre, peso in zip(nombres, mejor_pesos)
        }

        return mejor_sharpe, pesos_dict

    def optimizar_portafolio_scipy(
        self,
        df_rendimientos: pd.DataFrame,
        predicciones_ia: np.ndarray,
        nombres: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimización determinista usando scipy.optimize.minimize.
        Maximiza el Sharpe Ratio basándose en rendimientos predichos y covarianza histórica.

        Args:
            df_rendimientos: DataFrame con rendimientos logarítmicos históricos
            predicciones_ia: Array con rendimientos predichos por LSTM
            nombres: Lista de nombres/tickers de activos

        Returns:
            Tuple de (mejor_sharpe_ratio, diccionario_pesos_optimos)
        """
        cov_matrix = (df_rendimientos.cov() * DIAS_TRADING_ANUALES).values
        rend_esperados = predicciones_ia * DIAS_TRADING_ANUALES
        n_activos = len(nombres)

        def objetivo(weights):
            retorno_p = np.sum(rend_esperados * weights)
            vol_p = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = retorno_p / (vol_p + 1e-9)
            # Penalización para distribución más suave
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

    def formatear_pesos_porcentaje(
        self,
        pesos: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Formatea los pesos como strings de porcentaje.

        Args:
            pesos: Diccionario ticker → peso (0-1)

        Returns:
            Diccionario ticker → "XX.XX%"
        """
        return {
            ticker: f"{peso * 100:.2f}%"
            for ticker, peso in pesos.items()
        }


# Instancia singleton del servicio
optimizer_service = OptimizerService()

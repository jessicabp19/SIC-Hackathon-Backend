"""
Servicio de Optimización: búsqueda de pesos óptimos de cartera.
"""
import numpy as np
from typing import Dict, List, Tuple

from app.config import N_SIMULACIONES


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

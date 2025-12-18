"""
Servicio de datos: descarga de Yahoo Finance y resolución de tickers.
Incluye modo mock para desarrollo cuando Yahoo Finance no está disponible.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from fuzzywuzzy import process
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from app.config import (
    FECHA_INICIO_DEFAULT,
    TAMANO_VENTANA,
    DIAS_TRADING_ANUALES,
    MIN_TICKERS_REQUERIDOS,
    USE_MOCK_DATA
)


# Datos mock realistas para desarrollo
MOCK_STOCK_DATA = {
    "AAPL": {"precio_base": 175.0, "volatilidad": 0.02, "drift": 0.0003},
    "MSFT": {"precio_base": 380.0, "volatilidad": 0.018, "drift": 0.0004},
    "TSLA": {"precio_base": 250.0, "volatilidad": 0.035, "drift": 0.0002},
    "GOOGL": {"precio_base": 140.0, "volatilidad": 0.022, "drift": 0.0003},
    "AMZN": {"precio_base": 180.0, "volatilidad": 0.025, "drift": 0.0003},
    "NVDA": {"precio_base": 480.0, "volatilidad": 0.03, "drift": 0.0005},
    "META": {"precio_base": 500.0, "volatilidad": 0.028, "drift": 0.0004},
}


class DataService:
    """Servicio para descarga y procesamiento de datos financieros."""

    def __init__(self):
        self._sp500_cache: Optional[Dict[str, str]] = None
        self._nombres_sp500: List[str] = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def obtener_sp500(self) -> Dict[str, str]:
        """
        Obtiene el diccionario nombre → ticker del S&P 500.
        Usa caché para evitar múltiples requests.
        """
        if self._sp500_cache is not None:
            return self._sp500_cache

        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            tables = pd.read_html(response.text)
            df = tables[0]

            security_col = 'Security' if 'Security' in df.columns else 'Company'
            self._sp500_cache = dict(zip(df[security_col], df["Symbol"]))
            self._nombres_sp500 = list(self._sp500_cache.keys())

        except Exception as e:
            print(f"Advertencia: No se pudo cargar S&P 500: {e}")
            # Fallback con empresas comunes
            self._sp500_cache = {
                "Apple Inc.": "AAPL",
                "Microsoft Corporation": "MSFT",
                "Tesla, Inc.": "TSLA",
                "Amazon.com, Inc.": "AMZN",
                "Alphabet Inc.": "GOOGL",
                "NVIDIA Corporation": "NVDA",
                "Meta Platforms, Inc.": "META"
            }
            self._nombres_sp500 = list(self._sp500_cache.keys())

        return self._sp500_cache

    def buscar_ticker(self, nombre: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Busca tickers por nombre usando fuzzy matching.
        """
        sp500 = self.obtener_sp500()

        if not self._nombres_sp500:
            return []

        matches = process.extract(nombre.lower(), self._nombres_sp500, limit=limit)

        return [
            {"nombre": match[0], "ticker": sp500.get(match[0], "")}
            for match in matches
            if match[1] > 50
        ]

    def buscar_tickers(self, entrada_usuario: str) -> List[str]:
        """
        Convierte nombres comunes en tickers oficiales.
        Recibe una cadena con nombres separados por coma.
        """
        sp500 = self.obtener_sp500()
        nombres = [n.strip() for n in entrada_usuario.split(",") if n.strip()]
        tickers_finales = []
        
        for nombre in nombres:
            if nombre.upper() in sp500.values():
                tickers_finales.append(nombre.upper())
            else:
                # Búsqueda para corregir errores ortográficos
                if self._nombres_sp500:
                    match, score = process.extractOne(nombre, self._nombres_sp500)
                    tickers_finales.append(sp500.get(match, nombre.upper()))
                else:
                    tickers_finales.append(nombre.upper())
        
        return list(set(tickers_finales))

    def resolver_ticker(self, entrada: str) -> str:
        """
        Resuelve una entrada (nombre o ticker) a un ticker válido.
        """
        sp500 = self.obtener_sp500()
        entrada_upper = entrada.upper()

        if entrada_upper in sp500.values():
            return entrada_upper

        if self._nombres_sp500:
            match, score = process.extractOne(entrada.lower(), self._nombres_sp500)
            if score > 60:
                return sp500.get(match, entrada_upper)

        return entrada_upper

    def _generar_datos_mock(
        self,
        lista_tickers: List[str],
        n_dias: int = 1500
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Genera datos simulados realistas usando Geometric Brownian Motion.

        Args:
            lista_tickers: Lista de tickers
            n_dias: Número de días de datos a generar

        Returns:
            Tuple de (df_precios, df_rendimientos)
        """
        print(f"[MOCK] Generando datos simulados para: {lista_tickers}")

        # Generar fechas (días hábiles)
        fecha_fin = datetime.now()
        fecha_inicio = fecha_fin - timedelta(days=int(n_dias * 1.4))  # Factor para días no hábiles
        fechas = pd.bdate_range(start=fecha_inicio, end=fecha_fin)[-n_dias:]

        precios_dict = {}

        for ticker in lista_tickers:
            # Obtener parámetros del ticker o usar defaults
            params = MOCK_STOCK_DATA.get(ticker, {
                "precio_base": 100.0,
                "volatilidad": 0.025,
                "drift": 0.0003
            })

            precio_base = params["precio_base"]
            volatilidad = params["volatilidad"]
            drift = params["drift"]

            # Generar precios usando GBM
            np.random.seed(hash(ticker) % (2**32))  # Seed consistente por ticker
            rendimientos = np.random.normal(drift, volatilidad, n_dias)
            precios = precio_base * np.exp(np.cumsum(rendimientos))

            precios_dict[ticker] = precios

        df_precios = pd.DataFrame(precios_dict, index=fechas)
        df_rendimientos = np.log(df_precios / df_precios.shift(1)).dropna()

        return df_precios, df_rendimientos

    def descargar_datos(
        self,
        lista_tickers: List[str],
        fecha_inicio: str = FECHA_INICIO_DEFAULT
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Descarga datos históricos y calcula rendimientos logarítmicos.
        Si USE_MOCK_DATA está activo o la descarga falla, usa datos simulados.
        """
        # Si el modo mock está activo, usar datos simulados directamente
        if USE_MOCK_DATA:
            print("[MOCK] Modo desarrollo activo - usando datos simulados")
            return self._generar_datos_mock(lista_tickers)

        try:
            df_precios = yf.download(
                tickers=lista_tickers,
                start=fecha_inicio,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if isinstance(df_precios.columns, pd.MultiIndex):
                df_precios = df_precios["Close"]
            elif "Close" in df_precios.columns:
                df_precios = df_precios[["Close"]]
                df_precios.columns = lista_tickers[:1]

            df_precios = df_precios.dropna()

            if df_precios.empty or len(df_precios.columns) < MIN_TICKERS_REQUERIDOS:
                print("Yahoo Finance falló - usando datos mock como fallback")
                return self._generar_datos_mock(lista_tickers)

            df_rendimientos = np.log(df_precios / df_precios.shift(1)).dropna()

            if len(df_rendimientos) < TAMANO_VENTANA + 1:
                print("Datos insuficientes - usando datos mock como fallback")
                return self._generar_datos_mock(lista_tickers)

            return df_precios, df_rendimientos

        except Exception as e:
            print(f"Error descargando datos: {e} - usando datos mock como fallback")
            return self._generar_datos_mock(lista_tickers)

    def preparar_secuencias(self, df_rend: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara ventanas X y objetivos Y para entrenamiento del LSTM.
        Utiliza el scaler del servicio para normalizar datos.

        Args:
            df_rend: DataFrame con rendimientos logarítmicos

        Returns:
            Tuple de (X, Y) arrays para entrenamiento
        """
        scaled_data = self.scaler.fit_transform(df_rend.values)

        X, Y = [], []
        for i in range(TAMANO_VENTANA, len(scaled_data)):
            X.append(scaled_data[i - TAMANO_VENTANA:i])
            Y.append(scaled_data[i])

        return np.array(X), np.array(Y)


# Instancia singleton del servicio
data_service = DataService()

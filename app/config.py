"""
Configuración centralizada del backend.
Todas las constantes del modelo y la API en un solo lugar.
Lee variables de entorno para producción.
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (si existe)
load_dotenv()


def get_bool_env(key: str, default: bool) -> bool:
    """Convierte variable de entorno a booleano."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_list_env(key: str, default: str) -> list:
    """Convierte variable de entorno a lista (separada por comas)."""
    value = os.getenv(key, default)
    if value == "*":
        return ["*"]
    return [v.strip() for v in value.split(",") if v.strip()]


# === ENTORNO ===
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# === CONFIGURACIÓN DEL MODELO ===
DIAS_TRADING_ANUALES = 252
TAMANO_VENTANA = 60  # Días de retrospectiva para LSTM
PROPORCION_ENTRENAMIENTO = 0.8

# === CONFIGURACIÓN DE LA RED NEURONAL ===
EPOCAS_NN = 50
EPOCAS_INCREMENTAL = 5
TAMANO_LOTE_NN = 32
UNIDADES_LSTM = 100
NOMBRE_ARCHIVO_MODELO = "modelo_lstm_v2.h5"
SEED = 42

# === CONFIGURACIÓN MONTE CARLO ===
N_SIMULACIONES = int(os.getenv("N_SIMULACIONES", "10000"))
N_RUTAS_MC = int(os.getenv("N_RUTAS_MC", "500"))
DIAS_PROYECCION = int(os.getenv("DIAS_PROYECCION", "30"))

# === CONFIGURACIÓN DE DATOS ===
FECHA_INICIO_DEFAULT = "2018-01-01"
MIN_TICKERS_REQUERIDOS = 2

# === MODO DESARROLLO ===
# En desarrollo: True (datos simulados)
# En producción: False (Yahoo Finance real)
USE_MOCK_DATA = get_bool_env("USE_MOCK_DATA", not IS_PRODUCTION)

# === CONFIGURACIÓN DE LA API ===
API_TITLE = "Portfolio Optimizer API"
API_DESCRIPTION = """
API para optimización de carteras de inversión usando LSTM y Monte Carlo.
Proyecto educativo - Samsung Innovation Campus 2025.

**Nota:** Esta herramienta es educativa, no constituye asesoría financiera.
"""
API_VERSION = "1.0.0"

# === CORS ===
ALLOWED_ORIGINS = get_list_env("ALLOWED_ORIGINS", "*")

# === SERVIDOR ===
PORT = int(os.getenv("PORT", "8000"))

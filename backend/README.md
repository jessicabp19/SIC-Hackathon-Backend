# Portfolio Optimizer API

API REST para optimización de carteras de inversión usando LSTM y Monte Carlo.

**Samsung Innovation Campus 2025 - Hackathon**

## Tecnologías

- **FastAPI** - Framework web
- **TensorFlow/Keras** - Red neuronal LSTM
- **NumPy/Pandas** - Procesamiento de datos
- **yfinance** - Datos de Yahoo Finance

## Instalación Local

```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Info de la API |
| GET | `/docs` | Documentación Swagger |
| POST | `/api/portfolio/analyze` | Optimizar portafolio |
| GET | `/api/portfolio/search` | Buscar empresas |
| POST | `/api/chatbot/message` | Chatbot educativo |
| GET | `/api/chatbot/suggestions` | Preguntas sugeridas |

## Variables de Entorno

Ver `.env.example` para la lista completa.

| Variable | Descripción | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | development / production | development |
| `USE_MOCK_DATA` | Usar datos simulados | true (dev) |
| `ALLOWED_ORIGINS` | Dominios CORS permitidos | * |
| `PORT` | Puerto del servidor | 8000 |

## Deploy en Railway

### Opción 1: Desde GitHub (Recomendado)

1. Ir a [railway.app](https://railway.app)
2. Click en "New Project"
3. Seleccionar "Deploy from GitHub repo"
4. Elegir este repositorio
5. Railway detectará automáticamente el `Procfile`
6. Configurar variables de entorno:
   - `ENVIRONMENT=production`
   - `USE_MOCK_DATA=false`
   - `ALLOWED_ORIGINS=https://tu-frontend.vercel.app`

### Opción 2: Railway CLI

```bash
# Instalar CLI
npm install -g @railway/cli

# Login
railway login

# Crear proyecto
railway init

# Deploy
railway up
```

## Estructura del Proyecto

```
backend/
├── app/
│   ├── main.py              # Entry point FastAPI
│   ├── config.py            # Configuración
│   ├── api/routes/
│   │   ├── portfolio.py     # Endpoints portafolio
│   │   └── chatbot.py       # Endpoints chatbot
│   ├── services/
│   │   ├── data_service.py  # Yahoo Finance
│   │   ├── lstm_service.py  # Modelo LSTM
│   │   ├── montecarlo_service.py
│   │   ├── optimizer_service.py
│   │   └── chatbot_service.py
│   └── models/
│       └── schemas.py       # Pydantic schemas
├── requirements.txt
├── Procfile
├── railway.json
└── .env.example
```

## Ejemplo de Uso

### Analizar Portafolio

```bash
curl -X POST "https://tu-api.railway.app/api/portfolio/analyze" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "TSLA"]}'
```

### Chatbot

```bash
curl -X POST "https://tu-api.railway.app/api/chatbot/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué es el riesgo?"}'
```

## Notas

- Esta herramienta es **educativa**, no constituye asesoría financiera
- El modelo LSTM se entrena en la primera ejecución (~1-2 min)
- En producción, usa datos reales de Yahoo Finance

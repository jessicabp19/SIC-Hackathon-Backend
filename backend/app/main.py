"""
Portfolio Optimizer API - Samsung Innovation Campus 2025
Punto de entrada de la aplicación FastAPI.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, ALLOWED_ORIGINS
from app.api.routes import portfolio, chatbot

# Crear aplicación FastAPI
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(portfolio.router, prefix="/api")
app.include_router(chatbot.router, prefix="/api")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "nombre": "Portfolio Optimizer API",
        "version": API_VERSION,
        "descripcion": "API para optimización de carteras de inversión",
        "docs": "/docs",
        "endpoints": {
            "portfolio": {
                "analizar": "POST /api/portfolio/analyze",
                "buscar": "GET /api/portfolio/search?query=apple"
            },
            "chatbot": {
                "mensaje": "POST /api/chatbot/message",
                "sugerencias": "GET /api/chatbot/suggestions"
            }
        },
        "nota": "Esta herramienta es educativa, no constituye asesoría financiera."
    }


@app.get("/health")
async def health():
    """Health check general de la API."""
    return {"status": "ok", "api": "portfolio-optimizer"}

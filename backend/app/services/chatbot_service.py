"""
Servicio de Chatbot Educativo sobre Inversión.
Implementación rule-based con fuzzy matching, escalable a NLP.
"""
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Tuple, Optional


# Base de conocimiento: preguntas frecuentes sobre inversión
KNOWLEDGE_BASE: List[Dict] = [
    # === CONCEPTOS BÁSICOS ===
    {
        "categoria": "conceptos_basicos",
        "keywords": ["qué es invertir", "que es invertir", "invertir significa", "definicion invertir"],
        "pregunta": "¿Qué es invertir?",
        "respuesta": "Invertir es destinar dinero a un activo (acciones, bonos, fondos) con la expectativa de obtener un rendimiento futuro. A diferencia del ahorro tradicional, la inversión busca que tu dinero crezca por encima de la inflación."
    },
    {
        "categoria": "conceptos_basicos",
        "keywords": ["qué es una acción", "que es una accion", "acciones", "comprar acciones"],
        "pregunta": "¿Qué es una acción?",
        "respuesta": "Una acción representa una pequeña parte de propiedad de una empresa. Al comprar acciones de Apple, por ejemplo, te conviertes en dueño de una fracción de la compañía y puedes beneficiarte si su valor aumenta o si paga dividendos."
    },
    {
        "categoria": "conceptos_basicos",
        "keywords": ["qué es un portafolio", "que es un portafolio", "cartera", "cartera de inversión"],
        "pregunta": "¿Qué es un portafolio de inversión?",
        "respuesta": "Un portafolio o cartera es el conjunto de todas tus inversiones. Puede incluir acciones, bonos, fondos y otros activos. La idea es diversificar para reducir el riesgo: 'no poner todos los huevos en una sola canasta'."
    },

    # === RIESGO ===
    {
        "categoria": "riesgo",
        "keywords": ["qué es el riesgo", "que es el riesgo", "riesgo inversión", "riesgo significa"],
        "pregunta": "¿Qué es el riesgo en inversión?",
        "respuesta": "El riesgo es la posibilidad de que tu inversión pierda valor o no genere los rendimientos esperados. A mayor riesgo, mayor potencial de ganancia (pero también de pérdida). Por eso es importante entender tu tolerancia al riesgo antes de invertir."
    },
    {
        "categoria": "riesgo",
        "keywords": ["volatilidad", "qué es volatilidad", "que es volatilidad", "mercado volátil"],
        "pregunta": "¿Qué es la volatilidad?",
        "respuesta": "La volatilidad mide cuánto varía el precio de un activo en el tiempo. Una acción muy volátil puede subir 10% un día y bajar 8% al siguiente. Las criptomonedas son muy volátiles; los bonos gubernamentales son poco volátiles."
    },
    {
        "categoria": "riesgo",
        "keywords": ["diversificar", "diversificación", "como diversificar", "no poner huevos"],
        "pregunta": "¿Por qué debo diversificar?",
        "respuesta": "Diversificar significa invertir en diferentes activos para reducir el riesgo. Si una empresa quiebra pero tienes inversiones en otras 10, el impacto es menor. La regla de oro: nunca inviertas todo tu dinero en un solo activo."
    },

    # === MÉTRICAS ===
    {
        "categoria": "metricas",
        "keywords": ["sharpe", "ratio sharpe", "qué es sharpe", "que es sharpe"],
        "pregunta": "¿Qué es el Ratio de Sharpe?",
        "respuesta": "El Ratio de Sharpe mide el rendimiento ajustado por riesgo. Un Sharpe de 1.0 o más se considera bueno. Significa que estás obteniendo buen rendimiento por cada unidad de riesgo que asumes. Nuestra herramienta optimiza tu portafolio buscando el mejor Sharpe."
    },
    {
        "categoria": "metricas",
        "keywords": ["rendimiento", "rentabilidad", "ganancia", "cuanto gano"],
        "pregunta": "¿Qué es el rendimiento?",
        "respuesta": "El rendimiento es la ganancia (o pérdida) de tu inversión, expresada como porcentaje. Si inviertes $1000 y después tienes $1100, tu rendimiento es 10%. El rendimiento histórico no garantiza rendimientos futuros."
    },
    {
        "categoria": "metricas",
        "keywords": ["monte carlo", "simulación", "qué es monte carlo", "que es monte carlo"],
        "pregunta": "¿Qué es la simulación Monte Carlo?",
        "respuesta": "Monte Carlo es una técnica que simula miles de escenarios posibles para predecir resultados. En nuestra herramienta, simulamos 500 posibles trayectorias de precios para entender mejor el riesgo futuro de tu portafolio."
    },

    # === ESTRATEGIAS ===
    {
        "categoria": "estrategias",
        "keywords": ["largo plazo", "corto plazo", "horizonte", "cuanto tiempo"],
        "pregunta": "¿Debo invertir a largo o corto plazo?",
        "respuesta": "Para principiantes, se recomienda invertir a largo plazo (5+ años). El mercado puede ser volátil en el corto plazo, pero históricamente tiende a crecer a largo plazo. La paciencia es clave en la inversión."
    },
    {
        "categoria": "estrategias",
        "keywords": ["cuanto invertir", "cuánto invertir", "dinero invertir", "empezar invertir"],
        "pregunta": "¿Cuánto dinero necesito para empezar a invertir?",
        "respuesta": "Puedes empezar con cantidades pequeñas, incluso $100 o menos. Lo importante es: 1) Tener un fondo de emergencia primero (3-6 meses de gastos). 2) No invertir dinero que necesites en el corto plazo. 3) Solo invertir lo que puedas permitirte perder."
    },
    {
        "categoria": "estrategias",
        "keywords": ["buy and hold", "comprar y mantener", "estrategia pasiva"],
        "pregunta": "¿Qué es la estrategia Buy and Hold?",
        "respuesta": "Buy and Hold significa comprar activos y mantenerlos por largo tiempo, ignorando las fluctuaciones del mercado. Es una estrategia simple y efectiva para principiantes, ya que evita el estrés de intentar 'timing' del mercado."
    },

    # === HERRAMIENTA ===
    {
        "categoria": "herramienta",
        "keywords": ["cómo funciona", "como funciona", "herramienta", "optimizador"],
        "pregunta": "¿Cómo funciona esta herramienta?",
        "respuesta": "Nuestra herramienta: 1) Descarga datos históricos de las empresas que elijas. 2) Usa inteligencia artificial (LSTM) para proyectar rendimientos futuros. 3) Simula 500 escenarios con Monte Carlo. 4) Calcula la distribución óptima de tu dinero maximizando el Ratio de Sharpe."
    },
    {
        "categoria": "herramienta",
        "keywords": ["lstm", "red neuronal", "inteligencia artificial", "ia", "modelo"],
        "pregunta": "¿Qué es el modelo LSTM que usan?",
        "respuesta": "LSTM (Long Short-Term Memory) es un tipo de red neuronal especializada en aprender patrones en series de tiempo. Analiza los rendimientos históricos para identificar tendencias y proyectar comportamientos futuros del mercado."
    },
    {
        "categoria": "herramienta",
        "keywords": ["pesos", "distribución", "porcentaje", "asignar"],
        "pregunta": "¿Qué significan los pesos del portafolio?",
        "respuesta": "Los pesos indican qué porcentaje de tu dinero deberías asignar a cada activo. Por ejemplo, si AAPL tiene peso 40%, significa que de cada $100 que inviertas, $40 deberían ir a Apple. Los pesos siempre suman 100%."
    },

    # === ADVERTENCIAS ===
    {
        "categoria": "advertencias",
        "keywords": ["es seguro", "garantizado", "voy a ganar", "perder dinero"],
        "pregunta": "¿Es seguro invertir? ¿Puedo perder dinero?",
        "respuesta": "⚠️ IMPORTANTE: Toda inversión conlleva riesgo. Puedes perder parte o todo tu dinero. Esta herramienta es EDUCATIVA y no garantiza ganancias. Nunca inviertas dinero que no puedas permitirte perder. Considera consultar un asesor financiero profesional."
    },
    {
        "categoria": "advertencias",
        "keywords": ["asesor", "consejo financiero", "recomendación", "debo invertir"],
        "pregunta": "¿Esto es asesoría financiera?",
        "respuesta": "⚠️ NO. Esta herramienta es puramente educativa. No somos asesores financieros certificados y no damos recomendaciones de inversión. Los resultados son simulaciones basadas en datos históricos y NO garantizan rendimientos futuros."
    },

    # === S&P 500 ===
    {
        "categoria": "conceptos_basicos",
        "keywords": ["s&p 500", "sp500", "sp 500", "indice", "índice"],
        "pregunta": "¿Qué es el S&P 500?",
        "respuesta": "El S&P 500 es un índice que agrupa las 500 empresas más grandes de Estados Unidos (Apple, Microsoft, Amazon, etc.). Es el indicador más usado para medir la salud del mercado. Nuestra herramienta te permite elegir empresas de este índice."
    },
    {
        "categoria": "conceptos_basicos",
        "keywords": ["ticker", "símbolo", "simbolo", "aapl", "msft"],
        "pregunta": "¿Qué es un ticker?",
        "respuesta": "El ticker es el código abreviado de una empresa en la bolsa. Por ejemplo: AAPL = Apple, MSFT = Microsoft, TSLA = Tesla. En nuestra herramienta puedes buscar por nombre de empresa y te mostramos el ticker correspondiente."
    },
]

# Respuestas por defecto
DEFAULT_RESPONSES = [
    "No tengo información específica sobre eso, pero puedo ayudarte con conceptos de inversión, riesgo, estrategias o cómo usar esta herramienta. ¿Qué te gustaría saber?",
    "Esa pregunta está fuera de mi conocimiento actual. Puedo explicarte sobre: portafolios, riesgo, diversificación, el Ratio de Sharpe, o cómo funciona nuestra herramienta.",
    "No estoy seguro de entender tu pregunta. Intenta preguntar sobre: qué es invertir, qué es el riesgo, cómo diversificar, o cómo funciona el optimizador.",
]

# Saludos
GREETINGS = {
    "keywords": ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "hi", "hello"],
    "respuesta": "¡Hola! Soy el asistente educativo de inversión. Puedo ayudarte a entender conceptos como riesgo, diversificación, el Ratio de Sharpe, y cómo usar esta herramienta de optimización de portafolios. ¿Qué te gustaría aprender?"
}


class ChatbotService:
    """Chatbot educativo rule-based con fuzzy matching."""

    def __init__(self):
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """Construye un índice de todas las keywords para búsqueda rápida."""
        self._all_keywords: List[str] = []
        self._keyword_to_entry: Dict[str, Dict] = {}

        for entry in KNOWLEDGE_BASE:
            for keyword in entry["keywords"]:
                self._all_keywords.append(keyword)
                self._keyword_to_entry[keyword] = entry

    def _check_greeting(self, mensaje: str) -> Optional[str]:
        """Verifica si el mensaje es un saludo."""
        mensaje_lower = mensaje.lower().strip()
        for greeting in GREETINGS["keywords"]:
            if greeting in mensaje_lower:
                return GREETINGS["respuesta"]
        return None

    def _find_best_match(self, mensaje: str) -> Tuple[Optional[Dict], int]:
        """
        Encuentra la mejor coincidencia en la base de conocimiento.

        Returns:
            Tuple de (entrada_encontrada, score)
        """
        mensaje_lower = mensaje.lower()

        # Buscar coincidencia exacta primero
        for entry in KNOWLEDGE_BASE:
            for keyword in entry["keywords"]:
                if keyword in mensaje_lower:
                    return entry, 100

        # Fuzzy matching si no hay coincidencia exacta
        if self._all_keywords:
            best_match, score = process.extractOne(
                mensaje_lower,
                self._all_keywords,
                scorer=fuzz.token_set_ratio
            )
            if score >= 60:
                return self._keyword_to_entry.get(best_match), score

        return None, 0

    def responder(self, mensaje: str) -> Dict[str, str]:
        """
        Genera una respuesta al mensaje del usuario.

        Args:
            mensaje: Pregunta o mensaje del usuario

        Returns:
            Dict con 'response' y 'categoria'
        """
        if not mensaje or len(mensaje.strip()) < 2:
            return {
                "response": "Por favor, escribe una pregunta más específica.",
                "categoria": "error"
            }

        # Verificar si es un saludo
        greeting_response = self._check_greeting(mensaje)
        if greeting_response:
            return {
                "response": greeting_response,
                "categoria": "saludo"
            }

        # Buscar en la base de conocimiento
        entry, score = self._find_best_match(mensaje)

        if entry and score >= 60:
            return {
                "response": entry["respuesta"],
                "categoria": entry["categoria"]
            }

        # Respuesta por defecto
        import random
        return {
            "response": random.choice(DEFAULT_RESPONSES),
            "categoria": "no_encontrado"
        }

    def obtener_categorias(self) -> List[str]:
        """Retorna las categorías disponibles."""
        categorias = set()
        for entry in KNOWLEDGE_BASE:
            categorias.add(entry["categoria"])
        return list(categorias)

    def obtener_preguntas_sugeridas(self) -> List[str]:
        """Retorna algunas preguntas sugeridas para el usuario."""
        return [
            "¿Qué es invertir?",
            "¿Qué es el riesgo?",
            "¿Cómo funciona esta herramienta?",
            "¿Qué es el Ratio de Sharpe?",
            "¿Por qué debo diversificar?",
            "¿Cuánto dinero necesito para empezar?"
        ]


# Instancia singleton del servicio
chatbot_service = ChatbotService()

"""Agente de inventario de Carla: un LLM que usa la API de inventario como tools.

El bucle del agente está escrito a mano, sin frameworks (nada de LangChain,
LlamaIndex ni AutoGen).

Arranque (con la API ya en marcha en otro terminal):
    uv run python agent.py
"""

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("INVENTORY_API_URL", "http://127.0.0.1:8000")


# =====================================================================
# 1. TOOLS: la "ficha" de cada endpoint que se le enseña al LLM.
#
# El modelo nunca ve el código de la API: solo ve estas fichas. Lo que
# pongas en "description" es literalmente todo lo que sabe para decidir
# qué herramienta usar y con qué argumentos. Formato: JSON Schema, el
# estándar de OpenAI que Groq también entiende.
# =====================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": (
                "Devuelve todos los productos del inventario con su id, nombre, cantidad "
                "y unidad. Úsala cuando pregunten por el stock de algo y SIEMPRE antes de "
                "update_stock para averiguar el id del producto a partir de su nombre."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_product",
            "description": (
                "Da de alta un producto que todavía NO existe en el inventario. Si el "
                "producto ya existe (aunque cambien mayúsculas o tildes), no la uses: usa "
                "update_stock para sumarle o restarle cantidad."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nombre del producto, p. ej. 'Leche de avena'.",
                    },
                    "quantity": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Cantidad inicial en stock (0 o más).",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unidad de medida: unidades, kg, litros, bolsas (1 kg), cajas...",
                    },
                },
                "required": ["name", "quantity", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": (
                "Suma o resta cantidad al stock de un producto que ya existe. Usa un delta "
                "POSITIVO para entradas (llega una entrega, devolución de un cliente) y "
                "NEGATIVO para salidas (ventas, roturas, mermas). Necesita el id del "
                "producto: si no lo conoces, llama antes a list_inventory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Id del producto, tal y como aparece en list_inventory.",
                    },
                    "delta": {
                        "type": "number",
                        "description": "Cantidad a sumar (positiva) o restar (negativa). Nunca 0.",
                    },
                },
                "required": ["product_id", "delta"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_stock_alerts",
            "description": (
                "Devuelve los productos cuya cantidad está por debajo de un umbral, de más "
                "a menos urgente. Úsala cuando pregunten qué se está agotando, qué hay que "
                "pedir o qué anda bajo de stock."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Umbral de aviso. Si el usuario no indica otro, usa 10.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


# =====================================================================
# 2. ACTUAR: traducir la tool que eligió el LLM a una llamada HTTP real.
# =====================================================================

def call_tool(name: str, args: dict) -> str:
    """Ejecuta una tool llamando al endpoint de la API y devuelve el resultado como texto JSON.

    Nunca lanza excepciones: si algo falla, devuelve un JSON con "error" para
    que el LLM lo lea y pueda corregirse (pedir el id correcto, avisar a Carla...).
    """
    try:
        with httpx.Client(base_url=API_URL, timeout=10) as http:
            if name == "list_inventory":
                response = http.get("/inventory")
            elif name == "add_product":
                response = http.post("/inventory", json=args)
            elif name == "update_stock":
                response = http.patch(
                    f"/inventory/{args['product_id']}", json={"delta": args["delta"]}
                )
            elif name == "get_low_stock_alerts":
                response = http.get(
                    "/inventory/alerts", params={"threshold": args.get("threshold", 10)}
                )
            else:
                return json.dumps({"error": f"La tool '{name}' no existe."}, ensure_ascii=False)

        if response.is_success:
            return json.dumps(response.json(), ensure_ascii=False)
        return json.dumps(
            {"error": response.json().get("detail"), "status_code": response.status_code},
            ensure_ascii=False,
        )
    except httpx.ConnectError:
        return json.dumps(
            {"error": f"No se puede conectar con la API en {API_URL}. ¿Está arrancada?"},
            ensure_ascii=False,
        )
    except KeyError as missing:
        return json.dumps(
            {"error": f"Falta el argumento obligatorio {missing} para la tool '{name}'."},
            ensure_ascii=False,
        )

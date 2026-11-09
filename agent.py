"""Agente de inventario de Carla: un LLM que usa la API de inventario como tools.

El bucle del agente está escrito a mano, sin frameworks (nada de LangChain,
LlamaIndex ni AutoGen).

Arranque (con la API ya en marcha en otro terminal):
    uv run python agent.py
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("INVENTORY_API_URL", "http://127.0.0.1:8000")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = 8  # vueltas máximas del bucle por mensaje: freno de seguridad
LOG_FILE = Path(__file__).resolve().parent / "conversation_log.csv"
LOG_FIELDS = ["actor", "message", "tool_call", "timestamp"]


def log_event(actor: str, message: str, tool_call: str = "") -> None:
    """Añade una fila a conversation_log.csv. Nunca borra ni reescribe las anteriores.

    actor: "user" (Carla), "agent" (el LLM: respuestas y peticiones de tools)
    o "tool" (el resultado que devolvió la API).
    """
    is_new = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    # Modo "a" (append): el sistema operativo escribe siempre al final del fichero.
    # Se abre y se cierra en cada evento, así cada fila queda guardada en disco al
    # momento y un Ctrl+C a mitad de sesión no pierde nada.
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "actor": actor,
                "message": message,
                "tool_call": tool_call,
                # ISO 8601 con zona horaria, p. ej. 2026-11-09T19:45:03+01:00
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )

SYSTEM_PROMPT = """Eres el asistente de inventario de la tienda de Carla, que vende \
suministros para cafeterías. Carla te habla en lenguaje natural para consultar el \
stock, registrar entregas y ventas, y saber qué se está agotando.

Reglas:
- Usa SIEMPRE las tools para consultar o cambiar el stock. Nunca inventes cantidades, \
productos ni ids.
- Entregas y llegadas suman stock (delta positivo). Ventas, roturas y mermas lo restan \
(delta negativo).
- Antes de cambiar stock, mira en list_inventory TODOS los productos que encajan con el \
nombre que dice Carla. Si encaja más de uno (p. ej. "leche" → Leche entera y Leche de \
avena), NO cambies nada y pregúntale cuál, aunque antes se haya hablado de uno de ellos.
- Si habla de un producto que no existe, pregúntale si quiere darlo de alta y con qué \
unidad antes de crearlo.
- Si da la cantidad en "unidades" o en otra palabra genérica, cuenta cada una como una \
unidad del producto (30 unidades de leche en litros → 30 litros), haz el cambio y \
menciona la equivalencia. Pregunta solo si la conversión necesita un dato que no tienes.
- Tras cada cambio, di cómo queda el stock y avisa si ha quedado por debajo de 10.
- Responde en español, en pocas frases y en tono cercano: es una conversación, no un \
informe. Sin tablas salvo que Carla pida un listado."""


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


# =====================================================================
# 3. EL BUCLE DEL AGENTE: Observar -> Pensar -> Actuar -> Actualizar -> Repetir
# =====================================================================

llm = openai.OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def run_turn(messages: list[dict]) -> str:
    """Procesa el último mensaje del usuario hasta tener una respuesta final.

    `messages` es el historial completo de la sesión: se modifica aquí mismo
    para que cada vuelta (y cada mensaje futuro) vea todo lo anterior.
    """
    for _ in range(MAX_STEPS):
        # PENSAR: el LLM ve todo el historial y las fichas de las tools.
        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,  # poca creatividad: queremos un agente predecible
        )
        reply = response.choices[0].message

        # Sin tools pendientes = respuesta final. Fin limpio del bucle.
        if not reply.tool_calls:
            answer = reply.content or ""
            messages.append({"role": "assistant", "content": answer})
            log_event("agent", answer)
            return answer

        # El LLM quiere usar tools: guardamos su petición en el historial...
        messages.append(
            {
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in reply.tool_calls
                ],
            }
        )

        for call in reply.tool_calls:
            log_event("agent", call.function.arguments or "{}", call.function.name)
            # ACTUAR: ejecutar la tool contra la API.
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if isinstance(args, dict):
                result = call_tool(call.function.name, args)
            else:
                result = json.dumps({"error": "Los argumentos no son un objeto JSON válido."})
            log_event("tool", result, call.function.name)
            print(f"   ⚙ {call.function.name}({call.function.arguments or ''}) → {result[:120]}")

            # ACTUALIZAR: inyectar el resultado en el historial. El tool_call_id
            # le dice al LLM a cuál de sus peticiones responde este resultado.
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # REPETIR: vuelta al principio, el LLM ya ve los resultados.

    answer = "Lo siento, no he conseguido terminar esta petición. ¿Puedes decírmela de otra forma?"
    messages.append({"role": "assistant", "content": answer})
    log_event("agent", answer)
    return answer


# =====================================================================
# 4. CLI: leer a Carla desde el terminal e imprimir las respuestas
# =====================================================================

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # tildes y ñ bien en Windows

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY. Copia .env.example como .env y pon tu clave.")
    try:
        httpx.get(f"{API_URL}/inventory", timeout=5).raise_for_status()
    except httpx.HTTPError:
        sys.exit(
            f"La API no responde en {API_URL}. Arráncala primero en otro terminal:\n"
            "    uv run uvicorn api.app:app --reload"
        )

    print("Asistente de inventario. Escribe tu mensaje (o 'salir' para terminar).")
    # El historial vive en memoria toda la sesión: así el LLM recuerda lo hablado.
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nTú > ").strip()  # OBSERVAR
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        log_event("user", user_input)
        try:
            answer = run_turn(messages)
        except openai.APIError as error:
            print(f"\n[Error al hablar con el LLM: {error}. Inténtalo de nuevo.]")
            log_event("agent", f"[error del LLM] {error}")
            continue
        except KeyboardInterrupt:
            break
        print(f"\nAgente > {answer}")

    print("\n¡Hasta luego!")


if __name__ == "__main__":
    main()

# Agente de inventario con IA

Asistente conversacional para la tienda de Carla, que vende suministros para cafeterías.
Carla le habla en lenguaje natural ("acaban de llegar 30 unidades de leche de avena",
"vendimos 12 bolsas de arábica", "¿qué se está agotando?") y el agente consulta y actualiza
el stock por ella.

Tiene dos piezas que funcionan juntas:

| Pieza | Fichero | Qué hace |
|---|---|---|
| API REST | [api/app.py](api/app.py) | FastAPI que gestiona el inventario y lo guarda en [products.csv](products.csv) |
| Agente | [agent.py](agent.py) | Bucle Observar → Pensar → Actuar → Actualizar → Repetir sobre un LLM de Groq que usa la API como tools. Registra cada evento en [conversation_log.csv](conversation_log.csv) |

El bucle está escrito a mano en Python. No usa LangChain, LlamaIndex, AutoGen ni ningún
otro framework de agentes.

---

## Puesta en marcha

### 1. Instalar

Hace falta Python 3.12 o superior y [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```bash
uv sync                     # crea .venv e instala las versiones exactas de uv.lock
cp .env.example .env        # y pon tu clave de https://console.groq.com/keys
```

`.env` contiene la clave y está en `.gitignore`: nunca se sube al repositorio.

### 2. Arrancar los dos procesos

Necesitas **dos terminales abiertos a la vez**, ambos en la raíz del proyecto. La API tiene
que estar en marcha **antes** que el agente; si no, el agente avisa y se cierra.

```bash
# Terminal 1: la API (se queda escuchando; no escribas nada más en este terminal)
uv run uvicorn api.app:app --reload

# Terminal 2: el agente
uv run python agent.py
```

> Si prefieres los comandos del enunciado (`uvicorn api.app:app --reload` y
> `python agent.py`), activa antes el entorno virtual en cada terminal:
> `.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en macOS y Linux.

Con la API en marcha, la documentación interactiva está en http://127.0.0.1:8000/docs.

### 3. Detener y relanzar

- **Detener:** escribe `salir` en el agente (o pulsa `Ctrl + C`) y después pulsa `Ctrl + C`
  en el terminal de la API.
- **Relanzar el agente:** vuelve a ejecutar `uv run python agent.py`; no hace falta
  reiniciar la API. Empieza una conversación nueva, pero `products.csv` y
  `conversation_log.csv` conservan todo lo anterior.

---

## Ejemplo de conversación

Este es un extracto real de la sesión de demostración que quedó registrada en
[conversation_log.csv](conversation_log.csv). Las líneas `⚙` muestran cada tool que el
agente decidió usar.

```
Tú > Acaban de llegar 30 unidades de leche de avena
   ⚙ list_inventory({}) → [...]
   ⚙ update_stock({"delta":30,"product_id":4}) → {"name": "Leche de avena", "quantity": 36.0, ...}

Agente > ¡Listo! Hemos sumado 30 litros a la leche de avena. Ahora tienes 36 litros en stock.

Tú > Necesito sacar 2 litros de leche

Agente > Tengo dos tipos de leche en el inventario (leche entera y leche de avena).
         ¿De cuál quieres sacar los 2 litros?

Tú > La entera
   ⚙ update_stock({"delta":-2,"product_id":3}) → {"name": "Leche entera", "quantity": 46.0, ...}

Agente > He restado 2 litros de leche entera. Ahora tienes 46 litros en stock.

Tú > Añade sirope de caramelo, 6 botellas, y dime si ya sale en las alertas
   ⚙ add_product({"name":"Sirope de caramelo","quantity":6,"unit":"botellas (1 l)"}) → {..., "id": 11}
   ⚙ get_low_stock_alerts({"threshold":10}) → [...]

Agente > He añadido 6 botellas de sirope de caramelo. En las alertas (menos de 10) aparecen
         ahora: cacao en polvo (3.5 kg), sirope de caramelo (6 botellas) y café robusta (8 bolsas).
```

---

## API

| Método y ruta | Qué hace | Respuestas |
|---|---|---|
| `GET /inventory` | Lista todos los productos | `200` |
| `POST /inventory` | Da de alta un producto (`name`, `quantity`, `unit`) | `201` creado · `409` ya existe ese nombre · `422` datos no válidos |
| `PATCH /inventory/{product_id}` | Suma o resta stock con `delta` (positivo para entradas, negativo para salidas) | `200` · `404` no existe ese id · `409` stock insuficiente · `422` delta 0 o no válido |
| `GET /inventory/alerts?threshold=10` | Productos con cantidad **menor** que el umbral (10 por defecto), los más urgentes primero | `200` · `422` umbral negativo |

Los mensajes de error explican qué ha pasado y qué hacer a continuación, por ejemplo:
*"No existe ningún producto con id 99. Consulta GET /inventory para ver los ids
disponibles."* Están pensados para que el LLM los lea y pueda corregirse solo.

**Persistencia:** cada petición lee `products.csv` y cada cambio lo reescribe entero. La
escritura es atómica (primero a `products.tmp` y luego se renombra), así que un corte a
mitad de escritura no corrompe el fichero. Un `threading.Lock` evita que dos cambios
simultáneos se pisen.

---

## Agente

### Tools

Cada tool tiene `name`, `description` y `parameters` en JSON Schema. Las descripciones
le dicen al LLM no solo qué hace cada tool, sino también **cuándo** usarla.

| Tool | Endpoint | Parámetros |
|---|---|---|
| `list_inventory` | `GET /inventory` | ninguno |
| `add_product` | `POST /inventory` | `name: string`, `quantity: number ≥ 0`, `unit: string` |
| `update_stock` | `PATCH /inventory/{product_id}` | `product_id: integer`, `delta: number` |
| `get_low_stock_alerts` | `GET /inventory/alerts` | `threshold?: number ≥ 0` |

### El bucle (`run_turn` en [agent.py](agent.py))

1. **Observar:** el mensaje de Carla entra en el historial `messages`.
2. **Pensar:** se envía al LLM el historial completo junto con las definiciones de las tools.
3. **Actuar:** si el LLM pide tools, `call_tool()` llama al endpoint correspondiente de la API.
4. **Actualizar:** cada resultado se añade al historial como mensaje `tool`, con el
   `tool_call_id` de la petición a la que responde.
5. **Repetir:** se vuelve al paso 2. El bucle termina cuando el LLM responde sin pedir
   tools, que es la respuesta final. `MAX_STEPS = 8` evita que pueda girar sin fin.

El historial vive en memoria durante toda la sesión, así que el agente recuerda lo hablado
(por ejemplo, "¿cuánta avena teníamos antes de la entrega?"). Si una tool falla, `call_tool`
no lanza una excepción: devuelve `{"error": ...}` para que el LLM lo lea y reaccione.

### Reglas del prompt de sistema

Salieron de probar el agente con frases reales:

- Nunca inventar cantidades ni ids: siempre se consultan con las tools.
- Si un nombre encaja con varios productos ("leche"), **preguntar antes de cambiar nada**.
  En las primeras pruebas el agente restaba de la leche de avena porque era de la que se
  acababa de hablar.
- Si Carla da la cantidad en "unidades", contar cada una como una unidad del producto
  (30 unidades de leche → 30 litros) y decírselo.
- Si el producto no existe, preguntar antes de darlo de alta.

---

## Registro de conversación

Cada evento se añade como una fila a `conversation_log.csv`:

| Campo | Contenido |
|---|---|
| `actor` | `user` (Carla), `agent` (respuesta final o petición de una tool) o `tool` (resultado que devolvió la API) |
| `message` | El texto del mensaje, los argumentos de la tool en JSON o el resultado de la API |
| `tool_call` | Nombre de la tool (vacío en los mensajes de `user` y en las respuestas finales del agente) |
| `timestamp` | ISO 8601 con zona horaria, p. ej. `2026-11-09T20:05:39+01:00` |

El fichero se abre en modo `"a"` (solo añadir) en cada evento: nunca se sobrescribe, y un
`Ctrl + C` a mitad de sesión no pierde nada. La cabecera solo se escribe cuando el fichero
no existe. El que hay en el repo contiene dos sesiones de demostración seguidas.

---

## Decisiones y limitaciones

- **Modelo `openai/gpt-oss-120b` en Groq.** Es el modelo de chat más capaz con tool calling
  que ofrecía Groq en noviembre de 2026 (ya no tenía modelos Llama de chat). Se cambia con la
  variable `GROQ_MODEL` en `.env`.
- **SDK `openai` con la `base_url` de Groq.** Groq implementa la misma API que OpenAI, así
  que el mismo código sirve para otros proveedores compatibles cambiando la URL y la clave.
- **`temperature=0.2`.** Un agente que modifica datos tiene que ser predecible.
- **Cantidades decimales.** `quantity` y `delta` admiten decimales (3.5 kg de cacao).
- **Un solo almacén.** La tienda tiene dos locales, pero el enunciado pide un único
  inventario. Una ampliación natural sería añadir un campo `location`.
- **Limitación conocida.** De vez en cuando el modelo escribe dos versiones de su
  respuesta final seguidas (por ejemplo, "¿Qué más necesitas?¡Listo! Si necesitas…"). Viene
  del propio texto que genera el modelo, no del bucle: el historial guarda exactamente lo
  que devuelve el LLM.

## Estructura

```
├── api/
│   ├── __init__.py
│   └── app.py              # API FastAPI
├── agent.py                # tools, bucle del agente, registro y CLI
├── products.csv            # inventario (se crea vacío si no existe)
├── conversation_log.csv    # registro de conversaciones, solo se añaden filas
├── .env.example            # plantilla de la clave de Groq
├── pyproject.toml / uv.lock
└── README.md
```

Proyecto del bootcamp de AI Engineering de [4Geeks Academy](https://4geeks.com), creado a
partir de la plantilla [python-hello](https://github.com/4GeeksAcademy/python-hello).

"""API de inventario de la tienda de Carla (suministros para cafeterías).

Los productos viven en ``products.csv``, en la raíz del proyecto, para que
sobrevivan a un reinicio del servidor. Cada petición lee el fichero y cada
cambio lo reescribe entero: con unas decenas de productos es lo más simple y
no hace falta base de datos.

Arranque (desde la raíz del proyecto):
    uv run uvicorn api.app:app --reload
"""

import csv
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

PRODUCTS_FILE = Path(__file__).resolve().parent.parent / "products.csv"
FIELDS = ["id", "name", "quantity", "unit"]
DEFAULT_THRESHOLD = 10

app = FastAPI(
    title="Inventario de Carla",
    description="Stock de una tienda de suministros para cafeterías. "
    "Lo usa el agente de IA como conjunto de herramientas.",
)

# FastAPI atiende las peticiones en varios hilos a la vez. Sin este candado,
# dos cambios simultáneos podrían leer el mismo CSV y el segundo pisaría al
# primero (una venta "desaparecería").
_lock = threading.Lock()


# ---------- Modelos: la forma de los datos que entran y salen ----------

class ProductIn(BaseModel):
    """Lo que hay que enviar para dar de alta un producto."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100, examples=["Leche de avena"])
    quantity: float = Field(ge=0, description="Cantidad inicial en stock", examples=[24])
    unit: str = Field(
        min_length=1,
        max_length=30,
        description="Unidad de medida: unidades, kg, litros, bolsas (1 kg)...",
        examples=["litros"],
    )


class Product(ProductIn):
    """Un producto tal y como está guardado (con su id)."""

    id: int


class StockChange(BaseModel):
    """Movimiento de stock sobre un producto existente."""

    delta: float = Field(
        description="Positivo para entradas (entregas), negativo para salidas (ventas, mermas)",
        examples=[30, -12],
    )


# ---------- Persistencia en CSV ----------

def _format_number(n: float) -> str:
    """30.0 -> '30' y 2.50 -> '2.5', para que el CSV se lea bien a mano."""
    return str(int(n)) if float(n).is_integer() else str(round(n, 3))


def _read_products() -> list[dict]:
    if not PRODUCTS_FILE.exists():
        _write_products([])
        return []
    with PRODUCTS_FILE.open(newline="", encoding="utf-8") as f:
        return [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "quantity": float(row["quantity"]),
                "unit": row["unit"],
            }
            for row in csv.DictReader(f)
        ]


def _write_products(products: list[dict]) -> None:
    # Escritura atómica: primero a un fichero temporal y luego se renombra.
    # Si el proceso muere a mitad, products.csv sigue intacto.
    tmp = PRODUCTS_FILE.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in products:
            writer.writerow({**p, "quantity": _format_number(p["quantity"])})
    os.replace(tmp, PRODUCTS_FILE)


def _describe(p: dict) -> str:
    return f"'{p['name']}' (id {p['id']}): {_format_number(p['quantity'])} {p['unit']}"


# ---------- Endpoints ----------

@app.get("/inventory", response_model=list[Product], summary="Lista todos los productos")
def list_products():
    return _read_products()


@app.post(
    "/inventory",
    response_model=Product,
    status_code=201,
    summary="Da de alta un producto nuevo",
)
def add_product(data: ProductIn):
    with _lock:
        products = _read_products()
        existing = next(
            (p for p in products if p["name"].casefold() == data.name.casefold()), None
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Ya existe el producto {_describe(existing)}. "
                f"Para cambiar su stock usa PATCH /inventory/{existing['id']}.",
            )
        product = {"id": max((p["id"] for p in products), default=0) + 1, **data.model_dump()}
        products.append(product)
        _write_products(products)
    return product


# Va antes que /inventory/{product_id} por costumbre: FastAPI prueba las rutas
# en orden y así "alerts" nunca se confundiría con un id.
@app.get(
    "/inventory/alerts",
    response_model=list[Product],
    summary="Productos con stock por debajo del umbral",
)
def low_stock_alerts(
    threshold: float = Query(
        DEFAULT_THRESHOLD, ge=0, description="Avisa de los productos con cantidad menor que esto"
    ),
):
    low = [p for p in _read_products() if p["quantity"] < threshold]
    return sorted(low, key=lambda p: p["quantity"])  # los más urgentes primero


@app.patch(
    "/inventory/{product_id}",
    response_model=Product,
    summary="Suma o resta stock a un producto existente",
)
def update_stock(product_id: int, change: StockChange):
    if change.delta == 0:
        raise HTTPException(
            status_code=422,
            detail="delta no puede ser 0: usa un número positivo para entradas o negativo para salidas.",
        )
    with _lock:
        products = _read_products()
        product = next((p for p in products if p["id"] == product_id), None)
        if product is None:
            raise HTTPException(
                status_code=404,
                detail=f"No existe ningún producto con id {product_id}. "
                "Consulta GET /inventory para ver los ids disponibles.",
            )
        new_quantity = round(product["quantity"] + change.delta, 3)
        if new_quantity < 0:
            raise HTTPException(
                status_code=409,
                detail=f"Stock insuficiente: hay {_describe(product)} y se intentan sacar "
                f"{_format_number(-change.delta)}. No se ha cambiado nada.",
            )
        product["quantity"] = new_quantity
        _write_products(products)
    return product

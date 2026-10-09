"""Contador de tokens y costo aproximado de una ejecución.

Lo comparten el agente principal, los investigadores delegados y el lector de
fuentes, que pueden correr en hilos distintos: por eso lleva un candado.
"""

import threading

# Dólares por millón de tokens: (entrada, salida, lectura de caché).
# Son precios de lista de Anthropic y pueden cambiar; revisa https://www.anthropic.com/pricing
PRECIOS = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
RECARGO_ESCRITURA_CACHE = 1.25  # escribir en caché cuesta 1,25 veces la entrada normal
PRECIO_BUSQUEDA = 0.01  # dólares por búsqueda web


class Uso:
    def __init__(self) -> None:
        self._candado = threading.Lock()
        self._por_modelo: dict[str, dict[str, int]] = {}
        self.busquedas = 0

    def registrar(
        self, modelo: str, entrada: int, salida: int, cache_lectura: int = 0, cache_escritura: int = 0, busquedas: int = 0
    ) -> None:
        """`entrada` son los tokens de entrada sin caché; `busquedas`, las búsquedas de pago."""
        with self._candado:
            fila = self._por_modelo.setdefault(
                modelo, {"entrada": 0, "salida": 0, "cache_lectura": 0, "cache_escritura": 0}
            )
            fila["entrada"] += entrada
            fila["salida"] += salida
            fila["cache_lectura"] += cache_lectura
            fila["cache_escritura"] += cache_escritura
            self.busquedas += busquedas

    def costo(self) -> tuple[float, list[str]]:
        """Devuelve (dólares, modelos sin precio conocido)."""
        total, desconocidos = self.busquedas * PRECIO_BUSQUEDA, []
        with self._candado:
            for modelo, fila in self._por_modelo.items():
                precio = next((p for nombre, p in PRECIOS.items() if modelo.startswith(nombre)), None)
                if precio is None:
                    desconocidos.append(modelo)
                    continue
                entrada, salida, lectura = precio
                total += (
                    fila["entrada"] * entrada
                    + fila["cache_escritura"] * entrada * RECARGO_ESCRITURA_CACHE
                    + fila["cache_lectura"] * lectura
                    + fila["salida"] * salida
                ) / 1_000_000
        return total, desconocidos

    def resumen(self) -> str:
        with self._candado:
            lineas = [
                f"  {modelo}: {f['entrada'] + f['cache_lectura'] + f['cache_escritura']:,} tokens de entrada "
                f"({f['cache_lectura']:,} desde caché), {f['salida']:,} de salida"
                for modelo, f in self._por_modelo.items()
            ]
        total, desconocidos = self.costo()
        if self.busquedas:
            lineas.append(f"  búsquedas web de pago: {self.busquedas}")
        if desconocidos:
            # Pasa con Azure OpenAI: el precio depende del modelo detrás de cada despliegue.
            lineas.append(f"  costo: sin precio conocido para {', '.join(desconocidos)}")
        if total:
            lineas.append(f"  costo aproximado: ${total:.2f}")
        return "\n".join(lineas)


USO = Uso()

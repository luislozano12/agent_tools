"""Proveedores de modelo: Anthropic (Claude) y Azure OpenAI.

Cada API tiene su propio formato de mensajes, herramientas y streaming. Estas
clases lo traducen a una interfaz común para que el bucle del agente sea el mismo:

    pedir(...)              -> Respuesta
    guardar_respuesta(...)  añade la respuesta del modelo al historial
    guardar_resultados(...) añade los resultados de las herramientas al historial
    completar(...)          una pregunta suelta, sin herramientas (la usa el lector)

El historial (`mensajes`) es una lista que solo entiende el proveedor que la creó.
"""

import json
import os
from dataclasses import dataclass, field

import anthropic

try:
    import openai
except ImportError:  # solo hace falta para Azure OpenAI
    openai = None

from uso import USO

ERRORES_API = (anthropic.APIError,) + ((openai.APIError,) if openai else ())

MAX_TOKENS = 64_000
MAX_TOKENS_TOPE = 128_000  # si una respuesta se corta, se repite una vez con este tope


@dataclass
class Llamada:
    id: str
    nombre: str
    entrada: object  # normalmente un dict; otra cosa si el modelo envió argumentos rotos


@dataclass
class Respuesta:
    # "fin" | "herramientas" | "pausa" | "cortada", o el motivo por el que el modelo se detuvo
    estado: str
    texto: str = ""
    llamadas: list[Llamada] = field(default_factory=list)
    busquedas: list[str] = field(default_factory=list)  # búsquedas hechas en el servidor
    cruda: object = None  # la respuesta en el formato del proveedor, para el historial


class ProveedorAnthropic:
    nombre = "anthropic"
    admite_mas_tokens = True

    MODELO = "claude-opus-5-5"
    MODELO_LECTOR = "claude-haiku-5-5"
    ESFUERZO = "medium"

    def __init__(self, modelo: str | None = None, esfuerzo: str | None = None, cliente=None) -> None:
        self.cliente = cliente or anthropic.Anthropic()
        self.modelo = modelo or self.MODELO
        self.esfuerzo = esfuerzo or self.ESFUERZO

    def _herramientas(self, herramientas: list) -> list:
        # La búsqueda web la ejecuta Anthropic en su servidor. Las demás se
        # marcan con eager_input_streaming para que sus argumentos lleguen en streaming.
        return [{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}] + [
            {**h, "eager_input_streaming": True} for h in herramientas
        ]

    def pedir(self, sistema: str, herramientas: list, mensajes: list, mas_tokens: bool = False, en_vivo: bool = False) -> Respuesta:
        # Con eager_input_streaming el SDK puede recibir JSON de herramienta roto y
        # lanzar ValueError; en ese caso se repite la petición una vez.
        for intento in range(2):
            try:
                with self.cliente.beta.messages.stream(
                    model=self.modelo,
                    max_tokens=MAX_TOKENS_TOPE if mas_tokens else MAX_TOKENS,
                    system=sistema,
                    tools=self._herramientas(herramientas),
                    messages=mensajes,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.esfuerzo},
                    cache_control={"type": "ephemeral"},
                    # Si el modelo rechaza la petición, la API la reintenta en otro modelo.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    if en_vivo:
                        # El texto se imprime según llega, en vez de esperar al final del paso.
                        for evento in stream:
                            if evento.type == "content_block_start" and evento.content_block.type == "text":
                                print()
                            elif evento.type == "text":
                                print(evento.text, end="", flush=True)
                        print()
                    mensaje = stream.get_final_message()
                break
            except ValueError:
                if intento == 1:
                    raise
                if en_vivo:
                    print("\n  [respuesta dañada, se repite la petición]")

        self._registrar(mensaje)
        estado = {"end_turn": "fin", "tool_use": "herramientas", "pause_turn": "pausa", "max_tokens": "cortada"}
        return Respuesta(
            estado=estado.get(mensaje.stop_reason, str(mensaje.stop_reason)),
            texto="\n".join(b.text for b in mensaje.content if b.type == "text").strip(),
            llamadas=[Llamada(b.id, b.name, b.input) for b in mensaje.content if b.type == "tool_use"],
            busquedas=[
                b.input["query"]
                for b in mensaje.content
                if b.type == "server_tool_use" and isinstance(b.input, dict) and b.input.get("query")
            ],
            cruda=mensaje,
        )

    def _registrar(self, mensaje) -> None:
        uso = mensaje.usage
        USO.registrar(
            mensaje.model,
            entrada=uso.input_tokens or 0,
            salida=uso.output_tokens or 0,
            cache_lectura=getattr(uso, "cache_read_input_tokens", None) or 0,
            cache_escritura=getattr(uso, "cache_creation_input_tokens", None) or 0,
            busquedas=getattr(getattr(uso, "server_tool_use", None), "web_search_requests", None) or 0,
        )

    def guardar_respuesta(self, mensajes: list, respuesta: Respuesta) -> None:
        # Se guarda el contenido completo (texto, razonamiento y llamadas a
        # herramientas), no solo el texto: el modelo lo necesita en la vuelta siguiente.
        mensajes.append({"role": "assistant", "content": respuesta.cruda.content})

    def guardar_resultados(self, mensajes: list, resultados: list[tuple[Llamada, str, bool]]) -> None:
        # Todos los resultados van juntos en un único mensaje de usuario.
        mensajes.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": llamada.id, "content": texto, "is_error": es_error}
                    for llamada, texto, es_error in resultados
                ],
            }
        )

    def completar(self, sistema: str, contenido: str) -> str:
        mensaje = self.cliente.messages.create(
            model=self.MODELO_LECTOR,
            max_tokens=8000,
            system=sistema,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": contenido}],
        )
        self._registrar(mensaje)
        if mensaje.stop_reason == "refusal":
            return ""
        return "\n".join(b.text for b in mensaje.content if b.type == "text").strip()


# Azure OpenAI no trae búsqueda web: se le da una herramienta local (DuckDuckGo).
BUSCAR_WEB = {
    "name": "buscar_web",
    "description": (
        "Busca en la web y devuelve título, URL y un extracto de cada resultado. Los "
        "extractos son superficiales: sirven para elegir qué fuentes leer después. "
        "Puedes lanzar varias búsquedas distintas en un mismo paso."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "consulta": {"type": "string", "description": "Términos de búsqueda, como en un buscador."},
        },
        "required": ["consulta"],
    },
}


class ProveedorAzure:
    """Azure OpenAI por la API de Chat Completions.

    Se configura con variables de entorno:
      AZURE_OPENAI_ENDPOINT           https://TU-RECURSO.openai.azure.com
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_DEPLOYMENT         nombre de tu despliegue (o --modelo)
      AZURE_OPENAI_DEPLOYMENT_LECTOR  despliegue barato para leer fuentes (opcional)
      OPENAI_API_VERSION              opcional
    """

    nombre = "azure"
    admite_mas_tokens = False  # no se fija tope de salida: se usa el del despliegue

    VERSION_API = "2024-10-21"

    def __init__(self, modelo: str | None = None, esfuerzo: str | None = None, cliente=None) -> None:
        self.modelo = modelo or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self.modelo_lector = os.environ.get("AZURE_OPENAI_DEPLOYMENT_LECTOR") or self.modelo
        # Azure OpenAI solo conoce low, medium y high, y solo en modelos de
        # razonamiento: por eso no se envía nada si no se pidió con --esfuerzo.
        self.esfuerzo = {"xhigh": "high", "max": "high"}.get(esfuerzo, esfuerzo)
        self.cliente = cliente or openai.AzureOpenAI(
            api_version=os.environ.get("OPENAI_API_VERSION") or self.VERSION_API
        )

    def _herramientas(self, herramientas: list) -> list:
        return [
            {
                "type": "function",
                "function": {"name": h["name"], "description": h["description"], "parameters": h["input_schema"]},
            }
            for h in herramientas + [BUSCAR_WEB]
        ]

    def pedir(self, sistema: str, herramientas: list, mensajes: list, mas_tokens: bool = False, en_vivo: bool = False) -> Respuesta:
        opciones = {"reasoning_effort": self.esfuerzo} if self.esfuerzo else {}
        stream = self.cliente.chat.completions.create(
            model=self.modelo,
            messages=[{"role": "system", "content": sistema}] + mensajes,
            tools=self._herramientas(herramientas),
            stream=True,
            stream_options={"include_usage": True},
            **opciones,
        )

        # El texto y los argumentos de cada llamada llegan en fragmentos que hay que unir.
        texto, parciales, motivo = [], {}, None
        for trozo in stream:
            if trozo.usage:
                self._registrar(self.modelo, trozo.usage)
            if not trozo.choices:
                continue
            opcion = trozo.choices[0]
            delta = opcion.delta
            if delta and delta.content:
                if en_vivo:
                    print(("" if texto else "\n") + delta.content, end="", flush=True)
                texto.append(delta.content)
            for parte in (delta.tool_calls or []) if delta else []:
                parcial = parciales.setdefault(parte.index, {"id": "", "nombre": "", "argumentos": ""})
                parcial["id"] = parte.id or parcial["id"]
                if parte.function:
                    parcial["nombre"] = parte.function.name or parcial["nombre"]
                    parcial["argumentos"] += parte.function.arguments or ""
            motivo = opcion.finish_reason or motivo
        if en_vivo:
            print()

        llamadas_crudas = [parciales[i] for i in sorted(parciales)]
        cruda = {"role": "assistant", "content": "".join(texto) or None}
        if llamadas_crudas:
            cruda["tool_calls"] = [
                {"id": p["id"], "type": "function", "function": {"name": p["nombre"], "arguments": p["argumentos"]}}
                for p in llamadas_crudas
            ]

        if motivo == "length":
            estado = "cortada"
        elif llamadas_crudas and motivo in ("tool_calls", "stop"):
            estado = "herramientas"
        elif motivo == "stop":
            estado = "fin"
        else:
            estado = str(motivo)  # p. ej. "content_filter"
        return Respuesta(
            estado=estado,
            texto="".join(texto).strip(),
            llamadas=[Llamada(p["id"], p["nombre"], _json(p["argumentos"])) for p in llamadas_crudas],
            cruda=cruda,
        )

    @staticmethod
    def _registrar(modelo: str, uso) -> None:
        # En esta API los tokens de entrada ya incluyen los leídos de caché.
        en_cache = getattr(getattr(uso, "prompt_tokens_details", None), "cached_tokens", None) or 0
        USO.registrar(
            modelo,
            entrada=(uso.prompt_tokens or 0) - en_cache,
            salida=uso.completion_tokens or 0,
            cache_lectura=en_cache,
        )

    def guardar_respuesta(self, mensajes: list, respuesta: Respuesta) -> None:
        mensajes.append(respuesta.cruda)

    def guardar_resultados(self, mensajes: list, resultados: list[tuple[Llamada, str, bool]]) -> None:
        # Un mensaje por resultado. No hay campo de error: el texto ya empieza por "Error:".
        for llamada, texto, _ in resultados:
            mensajes.append({"role": "tool", "tool_call_id": llamada.id, "content": texto})

    def completar(self, sistema: str, contenido: str) -> str:
        respuesta = self.cliente.chat.completions.create(
            model=self.modelo_lector,
            messages=[{"role": "system", "content": sistema}, {"role": "user", "content": contenido}],
        )
        if respuesta.usage:
            self._registrar(self.modelo_lector, respuesta.usage)
        return (respuesta.choices[0].message.content or "").strip() if respuesta.choices else ""


def _json(argumentos: str):
    try:
        return json.loads(argumentos or "{}")
    except json.JSONDecodeError:
        return None  # ejecutar() lo devuelve al modelo como error de argumentos


def crear(nombre: str, modelo: str | None, esfuerzo: str | None):
    """Crea el proveedor; termina con un mensaje claro si falta configuración."""
    if nombre == "azure":
        if openai is None:
            raise SystemExit("Falta el paquete 'openai'. Instálalo con: pip install openai ddgs")
        faltan = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY") if not os.environ.get(v)]
        if not (modelo or os.environ.get("AZURE_OPENAI_DEPLOYMENT")):
            faltan.append("AZURE_OPENAI_DEPLOYMENT (o --modelo)")
        if faltan:
            raise SystemExit("Faltan variables de entorno para Azure OpenAI: " + ", ".join(faltan))
        return ProveedorAzure(modelo, esfuerzo)
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("Falta la clave de API. Define la variable de entorno ANTHROPIC_API_KEY.")
    return ProveedorAnthropic(modelo, esfuerzo)


def proveedor_por_defecto() -> str:
    elegido = os.environ.get("AGENTE_PROVEEDOR", "").lower()
    if elegido in ("anthropic", "azure"):
        return elegido
    hay_anthropic = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    return "azure" if os.environ.get("AZURE_OPENAI_ENDPOINT") and not hay_anthropic else "anthropic"


def describir_error(e: Exception) -> str:
    # Los dos SDK usan los mismos nombres de clase para sus errores.
    clases = {c.__name__ for c in type(e).__mro__}
    if "AuthenticationError" in clases:
        return "La clave de API no es válida. Revisa tus variables de entorno."
    if "RateLimitError" in clases:
        return "Límite de uso alcanzado. Espera un momento y vuelve a intentarlo."
    if "APIStatusError" in clases:
        return f"Error de la API ({e.status_code}): {e.message}"
    if "APIConnectionError" in clases:
        return "No se pudo conectar con la API. Revisa tu conexión a internet y el endpoint."
    return f"Error de la API: {e}"

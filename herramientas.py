"""Herramientas del agente.

Cada herramienta tiene dos mitades:
  1. La definición (nombre, descripción, esquema): es lo único que ve el modelo.
  2. La función de Python: se ejecuta aquí, en tu máquina, cuando el modelo la pide.

Las definiciones están en un formato neutro (name, description, input_schema);
cada proveedor las traduce al suyo. La búsqueda web depende del proveedor: con
Anthropic la ejecuta su servidor; con Azure OpenAI se usa buscar_web, de aquí.
"""

import io
import ipaddress
import os
import socket
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PyPdfError

import lector
from proveedores import ERRORES_API

CARPETA_INFORMES = Path(__file__).parent / "informes"

# Una página puede tener cientos de miles de caracteres. Se entrega por tramos
# para no llenar la ventana de contexto con una sola llamada.
MAX_CARACTERES = 15_000

# Topes de la descarga: tamaño, redirecciones y tiempo total.
MAX_BYTES = 20_000_000
MAX_REDIRECCIONES = 5
MAX_SEGUNDOS = 60

# Wikipedia responde 403 si el cliente no se identifica con un contacto real
# (un correo o una URL). Defínelo en la variable de entorno AGENTE_CONTACTO;
# el resto de sitios funciona sin él.
CONTACTO = os.environ.get("AGENTE_CONTACTO", "").strip()
USER_AGENT = "agente-investigador/0.2 (proyecto educativo" + (f"; {CONTACTO})" if CONTACTO else ")")

_URL = {"type": "string", "description": "URL completa, http o https."}

HERRAMIENTAS = [
    {
        "name": "consultar_fuente",
        "description": (
            "Lee una página web o un PDF completo y devuelve solo lo que aporta a tu "
            "pregunta, con citas textuales. Es la forma normal de leer una fuente "
            "después de buscar: el documento entero lo lee un modelo auxiliar, así que "
            "tu contexto solo recibe lo relevante. Haz preguntas concretas (qué cifra, "
            "de qué fecha, según quién); puedes consultar varias fuentes a la vez."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": _URL,
                "pregunta": {"type": "string", "description": "Qué quieres saber de esta fuente."},
            },
            "required": ["url", "pregunta"],
        },
    },
    {
        "name": "leer_pagina",
        "description": (
            "Descarga una página web o un PDF y devuelve su texto literal, por tramos. "
            "Úsala solo cuando necesites el texto exacto (una tabla, una cita larga, un "
            "pasaje que consultar_fuente no resolvió). Si queda más contenido, el "
            "resultado indica con qué valor de 'inicio' continuar."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": _URL,
                "inicio": {
                    "type": "integer",
                    "description": "Carácter desde el que continuar la lectura. Por defecto 0.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "verificar_cita",
        "description": (
            "Comprueba si una fuente respalda de verdad una afirmación. Devuelve "
            "RESPALDADA, PARCIAL o NO RESPALDADA junto con el pasaje textual. Úsala "
            "antes de guardar el informe sobre las cifras, fechas y conclusiones más "
            "importantes, cada una contra la URL que vas a citar."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": _URL,
                "afirmacion": {
                    "type": "string",
                    "description": "La afirmación tal como aparecerá en el informe.",
                },
            },
            "required": ["url", "afirmacion"],
        },
    },
    {
        "name": "listar_informes",
        "description": (
            "Lista los informes guardados en investigaciones anteriores, del más "
            "reciente al más antiguo. Úsala al empezar para saber si el tema ya se "
            "investigó; con 'consulta' filtra por palabras clave."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Palabras clave separadas por espacios. Vacío: lista todos.",
                },
            },
        },
    },
    {
        "name": "leer_informe",
        "description": (
            "Devuelve el texto de un informe anterior. Sirve como punto de partida: "
            "los datos pueden haber cambiado desde la fecha del informe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre que devolvió listar_informes."},
                "inicio": {
                    "type": "integer",
                    "description": "Carácter desde el que continuar la lectura. Por defecto 0.",
                },
            },
            "required": ["nombre"],
        },
    },
    {
        "name": "guardar_informe",
        "description": (
            "Guarda el informe final en un archivo Markdown. Llámala al terminar la "
            "investigación, con el informe completo. Si después el usuario pide "
            "cambios, vuelve a llamarla con el mismo nombre y el informe completo "
            "actualizado."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {
                    "type": "string",
                    "description": "Nombre corto del archivo, sin ruta. Ej.: 'baterias-de-sodio'.",
                },
                "contenido": {"type": "string", "description": "Informe completo en Markdown."},
            },
            "required": ["nombre", "contenido"],
        },
    },
]

# Los investigadores delegados buscan y leen, pero no guardan informes.
HERRAMIENTAS_INVESTIGADOR = [h for h in HERRAMIENTAS if h["name"] in ("consultar_fuente", "leer_pagina")]

MAX_RESULTADOS_BUSQUEDA = 8


class ErrorHerramienta(Exception):
    """Fallo que se le devuelve al modelo como texto para que pueda corregir el rumbo."""


def _verificar_url_publica(url: str) -> None:
    # La URL la elige el modelo, influido por lo que leyó en la web: no debe
    # poder apuntar a servicios de la red local.
    partes = urlparse(url)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ErrorHerramienta("La URL debe empezar por http:// o https://")
    try:
        direcciones = socket.getaddrinfo(partes.hostname, None)
    except socket.gaierror:
        raise ErrorHerramienta(f"No se pudo resolver el dominio {partes.hostname}")
    for direccion in direcciones:
        if not ipaddress.ip_address(direccion[4][0]).is_global:
            raise ErrorHerramienta("Solo se permiten direcciones públicas de internet.")


def _descargar(url: str) -> tuple[bytes, str, str | None]:
    """Descarga la URL y devuelve (datos, tipo de contenido, codificación declarada)."""
    limite = time.monotonic() + MAX_SEGUNDOS
    try:
        with httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as cliente:
            # Las redirecciones se siguen a mano para verificar cada salto: una
            # página pública no debe poder redirigir hacia la red local.
            for _ in range(MAX_REDIRECCIONES + 1):
                _verificar_url_publica(url)
                with cliente.stream("GET", url) as respuesta:
                    if respuesta.has_redirect_location:
                        url = urljoin(url, respuesta.headers["location"])
                        continue
                    respuesta.raise_for_status()
                    datos = bytearray()
                    for trozo in respuesta.iter_bytes():
                        datos += trozo
                        if len(datos) > MAX_BYTES:
                            raise ErrorHerramienta(
                                f"El archivo supera el máximo de {MAX_BYTES // 1_000_000} MB."
                            )
                        if time.monotonic() > limite:
                            raise ErrorHerramienta("La descarga tardó demasiado.")
                    return bytes(datos), respuesta.headers.get("content-type", ""), respuesta.charset_encoding
            raise ErrorHerramienta("Demasiadas redirecciones.")
    except httpx.HTTPStatusError as e:
        raise ErrorHerramienta(f"La página respondió con el código {e.response.status_code}.")
    except httpx.HTTPError as e:
        raise ErrorHerramienta(f"No se pudo descargar la página: {type(e).__name__}")


def _texto_pdf(datos: bytes) -> str:
    try:
        textos = [(pagina.extract_text() or "").strip() for pagina in PdfReader(io.BytesIO(datos)).pages]
    except (PyPdfError, ValueError, KeyError) as e:
        raise ErrorHerramienta(f"No se pudo leer el PDF: {type(e).__name__}")
    if not any(textos):
        raise ErrorHerramienta("El PDF no tiene texto extraíble (quizá es un escaneo).")
    return "\n\n".join(f"[Página {numero}]\n{texto}" for numero, texto in enumerate(textos, start=1) if texto)


def _texto_html(datos: bytes, codificacion: str | None) -> str:
    sopa = BeautifulSoup(datos, "html.parser", from_encoding=codificacion)
    for etiqueta in sopa(["script", "style", "nav", "header", "footer", "aside", "form"]):
        etiqueta.decompose()
    lineas = (linea.strip() for linea in sopa.get_text("\n").splitlines())
    return "\n".join(linea for linea in lineas if linea)


# Leer una página por tramos, consultarla y luego verificar una cita son varias
# llamadas sobre la misma URL: el texto se guarda para descargarla una sola vez.
_candados_url: dict[str, threading.Lock] = {}
_candado_global = threading.Lock()


@lru_cache(maxsize=64)
def _texto_en_cache(url: str) -> str:
    datos, tipo, codificacion = _descargar(url)
    if "pdf" in tipo or datos.startswith(b"%PDF-"):
        texto = _texto_pdf(datos)
    elif "html" in tipo:
        texto = _texto_html(datos, codificacion)
    elif "text/plain" in tipo:
        texto = datos.decode(codificacion or "utf-8", errors="replace").strip()
    else:
        raise ErrorHerramienta(f"Tipo de contenido no soportado: {tipo or 'desconocido'}")
    if not texto:
        raise ErrorHerramienta("La página no tiene texto legible (quizá requiere JavaScript).")
    return texto


def texto_de(url: str) -> str:
    # Un candado por URL: si dos hilos piden la misma página a la vez, el
    # segundo espera y la encuentra ya en caché en lugar de descargarla otra vez.
    with _candado_global:
        candado = _candados_url.setdefault(url, threading.Lock())
    with candado:
        return _texto_en_cache(url)


def _tramo(texto: str, inicio: int) -> str:
    if inicio >= len(texto):
        raise ErrorHerramienta(f"'inicio' está fuera de rango: el texto tiene {len(texto)} caracteres.")
    fin = min(inicio + MAX_CARACTERES, len(texto))
    tramo = texto[inicio:fin]
    if fin < len(texto):
        tramo += (
            f"\n\n[Caracteres {inicio}-{fin} de {len(texto)}. "
            f"Para seguir leyendo, llama de nuevo con inicio={fin}.]"
        )
    return tramo


def leer_pagina(url: str, inicio: int = 0) -> str:
    return _tramo(texto_de(url), inicio)


def buscar_web(consulta: str) -> str:
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError:
        raise ErrorHerramienta("Falta el paquete 'ddgs' para buscar en la web (pip install ddgs).")
    try:
        resultados = DDGS().text(consulta, max_results=MAX_RESULTADOS_BUSQUEDA)
    except DDGSException as e:
        raise ErrorHerramienta(f"La búsqueda falló ({type(e).__name__}). Prueba con otros términos.")
    if not resultados:
        return "Sin resultados. Prueba con otros términos."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('href', '')}\n{(r.get('body') or '')[:400]}" for r in resultados
    )


def _ruta_informe(nombre: str) -> Path:
    # Solo se usa el nombre base: el modelo no puede salir de la carpeta.
    base = Path(nombre).name.removesuffix(".md").strip()
    if not base or base in (".", ".."):
        raise ErrorHerramienta("Nombre de archivo no válido.")
    return CARPETA_INFORMES / f"{base}.md"


def _titulo(texto: str) -> str:
    return next((linea.lstrip("# ").strip() for linea in texto.splitlines() if linea.strip()), "(vacío)")


def listar_informes(consulta: str = "") -> str:
    palabras = [p for p in consulta.lower().split() if len(p) > 2]
    filas = []
    rutas = sorted(CARPETA_INFORMES.glob("*.md"), key=lambda r: r.stat().st_mtime, reverse=True)
    for ruta in rutas:
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        if palabras and not any(p in f"{ruta.stem} {texto}".lower() for p in palabras):
            continue
        fecha = datetime.fromtimestamp(ruta.stat().st_mtime).date()
        filas.append(f"- {ruta.stem} ({fecha}, {len(texto):,} caracteres): {_titulo(texto)[:120]}")
    if not filas:
        return "No hay informes previos" + (" que coincidan con la consulta." if palabras else ".")
    return "\n".join(filas)


def leer_informe(nombre: str, inicio: int = 0) -> str:
    ruta = _ruta_informe(nombre)
    if not ruta.is_file():
        raise ErrorHerramienta(f"No existe el informe '{ruta.stem}'. Usa listar_informes para ver los nombres.")
    return _tramo(ruta.read_text(encoding="utf-8", errors="replace"), inicio)


# Informes escritos en esta ejecución: esos sí se pueden reemplazar (el usuario
# pidió cambios); los de ejecuciones anteriores nunca se sobrescriben.
_guardados_en_sesion: set[Path] = set()


def guardar_informe(nombre: str, contenido: str) -> str:
    ruta = _ruta_informe(nombre)
    CARPETA_INFORMES.mkdir(exist_ok=True)
    numero = 2
    while ruta.exists() and ruta not in _guardados_en_sesion:
        ruta = _ruta_informe(f"{Path(nombre).name.removesuffix('.md').strip()}-{numero}")
        numero += 1
    ruta.write_text(contenido, encoding="utf-8")
    _guardados_en_sesion.add(ruta)
    return f"Informe guardado en {ruta}"


def _texto(entrada: dict, campo: str) -> str:
    valor = entrada.get(campo)
    if not isinstance(valor, str) or not valor.strip():
        raise ErrorHerramienta(f"Argumento inválido: '{campo}' debe ser texto no vacío.")
    return valor


def _inicio(entrada: dict) -> int:
    valor = entrada.get("inicio", 0)
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < 0:
        raise ErrorHerramienta("Argumento inválido: 'inicio' debe ser un entero >= 0.")
    return valor


def ejecutar(nombre: str, entrada: dict, proveedor=None) -> tuple[str, bool]:
    """Ejecuta una herramienta y devuelve (resultado, es_error).

    Nunca lanza excepciones: un fallo se convierte en texto para el modelo,
    que así puede reintentar con otra URL o con otros argumentos.
    """
    try:
        if not isinstance(entrada, dict):
            raise ErrorHerramienta("Los argumentos deben ser un objeto JSON.")
        if nombre == "buscar_web":
            return buscar_web(_texto(entrada, "consulta")), False
        if nombre == "leer_pagina":
            return leer_pagina(_texto(entrada, "url"), _inicio(entrada)), False
        if nombre == "consultar_fuente":
            url, pregunta = _texto(entrada, "url"), _texto(entrada, "pregunta")
            return lector.consultar(proveedor, texto_de(url), url, pregunta), False
        if nombre == "verificar_cita":
            url, afirmacion = _texto(entrada, "url"), _texto(entrada, "afirmacion")
            return lector.verificar(proveedor, texto_de(url), url, afirmacion), False
        if nombre == "listar_informes":
            consulta = entrada.get("consulta", "")
            return listar_informes(consulta if isinstance(consulta, str) else ""), False
        if nombre == "leer_informe":
            return leer_informe(_texto(entrada, "nombre"), _inicio(entrada)), False
        if nombre == "guardar_informe":
            return guardar_informe(_texto(entrada, "nombre"), _texto(entrada, "contenido")), False
        raise ErrorHerramienta(f"La herramienta '{nombre}' no existe.")
    except (ErrorHerramienta, lector.ErrorLector) as e:
        return f"Error: {e}", True
    except ERRORES_API as e:
        return f"Error: el lector de fuentes falló ({type(e).__name__}). Prueba con leer_pagina.", True

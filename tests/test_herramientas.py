"""Pruebas de las herramientas locales. No usan la red ni la API."""

import httpx
import pytest

import herramientas
from herramientas import ErrorHerramienta

# Una IP literal pública: se valida sin consultar DNS.
PUBLICA = "http://93.184.216.34"


@pytest.fixture(autouse=True)
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(herramientas, "CARPETA_INFORMES", tmp_path / "informes")
    herramientas._texto_en_cache.cache_clear()
    herramientas._guardados_en_sesion.clear()


@pytest.fixture
def web(monkeypatch):
    """Sustituye la red por un diccionario {url: respuesta}; devuelve la lista de URL pedidas."""
    paginas: dict[str, httpx.Response] = {}
    pedidas: list[str] = []

    def manejador(peticion: httpx.Request) -> httpx.Response:
        pedidas.append(str(peticion.url))
        return paginas[str(peticion.url)]

    cliente_real = httpx.Client
    monkeypatch.setattr(
        herramientas.httpx, "Client", lambda **kw: cliente_real(transport=httpx.MockTransport(manejador), **kw)
    )
    return paginas, pedidas


def html(cuerpo: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=cuerpo.encode())


def pdf_con_texto(texto: str) -> bytes:
    flujo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode()
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(flujo), flujo),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    salida, posiciones = bytearray(b"%PDF-1.4\n"), []
    for numero, objeto in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        salida += b"%d 0 obj\n%s\nendobj\n" % (numero, objeto)
    inicio_xref = len(salida)
    salida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    for posicion in posiciones:
        salida += b"%010d 00000 n \n" % posicion
    salida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objetos) + 1, inicio_xref)
    return bytes(salida)


# --- URL permitidas ---


@pytest.mark.parametrize(
    "url",
    [
        "ftp://93.184.216.34/archivo",
        "file:///C:/Windows/win.ini",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "no-es-una-url",
    ],
)
def test_rechaza_url_no_publica(url):
    with pytest.raises(ErrorHerramienta):
        herramientas._verificar_url_publica(url)


def test_acepta_ip_publica():
    herramientas._verificar_url_publica(PUBLICA + "/pagina")


# --- Descarga ---


def test_lee_html_y_quita_lo_que_no_es_contenido(web):
    paginas, _ = web
    paginas[PUBLICA + "/"] = html("<nav>menú</nav><p>Hola mundo</p><script>x()</script><footer>pie</footer>")
    assert herramientas.leer_pagina(PUBLICA + "/") == "Hola mundo"


def test_sigue_redirecciones_publicas(web):
    paginas, _ = web
    paginas[PUBLICA + "/a"] = httpx.Response(302, headers={"location": "/b"})
    paginas[PUBLICA + "/b"] = html("<p>destino</p>")
    assert herramientas.leer_pagina(PUBLICA + "/a") == "destino"


def test_bloquea_redireccion_a_red_local(web):
    paginas, pedidas = web
    paginas[PUBLICA + "/a"] = httpx.Response(302, headers={"location": "http://127.0.0.1:8080/secreto"})
    with pytest.raises(ErrorHerramienta, match="públicas"):
        herramientas.leer_pagina(PUBLICA + "/a")
    assert pedidas == [PUBLICA + "/a"]  # nunca llegó a pedirse la dirección local


def test_corta_bucle_de_redirecciones(web):
    paginas, _ = web
    paginas[PUBLICA + "/a"] = httpx.Response(302, headers={"location": "/a"})
    with pytest.raises(ErrorHerramienta, match="redirecciones"):
        herramientas.leer_pagina(PUBLICA + "/a")


def test_limite_de_tamano(web, monkeypatch):
    paginas, _ = web
    monkeypatch.setattr(herramientas, "MAX_BYTES", 1000)
    paginas[PUBLICA + "/"] = html("<p>" + "x" * 5000 + "</p>")
    with pytest.raises(ErrorHerramienta, match="máximo"):
        herramientas.leer_pagina(PUBLICA + "/")


def test_error_http_y_tipo_no_soportado(web):
    paginas, _ = web
    paginas[PUBLICA + "/404"] = httpx.Response(404)
    paginas[PUBLICA + "/img"] = httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")
    with pytest.raises(ErrorHerramienta, match="404"):
        herramientas.leer_pagina(PUBLICA + "/404")
    with pytest.raises(ErrorHerramienta, match="no soportado"):
        herramientas.leer_pagina(PUBLICA + "/img")


def test_lee_pdf(web):
    paginas, _ = web
    paginas[PUBLICA + "/doc"] = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=pdf_con_texto("Hola PDF")
    )
    assert "Hola PDF" in herramientas.leer_pagina(PUBLICA + "/doc")


def test_tramos_y_cache(web, monkeypatch):
    paginas, pedidas = web
    monkeypatch.setattr(herramientas, "MAX_CARACTERES", 10)
    paginas[PUBLICA + "/"] = html("<p>0123456789ABCDEFGHIJ</p>")
    primero = herramientas.leer_pagina(PUBLICA + "/")
    assert primero.startswith("0123456789") and "inicio=10" in primero
    assert herramientas.leer_pagina(PUBLICA + "/", 10) == "ABCDEFGHIJ"
    assert len(pedidas) == 1  # el segundo tramo salió de la caché
    with pytest.raises(ErrorHerramienta, match="fuera de rango"):
        herramientas.leer_pagina(PUBLICA + "/", 20)


# --- Informes ---


def test_guardar_no_sale_de_la_carpeta():
    herramientas.guardar_informe("../../fuera", "contenido")
    assert (herramientas.CARPETA_INFORMES / "fuera.md").read_text(encoding="utf-8") == "contenido"
    assert not (herramientas.CARPETA_INFORMES.parent.parent / "fuera.md").exists()


def test_guardar_no_sobrescribe_informes_anteriores():
    herramientas.CARPETA_INFORMES.mkdir()
    anterior = herramientas.CARPETA_INFORMES / "tema.md"
    anterior.write_text("viejo", encoding="utf-8")

    herramientas.guardar_informe("tema", "nuevo")
    assert anterior.read_text(encoding="utf-8") == "viejo"
    assert (herramientas.CARPETA_INFORMES / "tema-2.md").read_text(encoding="utf-8") == "nuevo"


def test_guardar_reemplaza_lo_escrito_en_esta_sesion():
    herramientas.guardar_informe("tema", "v1")
    herramientas.guardar_informe("tema", "v2")
    assert (herramientas.CARPETA_INFORMES / "tema.md").read_text(encoding="utf-8") == "v2"
    assert len(list(herramientas.CARPETA_INFORMES.glob("*.md"))) == 1


def test_listar_y_leer_informes():
    assert "No hay informes" in herramientas.listar_informes()
    herramientas.guardar_informe("baterias", "# Baterías de sodio\n\nTexto sobre litio.")
    herramientas.guardar_informe("cafe", "# Precio del café\n\nTexto.")

    assert "baterias" in herramientas.listar_informes() and "cafe" in herramientas.listar_informes()
    filtrado = herramientas.listar_informes("litio")
    assert "Baterías de sodio" in filtrado and "cafe" not in filtrado
    assert herramientas.leer_informe("cafe").startswith("# Precio del café")
    with pytest.raises(ErrorHerramienta):
        herramientas.leer_informe("no-existe")


# --- ejecutar ---


@pytest.mark.parametrize(
    "nombre, entrada",
    [
        ("leer_pagina", {}),
        ("leer_pagina", {"url": PUBLICA, "inicio": -1}),
        ("leer_pagina", {"url": PUBLICA, "inicio": "0"}),
        ("leer_pagina", {"url": "http://127.0.0.1/"}),
        ("consultar_fuente", {"url": PUBLICA}),
        ("guardar_informe", {"nombre": "x", "contenido": "   "}),
        ("guardar_informe", "no es un objeto"),
        ("inventada", {}),
    ],
)
def test_ejecutar_convierte_fallos_en_texto(nombre, entrada):
    texto, es_error = herramientas.ejecutar(nombre, entrada)
    assert es_error and texto.startswith("Error:")

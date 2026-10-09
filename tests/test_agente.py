"""Pruebas del bucle del agente con clientes falsos. No usan la red ni la API.

Las pruebas del bucle corren con los dos proveedores: el mismo guion de
respuestas se expresa en el formato de cada API.
"""

import json
import threading
from types import SimpleNamespace as NS

import pytest

import agente
import proveedores
import uso


# --- Cliente falso de Anthropic ---


def a_respuesta(stop_reason, *bloques):
    return NS(
        content=list(bloques),
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=NS(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def a_texto(t):
    return NS(type="text", text=t)


def a_llamada(id_, nombre, entrada):
    return NS(type="tool_use", id=id_, name=nombre, input=entrada)


class AnthropicFalso:
    """Devuelve respuestas preparadas, en orden, y anota cada petición recibida."""

    def __init__(self, *respuestas):
        self._respuestas = list(respuestas)
        self.peticiones = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        self.peticiones.append({**kwargs, "messages": list(kwargs["messages"])})
        final = self._respuestas.pop(0)

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return iter(())

            def get_final_message(self):
                return final

        return Stream()


# --- Cliente falso de Azure OpenAI ---


def z_respuesta(motivo, texto=None, llamadas=()):
    """Una respuesta en streaming: el texto y los argumentos llegan partidos en dos trozos."""
    trozos = [NS(choices=[], usage=None)]  # Azure empieza con un trozo sin opciones
    if texto:
        mitad = len(texto) // 2
        for parte in (texto[:mitad], texto[mitad:]):
            trozos.append(NS(usage=None, choices=[NS(finish_reason=None, delta=NS(content=parte, tool_calls=None))]))
    for indice, (id_, nombre, entrada) in enumerate(llamadas):
        argumentos = json.dumps(entrada)
        mitad = len(argumentos) // 2
        primero = NS(index=indice, id=id_, function=NS(name=nombre, arguments=argumentos[:mitad]))
        segundo = NS(index=indice, id=None, function=NS(name=None, arguments=argumentos[mitad:]))
        for parte in (primero, segundo):
            trozos.append(NS(usage=None, choices=[NS(finish_reason=None, delta=NS(content=None, tool_calls=[parte]))]))
    trozos.append(NS(usage=None, choices=[NS(finish_reason=motivo, delta=NS(content=None, tool_calls=None))]))
    detalles = NS(cached_tokens=200)
    trozos.append(NS(choices=[], usage=NS(prompt_tokens=1000, completion_tokens=100, prompt_tokens_details=detalles)))
    return trozos


class AzureFalso:
    def __init__(self, *respuestas):
        self._respuestas = list(respuestas)
        self.peticiones = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.peticiones.append({**kwargs, "messages": list(kwargs["messages"])})
        return iter(self._respuestas.pop(0))


# --- El mismo guion para los dos proveedores ---


class Guion:
    """Construye un proveedor con respuestas preparadas, en el formato de su API."""

    def __init__(self, nombre):
        self.nombre = nombre

    def texto(self, t):
        return a_respuesta("end_turn", a_texto(t)) if self.nombre == "anthropic" else z_respuesta("stop", texto=t)

    def llamadas(self, *llamadas):
        if self.nombre == "anthropic":
            return a_respuesta("tool_use", *(a_llamada(*ll) for ll in llamadas))
        return z_respuesta("tool_calls", llamadas=llamadas)

    def cortada(self):
        if self.nombre == "anthropic":
            return a_respuesta("max_tokens", a_llamada("id1", "guardar_informe", {"nombre": "a"}))
        return z_respuesta("length", llamadas=[("id1", "guardar_informe", {"nombre": "a"})])

    def proveedor(self, *respuestas):
        if self.nombre == "anthropic":
            return proveedores.ProveedorAnthropic(cliente=AnthropicFalso(*respuestas))
        return proveedores.ProveedorAzure(modelo="mi-despliegue", cliente=AzureFalso(*respuestas))


@pytest.fixture(params=["anthropic", "azure"])
def guion(request):
    return Guion(request.param)


@pytest.fixture(autouse=True)
def uso_limpio(monkeypatch):
    contador = uso.Uso()
    monkeypatch.setattr(agente, "USO", contador)
    monkeypatch.setattr(proveedores, "USO", contador)
    return contador


def correr(proveedor, herramientas=(), max_pasos=5):
    mensajes = [{"role": "user", "content": "pregunta"}]
    final = agente.ejecutar_bucle(proveedor, "sistema", list(herramientas), mensajes, agente.Config(), max_pasos)
    return final, mensajes


def test_termina_y_devuelve_el_texto(guion):
    final, mensajes = correr(guion.proveedor(guion.texto("todo listo")))
    assert final == "todo listo"
    assert [m["role"] for m in mensajes] == ["user", "assistant"]


def test_ejecuta_herramientas_en_paralelo_y_conserva_el_orden(guion, monkeypatch):
    # La barrera solo se abre si las tres llamadas están en curso a la vez.
    barrera = threading.Barrier(3, timeout=5)

    def falsa(nombre, entrada, proveedor=None):
        barrera.wait()
        return f"resultado {entrada['n']}", entrada["n"] == 2

    monkeypatch.setattr(agente, "ejecutar", falsa)
    proveedor = guion.proveedor(
        guion.llamadas(*((f"id{n}", "leer_pagina", {"n": n}) for n in (1, 2, 3))),
        guion.texto("fin"),
    )
    final, mensajes = correr(proveedor)

    assert final == "fin"
    if guion.nombre == "anthropic":
        resultados = mensajes[2]["content"]  # un único mensaje con los tres resultados
        assert [r["tool_use_id"] for r in resultados] == ["id1", "id2", "id3"]
        assert [r["content"] for r in resultados] == ["resultado 1", "resultado 2", "resultado 3"]
        assert [r["is_error"] for r in resultados] == [False, True, False]
    else:
        assert [c["function"]["name"] for c in mensajes[1]["tool_calls"]] == ["leer_pagina"] * 3
        assert json.loads(mensajes[1]["tool_calls"][2]["function"]["arguments"]) == {"n": 3}
        assert mensajes[2:5] == [
            {"role": "tool", "tool_call_id": f"id{n}", "content": f"resultado {n}"} for n in (1, 2, 3)
        ]


def test_respuesta_cortada_no_ejecuta_nada(guion, monkeypatch):
    monkeypatch.setattr(agente, "ejecutar", lambda *a, **k: pytest.fail("no debe ejecutarse"))
    proveedor = guion.proveedor(guion.cortada(), guion.texto("fin"))
    final, mensajes = correr(proveedor)

    if guion.nombre == "anthropic":
        # Se repite una vez con más margen, sin guardar la respuesta cortada.
        peticiones = proveedor.cliente.peticiones
        assert final == "fin"
        assert [p["max_tokens"] for p in peticiones] == [proveedores.MAX_TOKENS, proveedores.MAX_TOKENS_TOPE]
        assert peticiones[1]["messages"] == peticiones[0]["messages"]
        assert len(mensajes) == 2
    else:
        # En Azure el tope de salida es el del despliegue: el agente se detiene.
        assert final is None
        assert len(mensajes) == 1


def test_tope_de_pasos_devuelve_none(guion, monkeypatch):
    monkeypatch.setattr(agente, "ejecutar", lambda *a, **k: ("ok", False))
    sin_fin = guion.proveedor(*(guion.llamadas((f"id{n}", "leer_pagina", {})) for n in range(3)))
    assert correr(sin_fin, max_pasos=3)[0] is None


def test_delegar_corre_un_investigador_con_sus_propias_herramientas(guion):
    proveedor = guion.proveedor(
        guion.llamadas(("id1", "delegar", {"subpregunta": "¿cuánto cuesta X?"})),
        guion.texto("hallazgo: X cuesta 3"),  # el investigador
        guion.texto("informe"),
    )
    final, _ = correr(proveedor, herramientas=agente.HERRAMIENTAS + [agente.DELEGAR])

    assert final == "informe"
    principal, investigador, cierre = proveedor.cliente.peticiones
    if guion.nombre == "anthropic":
        assert investigador["system"] == agente.SISTEMA_INVESTIGADOR
        assert "¿cuánto cuesta X?" in investigador["messages"][0]["content"]
        nombres = {h["name"] for h in investigador["tools"]}
        assert cierre["messages"][-1]["content"][0]["content"] == "hallazgo: X cuesta 3"
    else:
        assert investigador["messages"][0] == {"role": "system", "content": agente.SISTEMA_INVESTIGADOR}
        assert "¿cuánto cuesta X?" in investigador["messages"][1]["content"]
        nombres = {h["function"]["name"] for h in investigador["tools"]}
        assert cierre["messages"][-1]["content"] == "hallazgo: X cuesta 3"
    assert "guardar_informe" not in nombres and "delegar" not in nombres
    assert ("web_search" if guion.nombre == "anthropic" else "buscar_web") in nombres


# --- Particularidades de cada proveedor ---


def test_anthropic_pause_turn_continua_sin_mensaje_de_usuario():
    proveedor = proveedores.ProveedorAnthropic(
        cliente=AnthropicFalso(a_respuesta("pause_turn", a_texto("buscando")), a_respuesta("end_turn", a_texto("fin")))
    )
    final, mensajes = correr(proveedor)
    assert final == "fin"
    assert [m["role"] for m in mensajes] == ["user", "assistant", "assistant"]


def test_anthropic_rechazo_devuelve_none():
    proveedor = proveedores.ProveedorAnthropic(cliente=AnthropicFalso(a_respuesta("refusal")))
    assert correr(proveedor)[0] is None


def test_anthropic_marca_las_herramientas_y_anade_la_busqueda():
    proveedor = proveedores.ProveedorAnthropic(cliente=AnthropicFalso(a_respuesta("end_turn", a_texto("fin"))))
    correr(proveedor, herramientas=agente.HERRAMIENTAS)
    enviadas = proveedor.cliente.peticiones[0]["tools"]
    assert enviadas[0]["type"] == "web_search_20260209"
    assert all(h["eager_input_streaming"] for h in enviadas[1:])


def test_azure_formato_de_herramientas_esfuerzo_y_uso(uso_limpio):
    cliente = AzureFalso(z_respuesta("stop", texto="fin"), z_respuesta("stop", texto="fin"))
    correr(proveedores.ProveedorAzure(modelo="mi-despliegue", cliente=cliente), herramientas=agente.HERRAMIENTAS)
    correr(proveedores.ProveedorAzure(modelo="mi-despliegue", esfuerzo="max", cliente=cliente))

    sin_esfuerzo, con_esfuerzo = cliente.peticiones
    assert sin_esfuerzo["model"] == "mi-despliegue"
    assert "reasoning_effort" not in sin_esfuerzo and con_esfuerzo["reasoning_effort"] == "high"
    funcion = sin_esfuerzo["tools"][0]
    assert funcion["type"] == "function" and set(funcion["function"]) == {"name", "description", "parameters"}
    # 1000 tokens de entrada, de los que 200 venían de caché, por cada una de las dos peticiones.
    assert uso_limpio._por_modelo["mi-despliegue"] == {
        "entrada": 1600, "salida": 200, "cache_lectura": 400, "cache_escritura": 0
    }


def test_azure_argumentos_rotos_llegan_al_modelo_como_error():
    rota = z_respuesta("tool_calls", llamadas=[("id1", "leer_pagina", {})])
    rota[1].choices[0].delta.tool_calls[0].function.arguments = '{"url": '
    rota[2].choices[0].delta.tool_calls[0].function.arguments = ""
    final, mensajes = correr(
        proveedores.ProveedorAzure(modelo="d", cliente=AzureFalso(rota, z_respuesta("stop", texto="fin")))
    )
    assert final == "fin"
    assert mensajes[2]["content"].startswith("Error:")


def test_lector_usa_el_despliegue_barato(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_LECTOR", "mini")
    peticiones = []

    def create(**kwargs):
        peticiones.append(kwargs)
        mensaje = NS(content='RESPALDADA: "cita"')
        return NS(choices=[NS(message=mensaje)], usage=NS(prompt_tokens=10, completion_tokens=5))

    proveedor = proveedores.ProveedorAzure(modelo="grande", cliente=NS(chat=NS(completions=NS(create=create))))
    assert proveedor.completar("sistema", "documento") == 'RESPALDADA: "cita"'
    assert peticiones[0]["model"] == "mini"


# --- Configuración y costo ---


def test_proveedor_por_defecto(monkeypatch):
    for variable in ("AGENTE_PROVEEDOR", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(variable, raising=False)
    assert proveedores.proveedor_por_defecto() == "anthropic"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    assert proveedores.proveedor_por_defecto() == "azure"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave")
    assert proveedores.proveedor_por_defecto() == "anthropic"
    monkeypatch.setenv("AGENTE_PROVEEDOR", "azure")
    assert proveedores.proveedor_por_defecto() == "azure"


def test_crear_azure_avisa_de_lo_que_falta(monkeypatch):
    for variable in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit, match="AZURE_OPENAI_ENDPOINT.*AZURE_OPENAI_API_KEY.*AZURE_OPENAI_DEPLOYMENT"):
        proveedores.crear("azure", None, None)


def test_costo():
    contador = uso.Uso()
    contador.registrar(
        "claude-opus-5-5",
        entrada=1_000_000, salida=1_000_000, cache_lectura=1_000_000, cache_escritura=1_000_000, busquedas=10,
    )
    contador.registrar("mi-despliegue", entrada=5, salida=5)
    total, desconocidos = contador.costo()
    assert total == pytest.approx(4.00 + 20.00 + 0.20 + 5.00 + 0.10)
    assert desconocidos == ["mi-despliegue"]
    assert "sin precio conocido para mi-despliegue" in contador.resumen()

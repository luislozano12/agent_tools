# Agente investigador

Busca en la web, lee las fuentes (páginas y PDF), verifica las citas y escribe un
informe en Markdown en la carpeta `informes/`.

## Instalación

```powershell
pip install -r requirements.txt
$env:AGENTE_CONTACTO = "tu-correo-o-url"   # opcional; Wikipedia lo exige
```

Funciona con dos proveedores. Configura uno, en tu terminal como abajo o en un
archivo `.env` (copia `.env.example` y rellénalo; se carga solo al arrancar):

**Anthropic (Claude)**

```powershell
$env:ANTHROPIC_API_KEY = "tu-clave"
```

**Azure OpenAI**

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://TU-RECURSO.openai.azure.com"
$env:AZURE_OPENAI_API_KEY = "tu-clave"
$env:AZURE_OPENAI_DEPLOYMENT = "nombre-de-tu-despliegue"
$env:AZURE_OPENAI_DEPLOYMENT_LECTOR = "despliegue-barato"   # opcional; lee las fuentes
$env:OPENAI_API_VERSION = "2024-10-21"                      # opcional
```

Si solo están las variables de Azure, se usa Azure. Si están las dos, se usa
Anthropic salvo que pases `--proveedor azure` o definas `AGENTE_PROVEEDOR=azure`.

Diferencias con Azure OpenAI:

- La búsqueda web se hace desde tu máquina con DuckDuckGo (paquete `ddgs`, sin clave),
  porque Azure OpenAI no incluye buscador.
- `--esfuerzo` solo se envía si lo pasas, y solo lo aceptan los modelos de
  razonamiento; puede requerir una `OPENAI_API_VERSION` más reciente.
- El costo no se calcula: depende del modelo que haya detrás de cada despliegue.
- La autenticación es por clave; Entra ID no está configurado.

## Uso

```powershell
python agente.py "¿Qué tan cerca están las baterías de sodio de reemplazar a las de litio?"
python agente.py --profundo --esfuerzo high "tu pregunta"
```

| Opción | Qué hace |
|---|---|
| `--esfuerzo` | `low`, `medium` (por defecto), `high`, `xhigh`, `max`. Más esfuerzo, mejor y más caro. |
| `--profundo` | Divide la pregunta en subpreguntas y las delega a investigadores en paralelo. |
| `--proveedor` | `anthropic` o `azure`. |
| `--modelo` | Modelo de Anthropic (por defecto `claude-opus-5-5`) o despliegue de Azure. |
| `--max-pasos` | Tope de pasos por turno. Por defecto 25. |
| `--sin-chat` | Sale al terminar, sin ofrecer seguimiento. |

Al terminar, en una terminal interactiva, puedes pedir cambios ("profundiza en la
sección 2") y el agente actualiza el mismo informe. Al salir se muestra el uso de
tokens y el costo aproximado.

## Cómo está armado

- `agente.py`: el bucle del agente, los prompts, el modo profundo y la línea de comandos.
- `proveedores.py`: traduce entre el bucle y la API de cada proveedor (Anthropic, Azure OpenAI).
- `herramientas.py`: definición y código de las herramientas (descarga, PDF, búsqueda, informes).
- `lector.py`: un modelo barato (`claude-haiku-5-5`, o tu despliegue lector en Azure) lee
  cada fuente completa y devuelve solo lo relevante, y comprueba si una fuente respalda
  una afirmación.
- `uso.py`: contador de tokens y costo.

## Pruebas

```powershell
pip install pytest
python -m pytest
```

Las pruebas no usan la red ni la API.

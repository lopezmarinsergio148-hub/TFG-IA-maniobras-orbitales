# ═══════════════════════════════════════════════════════════════════════════
#  ia/llm_orquestador.py — Capa LLM (Bloque 5): orquestador + explicador
#
#  Patrón TOOL USE (function calling). El LLM (Llama 3.3 en Groq) NO calcula nada:
#    1) lee la petición en lenguaje natural,
#    2) ELIGE la herramienta adecuada y sus argumentos,
#    3) NUESTRO código ejecuta el agente/solver REAL (llm_herramientas.py),
#    4) el LLM EXPLICA los números reales que ha devuelto la herramienta.
#
#  El bucle es MANUAL (no automático): así se ve cada paso, es más didáctico y
#  defendible. La regla de oro (no inventar cifras) se impone en el system prompt.
#
#  La API key se lee de la variable de entorno GROQ_API_KEY (NO va en el código).
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 LLM_ORQUESTADOR — Capa LLM del TFG (Bloque 5): orquestador + explicador
 Asistente conversacional que traduce peticiones en lenguaje natural a llamadas de
 herramienta (tool use manual, multi-ronda, con memoria) contra el modelo Llama 3.3
 servido por Groq. Expone el catálogo de 13 tools (5 de cálculo + 8 de dibujo), su
 mapa de despacho a funciones Python y el system prompt acotado al dominio.

 ÍNDICE DE FUNCIONES:
   - _envolver(texto, ancho)              : envuelve el texto de respuesta a un ancho fijo para leerlo mejor.
   - responder(mensajes, verbose, max_rondas) : bucle de tool use multi-ronda; amplía el historial y devuelve la respuesta.
   - chat()                               : REPL interactivo con memoria conversacional.

 OBJETOS DE MÓDULO:
   - MODELO    : nombre del modelo de Groq en uso.
   - TOOLS     : schemas de las 13 herramientas que lee el LLM (NO tocar).
   - DISPATCH  : mapa nombre-de-herramienta -> función real de Python.
   - SYSTEM    : system prompt que impone el dominio y la regla de no inventar cifras (NO tocar).
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import textwrap

from openai import OpenAI, RateLimitError, APIError

from llm_herramientas import (planificar_transferencia, planificar_aerofrenado,
                              planificar_cambio_plano, planificar_mantenimiento,
                              transferencia_interplanetaria)
from llm_figuras import (dibujar_transferencia, dibujar_aerofrenado,
                         dibujar_aerofrenado_orbitas, dibujar_interplanetaria,
                         dibujar_interplanetaria_3d, dibujar_cambio_plano_3d,
                         dibujar_mantenimiento, dibujar_porkchop)

sys.stdout.reconfigure(encoding="utf-8")        # evita errores de acentos en Windows

# Modelo OPEN servido por Groq (gratis). El 8B gasta mucho menos y tiene cupo aparte ->
# para trastear cuando el 70B se queda sin tokens del dia. Para la PRUEBA FINAL (que elija
# bien las herramientas) volver al 70B: basta comentar esta linea y descomentar la de abajo.
#MODELO = "llama-3.1-8b-instant"                 # 8B: barato, tool use algo mas flojo
MODELO = "llama-3.3-70b-versatile"            # 70B: mejor tool use (usar para la prueba final)

# Cliente: Groq habla la MISMA API que OpenAI, solo cambia la base_url
_clave = os.environ.get("GROQ_API_KEY")
if not _clave:
    raise SystemExit("Falta la variable de entorno GROQ_API_KEY (ver instrucciones).")
cliente = OpenAI(api_key=_clave, base_url="https://api.groq.com/openai/v1")

# ── Catálogo de herramientas que el LLM puede invocar (por ahora, una) ──────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "planificar_transferencia",
            "description": ("Planifica una transferencia orbital coplanar (sin cambio de "
                            "plano) entre dos órbitas circulares de altitudes h1 y h2 (km) "
                            "alrededor de un planeta. Devuelve el Delta-v del agente de IA y "
                            "el óptimo de Hohmann. Úsala para subir o bajar de órbita."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "tierra, marte, venus, jupiter, saturno, urano, neptuno, luna o mercurio"},
                    "h1_km": {"type": "number", "description": "altitud de la órbita inicial en km"},
                    "h2_km": {"type": "number", "description": "altitud de la órbita final en km"},
                },
                "required": ["planeta", "h1_km", "h2_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planificar_aerofrenado",
            "description": ("Planifica un AEROFRENADO (aerobraking): BAJAR el apogeo de una "
                            "órbita muy elíptica usando el rozamiento atmosférico en cada "
                            "paso por el perigeo, casi sin gastar combustible. Úsala cuando "
                            "el usuario quiera frenar/bajar el apogeo aprovechando la "
                            "atmósfera. Devuelve perigeo de operación, número de pasadas y "
                            "duración. Solo en planetas con atmósfera."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "cuerpo CON atmósfera: tierra, marte, venus, jupiter, saturno, urano o neptuno (la Luna y Mercurio NO valen: no tienen atmósfera)"},
                    "apo_ini_km": {"type": "number", "description": "altitud del apogeo inicial en km"},
                    "apo_obj_km": {"type": "number", "description": "altitud del apogeo objetivo en km (menor)"},
                },
                "required": ["planeta", "apo_ini_km", "apo_obj_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_transferencia",
            "description": ("Genera una IMAGEN de una transferencia entre dos órbitas "
                            "circulares (altitudes h1, h2 en km) alrededor de UN MISMO "
                            "planeta. Úsala cuando el usuario pida ver/dibujar una maniobra "
                            "de cambio de órbita en un cuerpo. NO la uses para viajes entre "
                            "dos planetas (para eso usa dibujar_interplanetaria)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "tierra, marte, venus, jupiter, saturno, urano, neptuno, luna o mercurio"},
                    "h1_km": {"type": "number", "description": "altitud de la órbita inicial en km"},
                    "h2_km": {"type": "number", "description": "altitud de la órbita final en km"},
                },
                "required": ["planeta", "h1_km", "h2_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_aerofrenado",
            "description": ("Genera la CURVA de un aerofrenado: cómo baja la altitud del "
                            "apogeo pasada a pasada (gráfico apogeo vs nº de pasada). Úsala si "
                            "el usuario pide ver la evolución/curva del aerofrenado. Si pide "
                            "ver cómo CAMBIA o se ENCOGE la ÓRBITA, usa dibujar_aerofrenado_orbitas."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "cuerpo CON atmósfera: tierra, marte, venus, jupiter, saturno, urano o neptuno (la Luna y Mercurio NO valen: no tienen atmósfera)"},
                    "apo_ini_km": {"type": "number", "description": "altitud del apogeo inicial en km"},
                    "apo_obj_km": {"type": "number", "description": "altitud del apogeo objetivo en km (menor)"},
                },
                "required": ["planeta", "apo_ini_km", "apo_obj_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transferencia_interplanetaria",
            "description": ("Calcula un viaje INTERPLANETARIO entre dos PLANETAS (ej. Tierra "
                            "a Marte) en una fecha de salida dada, resolviendo el problema de "
                            "Lambert con las posiciones reales de los planetas. Úsala para "
                            "viajes de un planeta a otro. Devuelve los Δv de salida y llegada, "
                            "el total y la energía característica C3."),
            "parameters": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string",
                               "description": "planeta de salida: mercurio, venus, tierra, marte, jupiter, saturno, urano o neptuno"},
                    "destino": {"type": "string", "description": "planeta de destino"},
                    "fecha_salida": {"type": "string",
                                     "description": "fecha de salida en formato AAAA-MM-DD"},
                    "dias_vuelo": {"type": "number",
                                   "description": "tiempo de vuelo en días (OPCIONAL; si no se indica, se usa el tiempo de una transferencia de Hohmann, que es lo recomendable salvo que el usuario dé un valor)"},
                },
                "required": ["origen", "destino", "fecha_salida"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_interplanetaria",
            "description": ("Genera una IMAGEN 2D (PNG estático) de un viaje INTERPLANETARIO: "
                            "el Sol, las órbitas de los dos planetas y la trayectoria. Úsala "
                            "para ver/dibujar un viaje ENTRE DOS PLANETAS (ej. Marte a Júpiter). "
                            "Si el usuario pide verlo en 3D o de forma INTERACTIVA, usa en su "
                            "lugar dibujar_interplanetaria_3d."),
            "parameters": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string",
                               "description": "planeta de salida: mercurio, venus, tierra, marte, jupiter, saturno, urano o neptuno"},
                    "destino": {"type": "string", "description": "planeta de destino"},
                    "fecha_salida": {"type": "string",
                                     "description": "fecha de salida en formato AAAA-MM-DD"},
                    "dias_vuelo": {"type": "number",
                                   "description": "tiempo de vuelo en días (OPCIONAL; si no se indica, se usa el tiempo de una transferencia de Hohmann, que es lo recomendable salvo que el usuario dé un valor)"},
                },
                "required": ["origen", "destino", "fecha_salida"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_interplanetaria_3d",
            "description": ("Genera una visualización INTERACTIVA EN 3D (archivo HTML que se "
                            "abre en el navegador y se puede ROTAR y hacer ZOOM) de un viaje "
                            "interplanetario, con las órbitas en 3D y su inclinación real. "
                            "Úsala cuando el usuario pida ver un viaje entre planetas en 3D o "
                            "de forma interactiva."),
            "parameters": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string",
                               "description": "planeta de salida: mercurio, venus, tierra, marte, jupiter, saturno, urano o neptuno"},
                    "destino": {"type": "string", "description": "planeta de destino"},
                    "fecha_salida": {"type": "string",
                                     "description": "fecha de salida en formato AAAA-MM-DD"},
                    "dias_vuelo": {"type": "number",
                                   "description": "tiempo de vuelo en días (OPCIONAL; si no se indica, se usa el tiempo de una transferencia de Hohmann, que es lo recomendable salvo que el usuario dé un valor)"},
                },
                "required": ["origen", "destino", "fecha_salida"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planificar_cambio_plano",
            "description": ("Planifica una transferencia con CAMBIO DE PLANO: subir de una "
                            "órbita a otra MÁS ALTA cambiando además la inclinación. Úsala "
                            "cuando el usuario quiera cambiar el plano/inclinación al subir de "
                            "órbita (ej. inyección a GEO desde una órbita inclinada). Solo "
                            "SUBE; cambio de plano hasta 40 grados. Devuelve el Δv del agente, "
                            "el óptimo y cómo reparte el giro entre perigeo y apogeo."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "tierra, marte, venus, jupiter, saturno, urano, neptuno, luna o mercurio"},
                    "h1_km": {"type": "number", "description": "altitud de la órbita inicial en km"},
                    "h2_km": {"type": "number", "description": "altitud de la órbita final en km (mayor que h1)"},
                    "inclinacion_grados": {"type": "number",
                                           "description": "cambio de plano (grados a girar la inclinación), de 0 a 40"},
                },
                "required": ["planeta", "h1_km", "h2_km", "inclinacion_grados"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_cambio_plano_3d",
            "description": ("Genera una visualización INTERACTIVA EN 3D (HTML que se abre en "
                            "el navegador, rotable) de una transferencia con CAMBIO DE PLANO "
                            "(subir de órbita cambiando la inclinación). Úsala cuando el "
                            "usuario pida ver/dibujar un cambio de plano o una transferencia "
                            "inclinada; la inclinación solo se aprecia en 3D."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "tierra, marte, venus, jupiter, saturno, urano, neptuno, luna o mercurio"},
                    "h1_km": {"type": "number", "description": "altitud de la órbita inicial en km"},
                    "h2_km": {"type": "number", "description": "altitud de la órbita final en km (mayor que h1)"},
                    "inclinacion_grados": {"type": "number",
                                           "description": "cambio de plano en grados, de 0 a 40"},
                },
                "required": ["planeta", "h1_km", "h2_km", "inclinacion_grados"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_porkchop",
            "description": ("Genera un PORKCHOP: el mapa del Δv de un viaje interplanetario "
                            "según la fecha de salida y el tiempo de vuelo, que revela la "
                            "VENTANA DE LANZAMIENTO óptima (el mínimo Δv) alrededor de una "
                            "fecha. Úsala cuando el usuario pregunte por la mejor fecha o "
                            "ventana para ir de un planeta a otro, o pida un porkchop. "
                            "Devuelve la figura y la fecha y días de vuelo óptimos."),
            "parameters": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string",
                               "description": "planeta de salida: mercurio, venus, tierra, marte, jupiter, saturno, urano o neptuno"},
                    "destino": {"type": "string", "description": "planeta de destino"},
                    "fecha_centro": {"type": "string",
                                     "description": "fecha aproximada alrededor de la cual buscar la ventana (AAAA-MM-DD)"},
                },
                "required": ["origen", "destino", "fecha_centro"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planificar_mantenimiento",
            "description": ("Planifica el MANTENIMIENTO ORBITAL (station-keeping) de una órbita "
                            "circular a una altitud dada: cuánto combustible (Δv al año) hace "
                            "falta para mantenerla contra el rozamiento atmosférico, si es "
                            "viable y cuánto precesa el plano por el achatamiento (J2). Úsala "
                            "cuando el usuario pregunte por mantener/conservar una órbita, el "
                            "coste de station-keeping o cuánto dura una órbita. Solo en cuerpos "
                            "CON atmósfera (sin aire la órbita no se degrada)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "cuerpo CON atmósfera: tierra, marte, venus, jupiter, saturno, urano o neptuno"},
                    "h_km": {"type": "number", "description": "altitud de la órbita circular a mantener, en km"},
                    "inclinacion_grados": {"type": "number",
                                           "description": "inclinación de la órbita en grados (OPCIONAL; por defecto 51.6, la de la ISS)"},
                },
                "required": ["planeta", "h_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_mantenimiento",
            "description": ("Dibuja el MANTENIMIENTO ORBITAL: la altitud frente al tiempo "
                            "durante un año, con el ciclo de caer y re-boostear, comparando el "
                            "agente (altitud estable) con la estrategia ingenua (diente de "
                            "sierra) y la banda de tolerancia. Úsala cuando el usuario pida "
                            "ver/dibujar el mantenimiento o el station-keeping de una órbita."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "cuerpo CON atmósfera: tierra, marte, venus, jupiter, saturno, urano o neptuno"},
                    "h_km": {"type": "number", "description": "altitud de la órbita a mantener, en km"},
                    "inclinacion_grados": {"type": "number",
                                           "description": "inclinación en grados (OPCIONAL; por defecto 51.6)"},
                },
                "required": ["planeta", "h_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dibujar_aerofrenado_orbitas",
            "description": ("Dibuja cómo la ÓRBITA se encoge y circulariza durante un "
                            "aerofrenado: varias órbitas superpuestas, de la elíptica inicial "
                            "a la casi circular final, con el planeta en el centro. Úsala si "
                            "el usuario pide ver cómo cambia/evoluciona la ÓRBITA en un "
                            "aerofrenado."),
            "parameters": {
                "type": "object",
                "properties": {
                    "planeta": {"type": "string",
                                "description": "cuerpo CON atmósfera: tierra, marte, venus, jupiter, saturno, urano o neptuno"},
                    "apo_ini_km": {"type": "number", "description": "altitud del apogeo inicial en km"},
                    "apo_obj_km": {"type": "number", "description": "altitud del apogeo objetivo en km (menor)"},
                },
                "required": ["planeta", "apo_ini_km", "apo_obj_km"],
            },
        },
    },
]

# Mapa nombre-de-herramienta -> función real de Python
DISPATCH = {"planificar_transferencia": planificar_transferencia,
            "planificar_aerofrenado": planificar_aerofrenado,
            "planificar_cambio_plano": planificar_cambio_plano,
            "planificar_mantenimiento": planificar_mantenimiento,
            "transferencia_interplanetaria": transferencia_interplanetaria,
            "dibujar_transferencia": dibujar_transferencia,
            "dibujar_aerofrenado": dibujar_aerofrenado,
            "dibujar_interplanetaria": dibujar_interplanetaria,
            "dibujar_interplanetaria_3d": dibujar_interplanetaria_3d,
            "dibujar_cambio_plano_3d": dibujar_cambio_plano_3d,
            "dibujar_aerofrenado_orbitas": dibujar_aerofrenado_orbitas,
            "dibujar_mantenimiento": dibujar_mantenimiento,
            "dibujar_porkchop": dibujar_porkchop}

SYSTEM = (
    "Eres un asistente ESPECIALIZADO en planificación de maniobras orbitales y astrodinámica. "
    "Ayudas con varios tipos de maniobra: transferencias entre órbitas (con o sin cambio de "
    "plano), aerofrenado, mantenimiento orbital (station-keeping) y viajes interplanetarios. "
    "Para cualquier CÁLCULO debes usar las herramientas disponibles; tú no "
    "calculas nada por tu cuenta. REGLA ABSOLUTA: usa EXCLUSIVAMENTE los números que devuelven "
    "las herramientas; nunca inventes ni estimes cifras. Si una herramienta devuelve un 'error' "
    "(p. ej. aerofrenado en un cuerpo sin atmósfera), explícaselo al usuario con naturalidad. "
    "Puedes explicar CONCEPTOS de astrodinámica (qué es una transferencia de Hohmann, el "
    "aerofrenado, el Delta-v, el C3, etc.) con tu propio conocimiento. Pero si te preguntan algo "
    "AJENO a la astrodinámica y el espacio, declina amablemente y recuerda que eres un asistente "
    "de maniobras orbitales. Explica los resultados de forma clara y breve. Responde en español."
)


def _envolver(texto, ancho=88):
    """Envuelve la respuesta a un ancho fijo (respetando los saltos de línea ya
    presentes) para que el texto largo no salga en un único renglón: así se lee mejor
    en pantalla y CABE en una captura para la memoria."""
    if not texto:
        return texto
    return "\n".join(textwrap.fill(p, width=ancho) if p.strip() else ""
                     for p in texto.split("\n"))


def responder(mensajes, verbose=True, max_rondas=5):
    """
    Procesa el último turno de la conversación con el bucle de tool use. Recibe la
    LISTA DE MENSAJES completa (system + historial + último mensaje del usuario) y la
    va AMPLIANDO con lo que ocurre en el turno (decisiones del LLM, resultados de las
    herramientas y respuesta final), de modo que la conversación tenga MEMORIA: el
    usuario puede decir "ahora lo mismo pero en Marte" y el LLM recuerda el contexto.

    Bucle de VARIAS rondas: el LLM puede encadenar herramientas (p. ej. planificar y
    luego dibujar) antes de dar la respuesta final. Devuelve el texto de esa respuesta.
    """
    try:
        for _ in range(max_rondas):
            r = cliente.chat.completions.create(model=MODELO, messages=mensajes, tools=TOOLS,
                                                tool_choice="auto", temperature=0)
            msg = r.choices[0].message
            if not msg.tool_calls:               # el LLM ya no pide herramientas: responde
                mensajes.append({"role": "assistant", "content": msg.content})
                return msg.content
            mensajes.append(msg)                  # guarda la decisión del LLM
            for tc in msg.tool_calls:             # ejecuta cada herramienta elegida
                args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"   [el LLM eligió: {tc.function.name}({args})]")
                resultado = DISPATCH[tc.function.name](**args)
                mensajes.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(resultado, ensure_ascii=False)})

        # red de seguridad si agota las rondas: una última explicación sin más herramientas
        r = cliente.chat.completions.create(model=MODELO, messages=mensajes, temperature=0)
        mensajes.append({"role": "assistant", "content": r.choices[0].message.content})
        return r.choices[0].message.content
    except RateLimitError:
        return ("[Aviso] Se ha alcanzado el limite GRATUITO de Groq (100.000 tokens al dia "
                "en el plan free). El limite se reinicia solo: espera unos minutos o vuelve "
                "a probarlo mas tarde / manana. No es un fallo de tu peticion.")
    except APIError as e:
        return (f"[Aviso] Error de la API del LLM ({type(e).__name__}). Revisa la conexion "
                "e intentalo de nuevo en un momento.")


def chat():
    """Modo CHAT con MEMORIA: escribes tus peticiones y el asistente responde,
    recordando lo anterior. 'salir' termina."""
    print("=" * 70)
    print("  ASISTENTE DE MANIOBRAS ORBITALES  (escribe 'salir' para terminar)")
    print("=" * 70)
    print("  Escribe en lenguaje natural, indicando el planeta y las dos altitudes.")
    print("  Recuerda lo anterior: puedes decir 'ahora lo mismo en Marte'.")
    print("  Ejemplos:")
    print("    - sube de 400 a 35786 km en la Tierra y enséñame la órbita")
    print("    - baja el apogeo de 6000 a 400 km en Marte aprovechando la atmósfera")
    mensajes = [{"role": "system", "content": SYSTEM}]   # el historial vive aquí
    while True:
        try:
            p = input("\n>>> Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if p.lower() in ("salir", "exit", "quit", ""):
            print("Hasta luego.")
            break
        mensajes.append({"role": "user", "content": p})
        print("\n--- Asistente:")
        print(_envolver(responder(mensajes)))


if __name__ == "__main__":
    # Sin argumentos -> modo CHAT interactivo con memoria (lo normal).
    # Con argumentos  -> responde a UNA petición suelta (sin memoria), p.ej.:
    #     python llm_orquestador.py "sube de 400 a 35786 km en la Tierra"
    if len(sys.argv) > 1:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": " ".join(sys.argv[1:])}]
        print(_envolver(responder(msgs)))
    else:
        chat()

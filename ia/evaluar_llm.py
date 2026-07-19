# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_llm.py — Evaluacion FORMAL de la capa LLM (Bloque 5)
#
#  No basta con capturas de conversacion: se evalua el orquestador sobre un
#  banco de prompts anotados, midiendo tres cosas por consulta:
#     - HERRAMIENTA CORRECTA : el LLM elige el tool adecuado (o declina si la
#                              consulta es ajena al dominio).
#     - ARGUMENTOS VALIDOS   : extrae bien planeta / altitudes / fechas.
#     - CIFRAS FIELES        : los numeros de su respuesta salen de lo que
#                              devolvio la herramienta (no inventa).
#
#  Las dos primeras son automaticas (controlamos el bucle de tool use). La
#  tercera se comprueba de forma automatica (cada numero de la respuesta debe
#  aparecer en el resultado de la herramienta o en la propia peticion) y ademas
#  se GUARDA el transcript completo de cada consulta en eval_llm_resultados.json
#  para poder revisarla a mano.
#
#  Resistente al limite gratuito de Groq: guarda el progreso tras cada consulta
#  y, si salta el RateLimitError (429), para de forma limpia. Al relanzarlo,
#  RETOMA donde se quedo (util para repartir el banco en varios dias).
#
#  Uso:   python ia/evaluar_llm.py            (corre las que falten)
#         python ia/evaluar_llm.py  --reset   (empieza de cero)
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 EVALUAR_LLM — Evaluacion FORMAL de la capa LLM (Bloque 5)

 Evalua el orquestador (mismo cliente/tools que el asistente) sobre un banco de
 prompts anotados, midiendo por consulta: HERRAMIENTA CORRECTA (elige el tool
 adecuado o declina si es ajeno al dominio), ARGUMENTOS VALIDOS (planeta,
 altitudes, fechas) y CIFRAS FIELES (los numeros de la respuesta salen de la
 herramienta o de la peticion, no inventados). Guarda el transcript de cada
 consulta en eval_llm_resultados.json y RETOMA donde se quedo si salta el limite
 gratuito de Groq (429).

 ÍNDICE DE FUNCIONES:
   - correr_consulta(texto, max_rondas) : corre una consulta por el bucle de tool use.
   - _num_variantes(tok)                : lecturas numericas de un token (coma/punto es-en).
   - _numeros(texto)                    : conjunto de floats extraidos de un texto.
   - puntuar(caso, salida)              : puntua las tres metricas de una consulta.
   - cargar() / guardar(datos)          : checkpoint del progreso en JSON.
   - resumen(datos)                     : tabla agregada por categoria.
   - main()                             : bucle principal con checkpoint y resistencia al 429.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json

# --- La API key: si no esta en el entorno, se lee del registro de Windows -----
if not os.environ.get("GROQ_API_KEY"):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["GROQ_API_KEY"] = winreg.QueryValueEx(k, "GROQ_API_KEY")[0]
    except Exception:
        pass

from openai import RateLimitError, APIError
# Reutilizamos el MISMO orquestador que usa el asistente (no duplicamos logica):
from llm_orquestador import cliente, TOOLS, DISPATCH, SYSTEM, MODELO

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "eval_llm_resultados.json")


# ───────────────────────────────────────────────────────────────────────────
#  BANCO DE PROMPTS anotados. Cada uno: categoria, texto, tool esperada (None =
#  debe DECLINAR por ser ajeno al dominio) y args clave que deberia extraer.
# ───────────────────────────────────────────────────────────────────────────
PROMPTS = [
    # --- Transferencia coplanar ---------------------------------------------
    ("transferencia", "Quiero subir un satelite de una orbita de 400 km a otra de 1500 km alrededor de la Tierra.",
     "planificar_transferencia", {"planeta": "tierra", "h1_km": 400, "h2_km": 1500}),
    ("transferencia", "Cuanto Delta-v necesito para bajar de 20000 a 500 km en Marte?",
     "planificar_transferencia", {"planeta": "marte", "h1_km": 20000, "h2_km": 500}),
    ("transferencia", "Sube de 200 a 800 km en la Luna.",
     "planificar_transferencia", {"planeta": "luna", "h1_km": 200, "h2_km": 800}),

    # --- Cambio de plano -----------------------------------------------------
    ("cambio_plano", "Sube de 400 a 35786 km en la Tierra cambiando la inclinacion 28 grados.",
     "planificar_cambio_plano", {"planeta": "tierra", "inclinacion_grados": 28}),
    ("cambio_plano", "Inyecta a una orbita de 35786 km desde una de 300 km inclinada 20 grados en la Tierra.",
     "planificar_cambio_plano", {"planeta": "tierra", "inclinacion_grados": 20}),

    # --- Aerofrenado ---------------------------------------------------------
    ("aerofrenado", "Frena el apogeo de 6000 a 400 km en Marte aprovechando la atmosfera.",
     "planificar_aerofrenado", {"planeta": "marte", "apo_ini_km": 6000, "apo_obj_km": 400}),
    ("aerofrenado", "Aerofrena en Venus de 8000 a 300 km.",
     "planificar_aerofrenado", {"planeta": "venus", "apo_ini_km": 8000, "apo_obj_km": 300}),
    ("aerofrenado", "Aerofrenado en Neptuno de 10000 a 1000 km.",
     "planificar_aerofrenado", {"planeta": "neptuno", "apo_ini_km": 10000, "apo_obj_km": 1000}),

    # --- Mantenimiento -------------------------------------------------------
    ("mantenimiento", "Cuanto combustible al ano necesito para mantener una orbita a 400 km en la Tierra?",
     "planificar_mantenimiento", {"planeta": "tierra", "h_km": 400}),
    ("mantenimiento", "Coste de station-keeping de una orbita a 300 km en Venus.",
     "planificar_mantenimiento", {"planeta": "venus", "h_km": 300}),
    ("mantenimiento", "Coste de mantener una orbita a 500 km en Saturno.",
     "planificar_mantenimiento", {"planeta": "saturno", "h_km": 500}),

    # --- Interplanetaria -----------------------------------------------------
    ("interplanetaria", "Calcula un viaje de la Tierra a Marte saliendo el 2026-10-21.",
     "transferencia_interplanetaria", {"origen": "tierra", "destino": "marte"}),
    ("interplanetaria", "Que Delta-v hace falta para ir de la Tierra a Jupiter el 2027-05-01?",
     "transferencia_interplanetaria", {"origen": "tierra", "destino": "jupiter"}),
    ("interplanetaria", "Viaje de Venus a Marte el 2028-01-15.",
     "transferencia_interplanetaria", {"origen": "venus", "destino": "marte"}),

    # --- Porkchop (ventana de lanzamiento) ----------------------------------
    ("porkchop", "Cual es la mejor ventana para ir de la Tierra a Marte alrededor de octubre de 2026?",
     "dibujar_porkchop", {"origen": "tierra", "destino": "marte"}),
    ("porkchop", "Dame un porkchop Tierra-Venus centrado en junio de 2026.",
     "dibujar_porkchop", {"origen": "tierra", "destino": "venus"}),

    # --- Dibujos -------------------------------------------------------------
    ("dibujo", "Dibujame la transferencia de 400 a 1500 km en la Tierra.",
     "dibujar_transferencia", {"planeta": "tierra"}),
    ("dibujo", "Ensename la curva del aerofrenado en Marte de 6000 a 400 km.",
     "dibujar_aerofrenado", {"planeta": "marte"}),
    ("dibujo", "Muestrame como se encoge la orbita en un aerofrenado en Marte de 6000 a 400 km.",
     "dibujar_aerofrenado_orbitas", {"planeta": "marte"}),
    ("dibujo", "Representame el viaje de la Tierra a Marte el 2026-10-21.",
     "dibujar_interplanetaria", {"origen": "tierra", "destino": "marte"}),
    ("dibujo", "Ensename en 3D e interactivo el viaje de Marte a Jupiter el 2027-01-01.",
     "dibujar_interplanetaria_3d", {"origen": "marte", "destino": "jupiter"}),
    ("dibujo", "Dibuja en 3D un cambio de plano de 400 a 35786 km con 28 grados en la Tierra.",
     "dibujar_cambio_plano_3d", {"planeta": "tierra"}),
    ("dibujo", "Dibuja el mantenimiento de una orbita a 300 km en Venus.",
     "dibujar_mantenimiento", {"planeta": "venus"}),

    # --- Fuera de dominio (debe DECLINAR, sin llamar herramientas) -----------
    ("fuera_dominio", "Cual es la capital de Francia?", None, None),
    ("fuera_dominio", "Escribeme un poema sobre el amor.", None, None),
    ("fuera_dominio", "Que tiempo hace hoy en Madrid?", None, None),
    ("fuera_dominio", "Dame una receta de tortilla de patatas.", None, None),
]


# ───────────────────────────────────────────────────────────────────────────
#  Motor: corre UNA consulta por el bucle de tool use, registrando todo.
# ───────────────────────────────────────────────────────────────────────────
def correr_consulta(texto, max_rondas=5):
    """Corre una consulta por el bucle de tool use; devuelve respuesta final, tools y resultados."""
    mensajes = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}]
    tools_llamadas, resultados = [], []
    for _ in range(max_rondas):
        r = cliente.chat.completions.create(model=MODELO, messages=mensajes, tools=TOOLS,
                                            tool_choice="auto", temperature=0)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return {"final": msg.content or "", "tools": tools_llamadas, "resultados": resultados}
        mensajes.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tools_llamadas.append({"name": tc.function.name, "args": args})
            # En los tools de dibujo forzamos abrir=False para no abrir el visor de
            # imagenes durante la evaluacion (se generan igual, pero sin lanzar ventanas).
            kwargs = dict(args)
            if tc.function.name.startswith("dibujar"):
                kwargs["abrir"] = False
            try:
                res = DISPATCH[tc.function.name](**kwargs)
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {e}"}
            resultados.append(res)
            mensajes.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(res, ensure_ascii=False, default=str)})
    r = cliente.chat.completions.create(model=MODELO, messages=mensajes, temperature=0)
    return {"final": r.choices[0].message.content or "", "tools": tools_llamadas,
            "resultados": resultados}


# ───────────────────────────────────────────────────────────────────────────
#  Puntuacion de las tres metricas.
# ───────────────────────────────────────────────────────────────────────────
def _num_variantes(tok):
    """Interpretaciones numericas de un token (maneja coma/punto es/en)."""
    cands = set()
    for t in {tok, tok.replace(".", "").replace(",", "."), tok.replace(",", ""),
              tok.replace(",", ".")}:
        try:
            cands.add(float(t))
        except ValueError:
            pass
    return cands


def _numeros(texto):
    """Extrae numeros de un texto como conjunto de floats (todas las lecturas)."""
    out = set()
    for tok in re.findall(r"-?\d[\d.,]*\d|\d", str(texto)):
        out |= _num_variantes(tok)
    return out


def puntuar(caso, salida):
    """Puntua una consulta: herramienta correcta, argumentos validos y cifras fieles."""
    categoria, texto, tool_esp, args_esp = caso
    tools = salida["tools"]
    nombres = [t["name"] for t in tools]

    # 1) HERRAMIENTA CORRECTA
    if tool_esp is None:                     # debia declinar
        herr_ok = (len(tools) == 0)
    else:
        herr_ok = (tool_esp in nombres)

    # 2) ARGUMENTOS VALIDOS (solo si eligio la herramienta esperada y hay que comprobar)
    args_ok = None
    if tool_esp is not None and herr_ok and args_esp:
        args_reales = next(t["args"] for t in tools if t["name"] == tool_esp)
        args_ok = True
        for k, v in args_esp.items():
            real = args_reales.get(k)
            if real is None:
                args_ok = False; break
            if isinstance(v, str):
                if str(real).strip().lower() != v.lower():
                    args_ok = False; break
            else:                            # numerico, con tolerancia
                try:
                    if abs(float(real) - float(v)) > max(1.0, 0.02 * abs(v)):
                        args_ok = False; break
                except (TypeError, ValueError):
                    args_ok = False; break

    # 3) CIFRAS FIELES: cada numero de la respuesta debe estar respaldado por el
    #    resultado de la herramienta o por la propia peticion. (heuristico +
    #    transcript guardado para revision manual)
    if tool_esp is None:
        cifras_ok = None                     # no aplica (no hay herramienta)
    else:
        respaldo = _numeros(texto)
        for res in salida["resultados"]:
            respaldo |= _numeros(json.dumps(res, ensure_ascii=False, default=str))
        no_resp = []
        for tok in re.findall(r"-?\d[\d.,]*\d|\d", salida["final"]):
            cands = _num_variantes(tok)
            if not any(any(abs(c - r) <= max(0.5, 0.02 * abs(r)) for r in respaldo)
                       for c in cands):
                no_resp.append(tok)
        cifras_ok = (len(no_resp) == 0)
        salida["numeros_sin_respaldo"] = no_resp

    return {"herramienta_correcta": herr_ok, "argumentos_validos": args_ok,
            "cifras_fieles": cifras_ok, "tools_llamadas": nombres}


# ───────────────────────────────────────────────────────────────────────────
#  Bucle principal con checkpoint y resistencia al 429.
# ───────────────────────────────────────────────────────────────────────────
def cargar():
    """Carga el progreso guardado (dict de resultados) del JSON, o {} si no existe."""
    if os.path.exists(SALIDA):
        with open(SALIDA, encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar(datos):
    """Vuelca el dict de resultados al JSON de salida (checkpoint)."""
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def resumen(datos):
    """Imprime la tabla agregada por categoria (herramienta, args y cifras) sobre lo evaluado."""
    cats = {}
    for r in datos.values():
        c = r["categoria"]
        cats.setdefault(c, {"n": 0, "herr": 0, "args_n": 0, "args_ok": 0, "cif_n": 0, "cif_ok": 0})
        s = cats[c]
        s["n"] += 1
        s["herr"] += int(r["herramienta_correcta"])
        if r["argumentos_validos"] is not None:
            s["args_n"] += 1; s["args_ok"] += int(r["argumentos_validos"])
        if r["cifras_fieles"] is not None:
            s["cif_n"] += 1; s["cif_ok"] += int(r["cifras_fieles"])
    print("\n" + "=" * 82)
    print(f"  EVALUACION DE LA CAPA LLM  ({MODELO})   —   {len(datos)}/{len(PROMPTS)} consultas")
    print("=" * 82)
    print(f"  {'Categoria':>16} | {'N':>3} | {'Herram. correcta':>16} | "
          f"{'Args validos':>14} | {'Cifras fieles':>14}")
    print("-" * 82)
    tot = {"n": 0, "herr": 0, "args_n": 0, "args_ok": 0, "cif_n": 0, "cif_ok": 0}
    for c, s in cats.items():
        for k in tot:
            tot[k] += s[k]
        av = f"{s['args_ok']}/{s['args_n']}" if s["args_n"] else "---"
        cf = f"{s['cif_ok']}/{s['cif_n']}" if s["cif_n"] else "---"
        print(f"  {c:>16} | {s['n']:>3} | {s['herr']:>13}/{s['n']:<2} | "
              f"{av:>14} | {cf:>14}")
    print("-" * 82)
    av = f"{tot['args_ok']}/{tot['args_n']}" if tot["args_n"] else "---"
    cf = f"{tot['cif_ok']}/{tot['cif_n']}" if tot["cif_n"] else "---"
    print(f"  {'TOTAL':>16} | {tot['n']:>3} | {tot['herr']:>13}/{tot['n']:<2} | "
          f"{av:>14} | {cf:>14}")
    print("=" * 82)
    print(f"  Transcripts completos en: {SALIDA}")


def main():
    """Recorre el banco de prompts (retomando lo pendiente), puntua cada uno e imprime el resumen."""
    if "--reset" in sys.argv and os.path.exists(SALIDA):
        os.remove(SALIDA)
    datos = cargar()
    for i, caso in enumerate(PROMPTS):
        cid = f"{i:02d}"
        if cid in datos:
            continue
        categoria, texto, tool_esp, args_esp = caso
        print(f"[{cid}] ({categoria}) {texto[:60]}...")
        try:
            salida = correr_consulta(texto)
        except RateLimitError:
            print("\n  >>> Limite gratuito de Groq alcanzado (429). Progreso guardado; "
                  "relanza el script mas tarde/manana para continuar.")
            break
        except APIError as e:
            print(f"\n  >>> Error de la API ({type(e).__name__}). Progreso guardado; reintenta.")
            break
        p = puntuar(caso, salida)
        datos[cid] = {"categoria": categoria, "prompt": texto,
                      "tool_esperada": tool_esp, "args_esperados": args_esp,
                      "final": salida["final"], "tools_llamadas": salida["tools"],
                      "numeros_sin_respaldo": salida.get("numeros_sin_respaldo", []),
                      **p}
        guardar(datos)
        marca = "OK " if p["herramienta_correcta"] else "MAL"
        print(f"       -> {marca} tool={p['tools_llamadas']}  args={p['argumentos_validos']}  "
              f"cifras={p['cifras_fieles']}")
    resumen(datos)


if __name__ == "__main__":
    main()

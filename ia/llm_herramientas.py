# ═══════════════════════════════════════════════════════════════════════════
#  ia/llm_herramientas.py — Las HERRAMIENTAS que el LLM podrá invocar (Bloque 5)
#
#  El LLM NO calcula nada: cuando recibe una petición, elige una de estas funciones
#  y nuestro código ejecuta el AGENTE REAL (ya entrenado) o un solver clásico. Cada
#  función devuelve un diccionario con DATOS REALES, que el LLM luego explica. Así
#  se cumple la regla de oro del proyecto: las cifras salen de la simulación, no se
#  inventan.
#
#  Estas funciones se prueban SOLAS (este archivo, abajo) antes de enchufarlas al
#  LLM en llm_orquestador.py.
# ═══════════════════════════════════════════════════════════════════════════

import os

import numpy as np
from astropy import units as u
from astropy.time import Time
from stable_baselines3 import PPO
from poliastro.bodies import (Sun, Earth, Mars, Venus, Jupiter, Saturn, Uranus, Neptune,
                              Moon, Mercury)
from poliastro.ephem import Ephem
from poliastro.iod import lambert

from env_transfer import TransferEnv
from env_drag import AeroBrakingEnv
from env_transfer3d import Transfer3DEnv
from baselines import delta_v_hohmann, delta_v_hohmann_plano

AQUI = os.path.dirname(os.path.abspath(__file__))

# Nombre del cuerpo (en español) -> cuerpo de poliastro. Las TRANSFERENCIAS valen
# para los 9 cuerpos (maniobra kepleriana, no necesita atmósfera) -> incluye Luna y Mercurio.
CUERPOS = {"tierra": Earth, "marte": Mars, "venus": Venus, "jupiter": Jupiter,
           "saturno": Saturn, "urano": Uranus, "neptuno": Neptune,
           "luna": Moon, "mercurio": Mercury}

# Cuerpos con agente de aerofrenado entrenado: SOLO los 7 con atmósfera (la Luna y
# Mercurio no tienen aire, así que NO se puede aerofrenar en ellos).
PLANETAS_AEROFRENADO = ["tierra", "venus", "marte", "jupiter", "saturno", "urano", "neptuno"]

# Planetas que orbitan el Sol (para las transferencias INTERPLANETARIAS; sin la Luna)
PLANETAS_SOL = {"mercurio": Mercury, "venus": Venus, "tierra": Earth, "marte": Mars,
                "jupiter": Jupiter, "saturno": Saturn, "urano": Uranus, "neptuno": Neptune}


def tof_hohmann_dias(body_o, body_d, t0):
    """
    Tiempo de vuelo (en días) de una transferencia tipo HOHMANN entre dos planetas,
    a partir de sus distancias heliocéntricas en t0. Sirve de tiempo de vuelo POR
    DEFECTO razonable: un valor fijo (p. ej. 250 días) da trayectorias absurdas para
    pares de planetas muy separados (Júpiter<->Venus), con Δv enormes y órbitas
    hiperbólicas que rompen el dibujo.
    """
    # un rango de varios puntos alrededor de t0 (la interpolación de efemérides
    # necesita suficientes puntos; con solo 2 falla el spline)
    eps = Time(np.linspace((t0 - 5.0 * u.day).jd, (t0 + 5.0 * u.day).jd, 11),
               format="jd", scale="tdb")
    r1 = float(np.linalg.norm(Ephem.from_body(body_o, eps).rv(t0)[0].to(u.km).value))
    r2 = float(np.linalg.norm(Ephem.from_body(body_d, eps).rv(t0)[0].to(u.km).value))
    a_t = 0.5 * (r1 + r2)
    mu_sol = float(Sun.k.to_value(u.km**3 / u.s**2))
    return float(np.pi * np.sqrt(a_t**3 / mu_sol) / 86400.0)


def mejor_tof_dias(body_o, body_d, t0, n=40):
    """
    Tiempo de vuelo (en días) que MINIMIZA el Δv total para la fecha de salida t0:
    barre un rango de tiempos de vuelo alrededor del de Hohmann, resuelve Lambert en
    cada uno con las posiciones REALES de los planetas y se queda con el más barato.
    Mejora al Hohmann teórico fijo (que ignora dónde está realmente el destino en esa
    fecha), dando trayectorias más limpias y económicas.
    """
    tof_h = tof_hohmann_dias(body_o, body_d, t0)
    tofs = np.linspace(0.5 * tof_h, 1.8 * tof_h, n)
    span = Time(np.linspace(t0.jd, (t0 + tofs[-1] * u.day).jd, 250), format="jd", scale="tdb")
    eo = Ephem.from_body(body_o, span)
    ed = Ephem.from_body(body_d, span)
    r0, v0 = eo.rv(t0)
    mejor_tof, mejor_dv = tof_h, np.inf
    for d in tofs:
        try:
            rf, vf = ed.rv(t0 + d * u.day)
            it = lambert(Sun.k, r0, rf, d * u.day)
            vdep, varr = it[0] if isinstance(it, list) else it
            dv = (float(np.linalg.norm((vdep - v0).to(u.km / u.s).value))
                  + float(np.linalg.norm((vf - varr).to(u.km / u.s).value)))
            if dv < mejor_dv:
                mejor_dv, mejor_tof = dv, float(d)
        except Exception:
            continue
    return mejor_tof

# Modelos cargados una sola vez (perezoso: solo al usarse)
_modelo_transfer = None
_modelo_transfer3d = None
_modelos_drag = {}


def _cargar_transfer():
    global _modelo_transfer
    if _modelo_transfer is None:
        _modelo_transfer = PPO.load(os.path.join(AQUI, "modelo_transfer", "best_model"))
    return _modelo_transfer


def _cargar_transfer3d():
    global _modelo_transfer3d
    if _modelo_transfer3d is None:
        _modelo_transfer3d = PPO.load(os.path.join(AQUI, "modelo_transfer3d", "best_model"))
    return _modelo_transfer3d


def _cargar_drag(planeta):
    if planeta not in _modelos_drag:
        _modelos_drag[planeta] = PPO.load(
            os.path.join(AQUI, "modelo_drag", planeta, "best_model"))
    return _modelos_drag[planeta]


def planificar_transferencia(planeta, h1_km, h2_km):
    """
    Planifica una transferencia coplanar entre dos órbitas circulares de altitudes
    h1_km y h2_km (km) alrededor de 'planeta', usando el AGENTE 2 (RL). Devuelve el
    Δv del agente, el óptimo de Hohmann de contraste y el exceso, en km/s.
    """
    planeta = str(planeta).lower().strip()
    if planeta not in CUERPOS:
        return {"error": f"Planeta '{planeta}' no reconocido. Opciones: {list(CUERPOS)}"}
    body = CUERPOS[planeta]
    mu = float(body.k.to_value(u.km**3 / u.s**2))
    Rb = float(body.R.to_value(u.km))
    r1, r2 = Rb + float(h1_km), Rb + float(h2_km)
    R = r2 / r1
    v_c1 = np.sqrt(mu / r1)                       # velocidad circular inicial real (km/s)

    model = _cargar_transfer()
    env = TransferEnv()
    obs, _ = env.reset(options={"R": R})          # el agente trabaja en adimensional
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    dv_agente = info["dv_total"] * v_c1           # adimensional -> km/s reales
    dv_optimo = abs(delta_v_hohmann(r1, r2, mu).dv_total)

    return {
        "planeta": planeta,
        "altitud_inicial_km": float(h1_km),
        "altitud_final_km": float(h2_km),
        "tipo": "subida" if r2 > r1 else "bajada",
        "dv_agente_km_s": round(dv_agente, 4),
        "dv_optimo_hohmann_km_s": round(dv_optimo, 4),
        "exceso_sobre_optimo_pct": round((dv_agente / dv_optimo - 1.0) * 100.0, 2),
        "error_de_llegada_pct": round(float(info.get("error", 0.0)) * 100.0, 2),
        "nota": ("Un exceso NEGATIVO no significa que el agente supere al óptimo de Hohmann: "
                 "ahorra un poco de combustible a costa de una pequeña imprecisión en la "
                 "órbita de llegada (ver error_de_llegada_pct). El óptimo de Hohmann sigue "
                 "siendo el mínimo teórico para llegar EXACTO."),
    }


def planificar_aerofrenado(planeta, apo_ini_km, apo_obj_km):
    """
    Planifica un aerofrenado en 'planeta': baja el apogeo desde apo_ini_km hasta
    apo_obj_km aprovechando el rozamiento atmosférico, usando el AGENTE 3 (RL,
    especialista de ese planeta). Devuelve el perigeo de operación, el número de
    pasadas, una estimación de la duración y si la maniobra tiene éxito.
    """
    planeta = str(planeta).lower().strip()
    if planeta not in PLANETAS_AEROFRENADO:
        return {"error": f"No hay agente de aerofrenado para '{planeta}'. "
                         f"Cuerpos con atmósfera: {PLANETAS_AEROFRENADO}"}
    if float(apo_obj_km) >= float(apo_ini_km):
        return {"error": "El aerofrenado solo BAJA el apogeo: apo_obj debe ser menor que apo_ini."}

    model = _cargar_drag(planeta)
    env = AeroBrakingEnv(planeta=planeta, aleatorio=False)
    obs, _ = env.reset(options={"apo_ini_km": float(apo_ini_km), "apo_obj_km": float(apo_obj_km)})

    term = trunc = False
    info = {}
    t_total_s = 0.0
    while not (term or trunc):
        accion, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(accion)
        # estima la duración sumando el periodo orbital de cada pasada
        r_a = env.R + env.h_apo
        r_p = env.R + env.h_per
        a = 0.5 * (r_a + r_p)
        t_total_s += 2.0 * np.pi * np.sqrt(a**3 / env.MU)

    resultado = ("exito" if info.get("exito") else
                 "destruido" if info.get("fallo") else "no_alcanzado")
    return {
        "planeta": planeta,
        "apogeo_inicial_km": float(apo_ini_km),
        "apogeo_objetivo_km": float(apo_obj_km),
        "resultado": resultado,
        "perigeo_operacion_km": round(info.get("h_per_km", 0.0), 1),
        "numero_de_pasadas": int(info.get("pasos", 0)),
        "duracion_estimada_dias": round(t_total_s / 86400.0, 1),
        "nota": "El aerofrenado casi no gasta combustible (usa el rozamiento atmosférico).",
    }


def planificar_cambio_plano(planeta, h1_km, h2_km, inclinacion_grados):
    """
    Transferencia con CAMBIO DE PLANO (Agente 4, 3D): sube de la órbita h1_km a la
    h2_km cambiando además la inclinación 'inclinacion_grados', con el reparto óptimo
    del giro (casi todo en el apogeo). Solo SUBE (h2 > h1), ratio r2/r1 hasta ~12 y
    cambio de plano hasta 40 grados. Devuelve el Δv del agente, el óptimo y el reparto.
    """
    planeta = str(planeta).lower().strip()
    if planeta not in CUERPOS:
        return {"error": f"Planeta '{planeta}' no reconocido. Opciones: {list(CUERPOS)}"}
    di = float(inclinacion_grados)
    body = CUERPOS[planeta]
    mu = float(body.k.to_value(u.km**3 / u.s**2))
    Rb = float(body.R.to_value(u.km))
    r1, r2 = Rb + float(h1_km), Rb + float(h2_km)
    if r2 <= r1:
        return {"error": "El agente de cambio de plano solo SUBE de órbita (h2 debe ser mayor que h1)."}
    R = r2 / r1
    if R > 11.94:
        return {"error": "El salto de órbitas es demasiado grande (ratio r2/r1 máximo ~12)."}
    if not (0.0 <= di <= 40.0):
        return {"error": "El cambio de plano debe estar entre 0 y 40 grados."}

    v_c1 = np.sqrt(mu / r1)
    model = _cargar_transfer3d()
    env = Transfer3DEnv()
    obs, _ = env.reset(options={"R": R, "di": np.radians(di)})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    dv_agente = info["dv_total"] * v_c1
    res = delta_v_hohmann_plano(r1, r2, np.radians(di), mu)     # óptimo (km/s) + reparto

    return {
        "planeta": planeta,
        "altitud_inicial_km": float(h1_km),
        "altitud_final_km": float(h2_km),
        "cambio_de_plano_grados": di,
        "dv_agente_km_s": round(dv_agente, 4),
        "dv_optimo_km_s": round(res.dv_total, 4),
        "exceso_sobre_optimo_pct": round((dv_agente / res.dv_total - 1.0) * 100.0, 2),
        "giro_en_perigeo_grados": round(res.di1_deg, 1),
        "giro_en_apogeo_grados": round(res.di2_deg, 1),
        "error_de_inclinacion_grados": round(info.get("err_incl_deg", 0.0), 2),
        "nota": "El cambio de plano se concentra en el apogeo, donde es más barato.",
    }


def transferencia_interplanetaria(origen, destino, fecha_salida, dias_vuelo=None):
    """
    Calcula una transferencia INTERPLANETARIA entre dos planetas resolviendo el
    problema de Lambert entre sus posiciones REALES (efemérides) en la fecha de
    salida y la de llegada (= salida + dias_vuelo). Devuelve el Δv de salida (exceso
    hiperbólico), el de llegada, el total y la energía característica C3. Método
    clásico de cónicas parcheadas; no usa RL (el óptimo se conoce).
    """
    origen, destino = str(origen).lower().strip(), str(destino).lower().strip()
    if origen not in PLANETAS_SOL or destino not in PLANETAS_SOL:
        return {"error": f"Origen y destino deben ser planetas: {list(PLANETAS_SOL)}"}
    if origen == destino:
        return {"error": "El origen y el destino deben ser distintos."}
    try:
        t0 = Time(str(fecha_salida), scale="tdb")
        if dias_vuelo is None:                       # por defecto: el TOF de MENOR Δv
            dias_vuelo = mejor_tof_dias(PLANETAS_SOL[origen], PLANETAS_SOL[destino], t0)
        tof = float(dias_vuelo) * u.day
        tf = t0 + tof
        epochs = Time(np.linspace(t0.jd, tf.jd, 50), format="jd", scale="tdb")
        eo = Ephem.from_body(PLANETAS_SOL[origen], epochs)
        ed = Ephem.from_body(PLANETAS_SOL[destino], epochs)
        r0, v0 = eo.rv(t0)
        rf, vf = ed.rv(tf)
        it = lambert(Sun.k, r0, rf, tof)
        v_dep, v_arr = it[0] if isinstance(it, list) else it
        dv_out = float(np.linalg.norm((v_dep - v0).to(u.km / u.s).value))    # exceso de salida
        dv_in = float(np.linalg.norm((vf - v_arr).to(u.km / u.s).value))     # exceso de llegada
        return {
            "origen": origen, "destino": destino,
            "fecha_salida": str(fecha_salida),
            "fecha_llegada": tf.datetime.strftime("%Y-%m-%d"),
            "dias_vuelo": round(float(dias_vuelo), 1),
            "dv_salida_km_s": round(dv_out, 3),
            "dv_llegada_km_s": round(dv_in, 3),
            "dv_total_km_s": round(dv_out + dv_in, 3),
            "C3_salida_km2_s2": round(dv_out**2, 2),
            "metodo": "Lambert heliocéntrico (cónicas parcheadas)",
        }
    except Exception as e:
        return {"error": f"No se pudo resolver la transferencia ({type(e).__name__}). "
                         f"Prueba otra fecha o días de vuelo. Detalle: {e}"}


if __name__ == "__main__":
    # PRUEBA de las herramientas SOLAS (sin LLM): deben devolver datos reales.
    import json
    print("=" * 64)
    print("  PRUEBA de las herramientas (sin LLM)")
    print("=" * 64)
    print("\n-- planificar_transferencia --")
    for planeta, h1, h2 in [("tierra", 400, 35786), ("marte", 400, 17000)]:
        print(f"  {planeta} {h1}->{h2}: ",
              json.dumps(planificar_transferencia(planeta, h1, h2), ensure_ascii=True))
    print("\n-- planificar_aerofrenado --")
    for planeta, a1, a2 in [("marte", 6000, 400), ("tierra", 3000, 500)]:
        print(f"  {planeta} {a1}->{a2}: ",
              json.dumps(planificar_aerofrenado(planeta, a1, a2), ensure_ascii=True))
    print("\n-- transferencia_interplanetaria --")
    for o, d, f, dv in [("tierra", "marte", "2026-11-01", 250), ("tierra", "venus", "2026-10-01", 150)]:
        print(f"  {o}->{d} ({f}, {dv}d): ",
              json.dumps(transferencia_interplanetaria(o, d, f, dv), ensure_ascii=True))

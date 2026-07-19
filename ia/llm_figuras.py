# ═══════════════════════════════════════════════════════════════════════════
#  ia/llm_figuras.py — Herramientas de DIBUJO para la capa LLM (Bloque 5)
#
#  A diferencia de los plotters de la memoria (figuras_*.py, que generan figuras
#  FIJAS para el documento), estas funciones son PARAMETRIZADAS: dibujan CUALQUIER
#  maniobra que pida el usuario por el chat, con datos del agente REAL, y devuelven
#  la ruta del PNG generado. El LLM las invoca como una herramienta más.
#
#  Las imágenes se guardan en imagenes/llm/ (aparte de las de la memoria).
#  Reutilizan el estilo (fondo oscuro) y la lógica de dibujo de figuras_orbitas.py.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 LLM_FIGURAS — Herramientas de DIBUJO de la capa LLM (Bloque 5)
 Genera las figuras que devuelven las 8 tools de dibujo del asistente: cada función
 dibuja la maniobra pedida con datos del agente RL o del solver REAL y devuelve la
 ruta del PNG (o HTML interactivo) generado en imagenes/llm/. El LLM las invoca como
 herramientas más; no producen figuras fijas de la memoria, sino parametrizadas.

 ÍNDICE DE FUNCIONES:
   - _abrir_imagen(ruta)                                  : abre el archivo con el visor del sistema (best-effort).
   - dibujar_transferencia(planeta, h1, h2)              : PNG de una transferencia coplanar (Agente 2).
   - dibujar_aerofrenado(planeta, apo_ini, apo_obj)      : PNG de la curva apogeo vs pasada (Agente 3).
   - dibujar_interplanetaria(o, d, fecha, tof)           : PNG 2D del viaje heliocéntrico (Lambert).
   - dibujar_interplanetaria_3d(o, d, fecha, tof)        : HTML 3D interactivo del viaje interplanetario (Plotly).
   - dibujar_cambio_plano_3d(planeta, h1, h2, di)        : HTML 3D interactivo del cambio de plano (Agente 4).
   - dibujar_aerofrenado_orbitas(planeta, apo_ini, apo_obj) : PNG de las órbitas encogiéndose (Agente 3).
   - dibujar_mantenimiento(planeta, h, inc)              : PNG altitud vs tiempo del station-keeping (Agente 5).
   - dibujar_porkchop(o, d, fecha_centro)                : PNG del porkchop / ventana de lanzamiento (Lambert).
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # sin ventanas: solo genera el archivo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Sun
from poliastro.ephem import Ephem
from poliastro.iod import lambert
from poliastro.twobody import Orbit

from env_transfer import TransferEnv
from env_drag import AeroBrakingEnv
from env_transfer3d import Transfer3DEnv
from env_keep import KeepEnv, BANDA_KM
from baselines import delta_v_hohmann
from llm_herramientas import (CUERPOS, _cargar_transfer, _cargar_drag, _cargar_transfer3d,
                              _cargar_keep, _ingenua_keep, PLANETAS_AEROFRENADO,
                              PLANETAS_MANTENIMIENTO, PLANETAS_SOL, mejor_tof_dias,
                              tof_hohmann_dias)

_AU_KM = 1.495978707e8                 # 1 unidad astronómica en km

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "llm"))
os.makedirs(IMG, exist_ok=True)

FONDO = "#0a0a1a"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})
COLOR_CUERPO = {"tierra": "#4ea8de", "marte": "#e07a5f", "venus": "#e9c46a",
                "jupiter": "#d9a066", "saturno": "#c9a66b", "urano": "#76c7c0",
                "neptuno": "#5e7ce2", "luna": "#dddddd", "mercurio": "#9b8cce"}


def _abrir_imagen(ruta):
    """Abre la imagen con el visor por defecto del sistema (best-effort: si falla,
    no pasa nada; la ruta se devuelve igual)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(ruta)                       # Windows
        elif sys.platform == "darwin":
            subprocess.Popen(["open", ruta])         # macOS
        else:
            subprocess.Popen(["xdg-open", ruta])     # Linux
    except Exception:
        pass


def dibujar_transferencia(planeta, h1_km, h2_km, abrir=True):
    """
    Dibuja, sobre el plano orbital, una transferencia entre dos órbitas circulares
    (altitudes h1_km y h2_km) alrededor de 'planeta', con los Δv del AGENTE 2.
    Guarda un PNG en imagenes/llm/, lo abre con el visor del sistema y devuelve su ruta.
    """
    planeta = str(planeta).lower().strip()
    if planeta not in CUERPOS:
        return {"error": f"Planeta '{planeta}' no reconocido. Opciones: {list(CUERPOS)}"}
    body = CUERPOS[planeta]
    mu = float(body.k.to_value(u.km**3 / u.s**2))
    Rb = float(body.R.to_value(u.km))
    r1, r2 = Rb + float(h1_km), Rb + float(h2_km)
    v_c1 = np.sqrt(mu / r1)

    # --- maniobra del agente (Δv reales en km/s) ---
    model = _cargar_transfer()
    env = TransferEnv()
    obs, _ = env.reset(options={"R": r2 / r1})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    dv1, dv2 = abs(info["dv1"]) * v_c1, abs(info["dv2"]) * v_c1
    dv_ag = info["dv_total"] * v_c1
    dv_opt = abs(delta_v_hohmann(r1, r2, mu).dv_total)

    # --- geometría de las órbitas ---
    th = np.linspace(0, 2 * np.pi, 400)
    sube = r2 > r1
    rp, ra = (r1, r2) if sube else (r2, r1)         # perigeo (+x) y apogeo (-x)
    a_t, e_t = 0.5 * (r1 + r2), abs(r2 - r1) / (r1 + r2)
    # media elipse: arco SUPERIOR si sube, INFERIOR si baja (mismo sentido de giro)
    th_tr = np.linspace(0, np.pi, 200) if sube else np.linspace(np.pi, 2 * np.pi, 200)
    r_tr = a_t * (1 - e_t**2) / (1 + e_t * np.cos(th_tr))

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    col = COLOR_CUERPO.get(planeta, "#4ea8de")
    ax.fill(Rb * np.cos(th), Rb * np.sin(th), color=col, alpha=0.9, zorder=6)
    ax.text(0, 0, planeta.capitalize(), ha="center", va="center", color="#fff",
            fontsize=8, zorder=7)
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), color="#4ade80", lw=2,
            label=f"Órbita inicial ({h1_km:.0f} km)")
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), color="#e9c46a", lw=2,
            label=f"Órbita final ({h2_km:.0f} km)")
    ax.plot(r_tr * np.cos(th_tr), r_tr * np.sin(th_tr), color="#f87171", lw=2.2, ls="--",
            label="Elipse de transferencia")
    ax.plot(rp, 0, "o", color="#fff", ms=8, zorder=8)
    ax.plot(-ra, 0, "o", color="#fff", ms=8, zorder=8)
    ax.annotate(f"Δv₁ = {dv1:.3f} km/s", xy=(rp, 0), xytext=(rp * 0.35, -ra * 0.45),
                color="#4ade80", fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.3))
    ax.annotate(f"Δv₂ = {dv2:.3f} km/s", xy=(-ra, 0), xytext=(-ra * 0.5, ra * 0.42),
                color="#e9c46a", fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#e9c46a", lw=1.3))
    txt = (f"Agente:  Δv = {dv_ag:.3f} km/s\n"
           f"Óptimo:  Δv = {dv_opt:.3f} km/s\n"
           f"exceso = {(dv_ag / dv_opt - 1) * 100:+.1f} %")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="#eee", bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Transferencia de {'subida' if sube else 'bajada'} en "
                 f"{planeta.capitalize()}  ({h1_km:.0f} → {h2_km:.0f} km)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", facecolor="#15152a", edgecolor="#555",
              labelcolor="#eee", fontsize=9)

    ruta = os.path.join(IMG, f"transferencia_{planeta}_{int(h1_km)}_{int(h2_km)}.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    if abrir:
        _abrir_imagen(ruta)
    return {"figura_png": ruta, "planeta": planeta,
            "tipo": "subida" if sube else "bajada"}


def dibujar_aerofrenado(planeta, apo_ini_km, apo_obj_km, abrir=True):
    """
    Dibuja la curva de altitud del APOGEO frente al número de pasada durante un
    aerofrenado en 'planeta', con la trayectoria real del AGENTE 3. Guarda un PNG
    en imagenes/llm/, lo abre con el visor del sistema y devuelve su ruta.
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
    pasos, apos = [0], [env.h_apo / 1000.0]
    term = trunc = False
    info = {}
    while not (term or trunc):
        accion, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(accion)
        pasos.append(info.get("pasos", len(pasos)))
        apos.append(env.h_apo / 1000.0)
    resultado = ("éxito" if info.get("exito") else
                 "destruido" if info.get("fallo") else "no alcanzado")

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    ax.plot(pasos, apos, color="#4ea8de", lw=2.2, label="Apogeo (agente RL)")
    ax.axhline(float(apo_obj_km), color="#4ade80", ls="--", lw=1.5,
               label=f"Objetivo ({apo_obj_km:.0f} km)")
    ax.set_xlabel("Número de pasada por el perigeo")
    ax.set_ylabel("Altitud del apogeo (km)")
    ax.set_title(f"Aerofrenado en {planeta.capitalize()}:  "
                 f"{apo_ini_km:.0f} → {apo_obj_km:.0f} km")
    txt = (f"resultado: {resultado}\n"
           f"perigeo de op.: {info.get('h_per_km', 0):.0f} km\n"
           f"pasadas: {info.get('pasos', 0)}")
    ax.text(0.03, 0.05, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color="#eee", bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", facecolor="#15152a", edgecolor="#555",
              labelcolor="#eee", fontsize=9)

    ruta = os.path.join(IMG, f"aerofrenado_{planeta}_{int(apo_ini_km)}_{int(apo_obj_km)}.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    if abrir:
        _abrir_imagen(ruta)
    return {"figura_png": ruta, "planeta": planeta, "resultado": resultado}


def dibujar_interplanetaria(origen, destino, fecha_salida, dias_vuelo=None, abrir=True):
    """
    Dibuja la trayectoria HELIOCÉNTRICA de un viaje interplanetario: el Sol en el
    centro, las órbitas de los dos planetas y el arco de transferencia (Lambert)
    entre la posición de salida y la de llegada. Guarda un PNG en imagenes/llm/,
    lo abre y devuelve su ruta. (Vista proyectada en el plano de la eclíptica, en UA.)
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
        v_dep, _ = it[0] if isinstance(it, list) else it

        # arco de transferencia: propagar la órbita de Lambert a lo largo del viaje
        trans = Orbit.from_vectors(Sun, r0, v_dep, epoch=t0)
        xs, ys = [], []
        for dt in np.linspace(0, float(dias_vuelo), 120):
            r = trans.propagate(dt * u.day).r.to(u.km).value
            xs.append(r[0] / _AU_KM); ys.append(r[1] / _AU_KM)

        r0_au = r0.to(u.km).value / _AU_KM        # posición de salida (UA)
        rf_au = rf.to(u.km).value / _AU_KM        # posición de llegada (UA)
        d0 = np.hypot(r0_au[0], r0_au[1])         # radio heliocéntrico de cada planeta
        d1 = np.hypot(rf_au[0], rf_au[1])

        th = np.linspace(0, 2 * np.pi, 300)
        fig, ax = plt.subplots(figsize=(7.6, 7.6))
        ax.plot(0, 0, marker="o", color="#ffd24a", ms=13, zorder=5)
        ax.text(0, -max(d0, d1) * 0.05, "Sol", color="#ffd24a", ha="center", va="top", fontsize=8)
        co = COLOR_CUERPO.get(origen, "#4ade80")
        cd = COLOR_CUERPO.get(destino, "#e9c46a")
        ax.plot(d0 * np.cos(th), d0 * np.sin(th), color=co, lw=1.2, ls=":",
                label=f"Órbita de {origen.capitalize()}")
        ax.plot(d1 * np.cos(th), d1 * np.sin(th), color=cd, lw=1.2, ls=":",
                label=f"Órbita de {destino.capitalize()}")
        ax.plot(xs, ys, color="#f87171", lw=2.3, label="Trayectoria de transferencia")
        ax.plot(r0_au[0], r0_au[1], "o", color=co, ms=10, zorder=6)
        ax.annotate(f"{origen.capitalize()} (salida)", (r0_au[0], r0_au[1]),
                    color="#eee", fontsize=8.5, ha="center", va="bottom")
        ax.plot(rf_au[0], rf_au[1], "o", color=cd, ms=10, zorder=6)
        ax.annotate(f"{destino.capitalize()} (llegada)", (rf_au[0], rf_au[1]),
                    color="#eee", fontsize=8.5, ha="center", va="bottom")
        ax.set_aspect("equal")
        ax.set_xlabel("x (UA)"); ax.set_ylabel("y (UA)")
        ax.set_title(f"Transferencia interplanetaria {origen.capitalize()} "
                     f"$\\rightarrow$ {destino.capitalize()}\n"
                     f"salida {fecha_salida}, {dias_vuelo:.0f} días", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", facecolor="#15152a", edgecolor="#555",
                  labelcolor="#eee", fontsize=8.5)

        ruta = os.path.join(IMG, f"interplanetaria_{origen}_{destino}_{t0.datetime:%Y%m%d}.png")
        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
        plt.close(fig)
        if abrir:
            _abrir_imagen(ruta)
        return {"figura_png": ruta, "origen": origen, "destino": destino}
    except Exception as e:
        return {"error": f"No se pudo dibujar la transferencia ({type(e).__name__}): {e}"}


def dibujar_interplanetaria_3d(origen, destino, fecha_salida, dias_vuelo=None, abrir=True):
    """
    Versión INTERACTIVA en 3D (Plotly) del viaje interplanetario: genera un archivo
    HTML con el Sol, las órbitas 3D de los dos planetas (con su inclinación real) y la
    trayectoria de transferencia, que se abre en el NAVEGADOR y se puede rotar y hacer
    zoom. Guarda el HTML en imagenes/llm/ y devuelve su ruta.
    """
    import plotly.graph_objects as go            # import local: solo si se usa

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
        v_dep, _ = it[0] if isinstance(it, list) else it

        # órbitas COMPLETAS (3D) de cada planeta, en UA
        def _orbita_au(ephem, epoch):
            c = Orbit.from_ephem(Sun, ephem, epoch).sample(360)
            return (c.x.to(u.km).value / _AU_KM, c.y.to(u.km).value / _AU_KM,
                    c.z.to(u.km).value / _AU_KM)
        ox, oy, oz = _orbita_au(eo, t0)
        dx, dy, dz = _orbita_au(ed, tf)

        # arco de transferencia (3D), en UA
        trans = Orbit.from_vectors(Sun, r0, v_dep, epoch=t0)
        tx, ty, tz = [], [], []
        for dt in np.linspace(0, float(dias_vuelo), 150):
            r = trans.propagate(dt * u.day).r.to(u.km).value
            tx.append(r[0] / _AU_KM); ty.append(r[1] / _AU_KM); tz.append(r[2] / _AU_KM)
        p0 = r0.to(u.km).value / _AU_KM
        p1 = rf.to(u.km).value / _AU_KM
        co = COLOR_CUERPO.get(origen, "#4ade80")
        cd = COLOR_CUERPO.get(destino, "#e9c46a")

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                   marker=dict(size=6, color="#ffd24a"), name="Sol"))
        fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode="lines",
                                   line=dict(color=co, width=3, dash="dot"),
                                   name=f"Órbita de {origen.capitalize()}"))
        fig.add_trace(go.Scatter3d(x=dx, y=dy, z=dz, mode="lines",
                                   line=dict(color=cd, width=3, dash="dot"),
                                   name=f"Órbita de {destino.capitalize()}"))
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines",
                                   line=dict(color="#f87171", width=5),
                                   name="Trayectoria de transferencia"))
        fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers+text",
                                   marker=dict(size=6, color=co),
                                   text=[f"{origen.capitalize()} (salida)"],
                                   textposition="top center", name=f"{origen.capitalize()} (salida)"))
        fig.add_trace(go.Scatter3d(x=[p1[0]], y=[p1[1]], z=[p1[2]], mode="markers+text",
                                   marker=dict(size=6, color=cd),
                                   text=[f"{destino.capitalize()} (llegada)"],
                                   textposition="top center", name=f"{destino.capitalize()} (llegada)"))
        fig.update_layout(
            template="plotly_dark",
            title=(f"Transferencia interplanetaria {origen.capitalize()} → {destino.capitalize()}  "
                   f"(salida {fecha_salida}, {dias_vuelo:.0f} días)"),
            scene=dict(xaxis_title="x (UA)", yaxis_title="y (UA)", zaxis_title="z (UA)",
                       aspectmode="data"),
            paper_bgcolor="#0a0a1a")

        ruta = os.path.join(IMG, f"interplanetaria3d_{origen}_{destino}_{t0.datetime:%Y%m%d}.html")
        fig.write_html(ruta)
        if abrir:
            _abrir_imagen(ruta)
        return {"figura_html": ruta, "origen": origen, "destino": destino, "interactiva": True}
    except Exception as e:
        return {"error": f"No se pudo dibujar la transferencia 3D ({type(e).__name__}): {e}"}


def dibujar_cambio_plano_3d(planeta, h1_km, h2_km, inclinacion_grados, abrir=True):
    """
    Visualización INTERACTIVA en 3D (Plotly → HTML) de una transferencia con CAMBIO DE
    PLANO (Agente 4): la órbita inicial en el plano de referencia, la elipse de
    transferencia y la órbita final INCLINADA. En 3D se aprecia la inclinación (en 2D
    no se vería). Se abre en el navegador, rotable. Devuelve la ruta del HTML.
    """
    import plotly.graph_objects as go

    planeta = str(planeta).lower().strip()
    if planeta not in CUERPOS:
        return {"error": f"Planeta '{planeta}' no reconocido. Opciones: {list(CUERPOS)}"}
    di = float(inclinacion_grados)
    body = CUERPOS[planeta]
    mu = float(body.k.to_value(u.km**3 / u.s**2))
    Rb = float(body.R.to_value(u.km))
    r1, r2 = Rb + float(h1_km), Rb + float(h2_km)
    if r2 <= r1:
        return {"error": "El cambio de plano solo SUBE de órbita (h2 debe ser mayor que h1)."}
    R = r2 / r1
    if R > 11.94 or not (0.0 <= di <= 40.0):
        return {"error": "Fuera de rango: sube hasta ratio r2/r1 ~12 y cambio de plano 0-40 grados."}
    try:
        v_c1 = np.sqrt(mu / r1)
        model = _cargar_transfer3d()
        env = Transfer3DEnv()
        obs, _ = env.reset(options={"R": R, "di": np.radians(di)})
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)

        # geometría 3D de la maniobra (igual que figuras_transfer3d), en km
        dv1t, dv1n = info["dv1t"], info["dv1n"]
        g1 = np.arctan2(dv1n, 1.0 + dv1t)                 # inclinación de la transferencia
        a1 = -1.0 / (2.0 * (0.5 * ((1.0 + dv1t)**2 + dv1n**2) - 1.0))
        r_apo = (2.0 * a1 - 1.0) * r1
        i_f = np.radians(info["i_f_deg"])                 # inclinación final lograda

        th = np.linspace(0, 2 * np.pi, 240)
        o0 = (r1 * np.cos(th), r1 * np.sin(th), np.zeros_like(th))           # inicial (plano ref)
        of = (r2 * np.cos(th), r2 * np.sin(th) * np.cos(i_f),
              r2 * np.sin(th) * np.sin(i_f))                                  # final inclinada
        a_t, e_t = 0.5 * (r1 + r_apo), (r_apo - r1) / (r_apo + r1)
        nu = np.linspace(0, np.pi, 160)
        r_tr = a_t * (1 - e_t**2) / (1 + e_t * np.cos(nu))
        xt, yt = r_tr * np.cos(nu), r_tr * np.sin(nu)
        tr = (xt, yt * np.cos(g1), yt * np.sin(g1))                          # transferencia (plano g1)

        col = COLOR_CUERPO.get(planeta, "#4ea8de")
        dv1 = np.hypot(dv1t, dv1n) * v_c1
        dv2 = np.hypot(info["dv2t"], info["dv2n"]) * v_c1

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                   marker=dict(size=8, color=col), name=planeta.capitalize()))
        fig.add_trace(go.Scatter3d(x=o0[0], y=o0[1], z=o0[2], mode="lines",
                                   line=dict(color="#4ade80", width=4),
                                   name=f"Órbita inicial ({h1_km:.0f} km)"))
        fig.add_trace(go.Scatter3d(x=tr[0], y=tr[1], z=tr[2], mode="lines",
                                   line=dict(color="#f87171", width=5, dash="dash"),
                                   name="Elipse de transferencia"))
        fig.add_trace(go.Scatter3d(x=of[0], y=of[1], z=of[2], mode="lines",
                                   line=dict(color="#e9c46a", width=4),
                                   name=f"Órbita final (inclinada {di:.0f}°)"))
        fig.add_trace(go.Scatter3d(x=[r1], y=[0], z=[0], mode="markers",
                                   marker=dict(size=4, color="#fff"),
                                   name=f"Δv₁ = {dv1:.3f} km/s (inyección)"))
        fig.add_trace(go.Scatter3d(x=[-r_apo], y=[0], z=[0], mode="markers",
                                   marker=dict(size=4, color="#fff"),
                                   name=f"Δv₂ = {dv2:.3f} km/s (circulariza + gira el plano)"))
        fig.update_layout(
            template="plotly_dark",
            title=(f"Transferencia con cambio de plano en {planeta.capitalize()}  "
                   f"({h1_km:.0f} → {h2_km:.0f} km, {di:.0f}°)"),
            scene=dict(xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="z (km)",
                       aspectmode="data"),
            paper_bgcolor="#0a0a1a")

        ruta = os.path.join(IMG, f"cambioplano3d_{planeta}_{int(h1_km)}_{int(h2_km)}_{int(di)}.html")
        fig.write_html(ruta)
        if abrir:
            _abrir_imagen(ruta)
        return {"figura_html": ruta, "planeta": planeta, "interactiva": True}
    except Exception as e:
        return {"error": f"No se pudo dibujar el cambio de plano 3D ({type(e).__name__}): {e}"}


def dibujar_aerofrenado_orbitas(planeta, apo_ini_km, apo_obj_km, abrir=True):
    """
    Dibuja cómo la ÓRBITA se encoge y circulariza durante el aerofrenado: varias
    órbitas superpuestas (de la elíptica inicial a la casi circular final), con el
    planeta en el centro y el perigeo fijo. Datos reales del AGENTE 3. Guarda un PNG.
    """
    planeta = str(planeta).lower().strip()
    if planeta not in PLANETAS_AEROFRENADO:
        return {"error": f"No hay agente de aerofrenado para '{planeta}'. "
                         f"Cuerpos con atmósfera: {PLANETAS_AEROFRENADO}"}
    if float(apo_obj_km) >= float(apo_ini_km):
        return {"error": "El aerofrenado solo BAJA el apogeo: apo_obj debe ser menor que apo_ini."}
    try:
        model = _cargar_drag(planeta)
        env = AeroBrakingEnv(planeta=planeta, aleatorio=False)
        obs, _ = env.reset(options={"apo_ini_km": float(apo_ini_km), "apo_obj_km": float(apo_obj_km)})
        Rk = env.R / 1000.0                                   # radio del planeta (km)
        apos, pers = [env.h_apo / 1000.0], [env.h_per / 1000.0]
        term = trunc = False
        info = {}
        while not (term or trunc):
            accion, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(accion)
            apos.append(env.h_apo / 1000.0); pers.append(env.h_per / 1000.0)

        n = len(apos)
        idxs = np.unique(np.linspace(0, n - 1, 7).astype(int))   # ~7 órbitas representativas
        th = np.linspace(0, 2 * np.pi, 400)
        fig, ax = plt.subplots(figsize=(7.6, 7.6))
        cp = COLOR_CUERPO.get(planeta, "#e07a5f")
        ax.fill(Rk * np.cos(th), Rk * np.sin(th), color=cp, alpha=0.9, zorder=6)
        ax.text(0, 0, planeta.capitalize(), ha="center", va="center", color="#fff",
                fontsize=8, zorder=7)
        cmap = plt.cm.plasma
        for k, i in enumerate(idxs):
            r_p, r_a = Rk + pers[i], Rk + apos[i]              # perigeo en +x, apogeo en -x
            a, e = 0.5 * (r_p + r_a), (r_a - r_p) / (r_a + r_p)
            r = a * (1 - e**2) / (1 + e * np.cos(th))          # foco (planeta) en el origen
            color = cmap(k / (len(idxs) - 1))
            etiqueta = ("órbita inicial" if i == 0 else
                        "órbita final" if i == idxs[-1] else None)
            ax.plot(r * np.cos(th), r * np.sin(th), color=color, lw=1.6, alpha=0.9,
                    label=etiqueta)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_title(f"Aerofrenado en {planeta.capitalize()}: la órbita se circulariza\n"
                     f"{apo_ini_km:.0f} → {apo_obj_km:.0f} km · {info.get('pasos', 0)} pasadas")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", facecolor="#15152a", edgecolor="#555",
                  labelcolor="#eee", fontsize=9)

        ruta = os.path.join(IMG, f"aerofrenado_orbitas_{planeta}_{int(apo_ini_km)}_{int(apo_obj_km)}.png")
        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
        plt.close(fig)
        if abrir:
            _abrir_imagen(ruta)
        return {"figura_png": ruta, "planeta": planeta, "pasadas": int(info.get("pasos", 0))}
    except Exception as e:
        return {"error": f"No se pudo dibujar el aerofrenado ({type(e).__name__}): {e}"}


def dibujar_mantenimiento(planeta, h_km, inclinacion_grados=51.6, abrir=True):
    """
    Dibuja el MANTENIMIENTO ORBITAL (station-keeping) a altitud h_km en 'planeta':
    la altitud frente al tiempo durante un año, mostrando el ciclo caer/re-boostear.
    Compara el AGENTE 5 (mantiene la altitud estable) con la estrategia ingenua
    (diente de sierra) y marca la banda de tolerancia. Guarda un PNG y devuelve su ruta.
    """
    planeta = str(planeta).lower().strip()
    if planeta not in PLANETAS_MANTENIMIENTO:
        return {"error": f"No hay agente de mantenimiento para '{planeta}'. Solo cuerpos con "
                         f"atmósfera: {PLANETAS_MANTENIMIENTO} (sin aire no hay nada que mantener)."}
    if float(h_km) <= 0:
        return {"error": "La altitud debe ser positiva."}
    try:
        model = _cargar_keep(planeta)
        opts = {"h_obj_km": float(h_km), "inc_deg": float(inclinacion_grados)}

        env = KeepEnv(planeta=planeta, aleatorio=False)
        obs, _ = env.reset(options=opts)
        h_ag = [env._h(env.a) / 1000.0]
        term = trunc = False
        while not (term or trunc):
            accion, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(accion)
            h_ag.append(info["h_km"])

        env2 = KeepEnv(planeta=planeta, aleatorio=False)
        obs, _ = env2.reset(options=opts)
        h_in = [env2._h(env2.a) / 1000.0]
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, info = env2.step(_ingenua_keep(env2))
            h_in.append(info["h_km"])

        h_obj = float(h_km)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhspan(h_obj - BANDA_KM, h_obj + BANDA_KM, color="#2a2a4a", alpha=0.5,
                   label=f"Banda de tolerancia (±{BANDA_KM:.0f} km)")
        ax.axhline(h_obj, color="#bbbbbb", ls=":", lw=1.2, label=f"Objetivo ({h_obj:.0f} km)")
        ax.plot(np.arange(len(h_in)), h_in, color="#f87171", lw=1.5, ls="--",
                label="Estrategia ingenua (diente de sierra)")
        ax.plot(np.arange(len(h_ag)), h_ag, color="#4ade80", lw=1.8,
                label="Agente RL (mantiene la altitud estable)")
        ax.set_xlabel("Tiempo (días)")
        ax.set_ylabel("Altitud (km)")
        ax.set_title(f"Mantenimiento orbital en {planeta.capitalize()}: "
                     f"{h_obj:.0f} km durante un año")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", facecolor="#15152a", edgecolor="#555",
                  labelcolor="#eee", fontsize=9)

        ruta = os.path.join(IMG, f"mantenimiento_{planeta}_{int(h_km)}.png")
        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
        plt.close(fig)
        if abrir:
            _abrir_imagen(ruta)
        return {"figura_png": ruta, "planeta": planeta, "altitud_km": float(h_km)}
    except Exception as e:
        return {"error": f"No se pudo dibujar el mantenimiento ({type(e).__name__}): {e}"}


def dibujar_porkchop(origen, destino, fecha_centro, abrir=True):
    """
    Genera el PORKCHOP de un viaje interplanetario: un mapa del Δv total según la
    FECHA DE SALIDA (eje X) y el TIEMPO DE VUELO (eje Y), que revela la VENTANA de
    lanzamiento óptima (el valle de mínimo Δv). Resuelve Lambert en una rejilla de
    fechas × tiempos de vuelo alrededor de fecha_centro. Guarda un PNG y devuelve la
    ruta + la mejor combinación (fecha y días de vuelo de menor Δv).
    """
    origen, destino = str(origen).lower().strip(), str(destino).lower().strip()
    if origen not in PLANETAS_SOL or destino not in PLANETAS_SOL:
        return {"error": f"Origen y destino deben ser planetas: {list(PLANETAS_SOL)}"}
    if origen == destino:
        return {"error": "El origen y el destino deben ser distintos."}
    try:
        t0c = Time(str(fecha_centro), scale="tdb")
        body_o, body_d = PLANETAS_SOL[origen], PLANETAS_SOL[destino]
        tof_h = tof_hohmann_dias(body_o, body_d, t0c)
        salidas = Time(np.linspace((t0c - 150 * u.day).jd, (t0c + 650 * u.day).jd, 45),
                       format="jd", scale="tdb")
        tofs = np.linspace(0.5 * tof_h, 1.6 * tof_h, 38)
        span = Time(np.linspace(salidas[0].jd, (salidas[-1] + tofs[-1] * u.day).jd, 400),
                    format="jd", scale="tdb")
        eo = Ephem.from_body(body_o, span)
        ed = Ephem.from_body(body_d, span)

        dv = np.full((len(tofs), len(salidas)), np.nan)
        for i, sal in enumerate(salidas):
            r0, v0 = eo.rv(sal)
            for j, d in enumerate(tofs):
                try:
                    rf, vf = ed.rv(sal + d * u.day)
                    it = lambert(Sun.k, r0, rf, d * u.day)
                    vd, va = it[0] if isinstance(it, list) else it
                    dv[j, i] = (float(np.linalg.norm((vd - v0).to(u.km / u.s).value))
                                + float(np.linalg.norm((vf - va).to(u.km / u.s).value)))
                except Exception:
                    continue

        idx = np.unravel_index(np.nanargmin(dv), dv.shape)
        dv_min = float(dv[idx])
        fecha_opt = salidas[idx[1]].datetime
        tof_opt = float(tofs[idx[0]])

        X, Y = np.meshgrid(mdates.date2num(salidas.datetime), tofs)
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        cf = ax.contourf(X, Y, dv, levels=30, cmap="viridis")
        fig.colorbar(cf, ax=ax).set_label("Δv total (km/s)")
        ax.plot(mdates.date2num(fecha_opt), tof_opt, "*", color="#ff4d4d", ms=18,
                markeredgecolor="white", label="Ventana óptima")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_xlabel("Fecha de salida"); ax.set_ylabel("Tiempo de vuelo (días)")
        ax.set_title(f"Porkchop {origen.capitalize()} → {destino.capitalize()}\n"
                     f"Mínimo: {dv_min:.2f} km/s · salida {fecha_opt:%d-%b-%Y} · {tof_opt:.0f} días")
        ax.legend(loc="upper right", facecolor="#15152a", edgecolor="#555", labelcolor="#eee")

        ruta = os.path.join(IMG, f"porkchop_{origen}_{destino}_{t0c.datetime:%Y%m%d}.png")
        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
        plt.close(fig)
        if abrir:
            _abrir_imagen(ruta)
        return {"figura_png": ruta, "origen": origen, "destino": destino,
                "dv_minimo_km_s": round(dv_min, 2),
                "fecha_salida_optima": fecha_opt.strftime("%Y-%m-%d"),
                "dias_vuelo_optimos": round(tof_opt)}
    except Exception as e:
        return {"error": f"No se pudo generar el porkchop ({type(e).__name__}): {e}"}


if __name__ == "__main__":
    import json
    # abrir=False en las pruebas para no lanzar el visor; en el uso real (LLM) es True
    print("PRUEBA de dibujar_transferencia:")
    for planeta, h1, h2 in [("tierra", 400, 35786), ("venus", 600, 12000)]:
        print(json.dumps(dibujar_transferencia(planeta, h1, h2, abrir=False), ensure_ascii=True))
    print("\nPRUEBA de dibujar_aerofrenado:")
    for planeta, a1, a2 in [("marte", 6000, 400)]:
        print(json.dumps(dibujar_aerofrenado(planeta, a1, a2, abrir=False), ensure_ascii=True))
    print("\nPRUEBA de dibujar_mantenimiento:")
    for planeta, h in [("tierra", 420), ("marte", 171)]:
        print(json.dumps(dibujar_mantenimiento(planeta, h, abrir=False), ensure_ascii=True))

# ═══════════════════════════════════════════════════════════════════════════
#  ia/figuras_keep.py — Figuras del Agente 5 (mantenimiento orbital) para la memoria
#
#  Uso:  python ia/figuras_keep.py
#  TODO con datos REALES: los agentes (modelo_keep/<planeta>/best_model) corridos en
#  env_keep.py (fisica secular drag + J2, atmosferas validadas). La "estrategia
#  ingenua" es el baseline (re-boost a tope al caer medio-banda).
#
#  Salida en imagenes/ia/ (PDF vectorial + PNG, fondo oscuro del proyecto):
#    1) ia_keep_diente_sierra   — altitud vs tiempo (ISS): el ciclo caer/re-boostear
#    2) ia_keep_franjas         — franja operativa derivada de cada cuerpo
#    3) ia_keep_coste_altitud   — Dv/ano de mantenimiento vs altitud (los 7 cuerpos)
#    4) ia_keep_ahorro          — ahorro del agente frente a la ingenua, por cuerpo
# ═══════════════════════════════════════════════════════════════════════════
"""
═══════════════════════════════════════════════════════════════════════════════
 FIGURAS DEL AGENTE 5 (MANTENIMIENTO ORBITAL) — gráficas para la memoria
 Corre los agentes PPO reales (modelo_keep/<planeta>/best_model) sobre env_keep
 (física secular drag + J2, atmósferas validadas) y vuelca las figuras a imagenes/ia/.

 ÍNDICE DE FUNCIONES:
   - guardar(fig, nombre)              : guarda la figura como PDF vectorial y PNG.
   - _ingenua(env, obs)               : política baseline (re-boost a tope al caer media banda).
   - _serie_altitud(planeta, politica): corre un episodio y devuelve la altitud por día y el Δv total.
   - fig_diente_sierra()              : altitud vs tiempo en la Tierra (ISS), agente vs ingenua.
   - fig_franjas()                    : franja operativa de mantenimiento de cada cuerpo.
   - fig_coste_altitud()              : Δv/año de mantenimiento frente a la altitud (7 cuerpos).
   - fig_ahorro(n)                    : ahorro de Δv del agente frente a la ingenua, por cuerpo.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env_keep import (KeepEnv, BANDA_KM, DV_OP_ALTO, DV_OP_BAJO,
                      _dv_anual_mantenimiento, PLANETAS)

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "ia"))
FONDO = "#0a0a1a"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})
# Orden terrestres -> gigantes (escala creciente)
CUERPOS = ["marte", "tierra", "venus", "jupiter", "neptuno", "saturno", "urano"]
COLORES = {"marte": "#e07a5f", "tierra": "#4ea8de", "venus": "#e9c46a",
           "saturno": "#c9a66b", "urano": "#76c7c0", "neptuno": "#5e7ce2",
           "jupiter": "#d9a066"}
LEG = dict(facecolor="#15152a", edgecolor="#555", labelcolor="#eee", framealpha=0.93)


def guardar(fig, nombre):
    """Guarda la figura en imagenes/ia/ como PDF vectorial y PNG (200 dpi), fondo oscuro."""
    fig.savefig(os.path.join(IMG, nombre + ".pdf"), bbox_inches="tight", facecolor=FONDO)
    fig.savefig(os.path.join(IMG, nombre + ".png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
    print("  guardada:", nombre)


def _ingenua(env, obs):
    """Política ingenua de baseline: re-boostea a tope (+1) al caer por debajo de media banda,
    y no hace nada (-1) en caso contrario. Sirve para comparar contra el agente RL."""
    desv_km = (env.a - env.a_obj) / 1000.0
    return np.array([1.0 if desv_km < -BANDA_KM * 0.5 else -1.0], dtype=np.float32)


def _serie_altitud(planeta, politica):
    """Corre un episodio (caso fijo) y devuelve la altitud por dia y el Dv total."""
    env = KeepEnv(planeta=planeta, aleatorio=False)
    obs, _ = env.reset(seed=0)
    h = [env._h(env.a) / 1000.0]
    term = trunc = False
    info = {}
    while not (term or trunc):
        obs, _, term, trunc, info = env.step(politica(env, obs))
        h.append(info["h_km"])
    return np.array(h), info.get("dv_acum", 0.0), env


# ── 1) Diente de sierra: altitud vs tiempo (ISS) ────────────────────────────
def fig_diente_sierra():
    """Figura ia_keep_diente_sierra: altitud vs tiempo en la Tierra (ISS), mostrando el
    diente de sierra de la estrategia ingenua frente al agente RL, que se mantiene estable."""
    m = PPO.load(os.path.join(AQUI, "modelo_keep", "tierra", "best_model"))
    h_ag, dv_ag, env = _serie_altitud("tierra", lambda e, o: m.predict(o, deterministic=True)[0])
    h_in, dv_in, _ = _serie_altitud("tierra", _ingenua)
    h_obj = env._h(env.a_obj) / 1000.0
    dias = np.arange(len(h_ag))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhspan(h_obj - BANDA_KM, h_obj + BANDA_KM, color="#2a2a4a", alpha=0.5,
               label=f"Banda de tolerancia (±{BANDA_KM:.0f} km)")
    ax.axhline(h_obj, color="#bbbbbb", ls=":", lw=1.2, label=f"Objetivo = {h_obj:.0f} km")
    ax.plot(dias, h_in, color="#f87171", lw=1.6, ls="--",
            label="Estrategia ingenua (diente de sierra: deja caer y re-boostea)")
    ax.plot(dias, h_ag, color="#4ade80", lw=1.8,
            label="Agente RL (mantiene la altitud estable cerca del objetivo)")
    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Altitud (km)")
    ax.set_title("Mantenimiento orbital en la Tierra (ISS): el ciclo caer / re-boostear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower left", **LEG)
    guardar(fig, "ia_keep_diente_sierra")
    plt.close(fig)


# ── 2) Franja operativa derivada de cada cuerpo ─────────────────────────────
def fig_franjas():
    """Figura ia_keep_franjas: barras horizontales (escala log) con la franja de altitudes
    operativas de mantenimiento derivada de la atmósfera de cada cuerpo."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k, p in enumerate(CUERPOS):
        env = KeepEnv(planeta=p, aleatorio=False)
        lo, hi = env.h_op_min / 1000.0, env.h_op_max / 1000.0
        ax.hlines(k, lo, hi, color=COLORES[p], lw=9, alpha=0.9)
        ax.plot([lo, hi], [k, k], "o", color=COLORES[p], ms=6)
        ax.text(hi * 1.05, k, f"{lo:.0f}–{hi:.0f} km", va="center", fontsize=9, color="#ddd")
    ax.set_yticks(range(len(CUERPOS)))
    ax.set_yticklabels([p.capitalize() for p in CUERPOS])
    ax.set_xscale("log")
    ax.set_xlabel("Altitud sobre la superficie (km, escala log)")
    ax.set_title("Franja operativa de mantenimiento derivada de cada atmósfera\n"
                 "(altitudes donde mantener un año cuesta entre "
                 f"{DV_OP_BAJO:.0f} y {DV_OP_ALTO:.0f} m/s)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(100, 12000)
    guardar(fig, "ia_keep_franjas")
    plt.close(fig)


# ── 3) Coste de mantenimiento Dv/año vs altitud (los 7 cuerpos) ─────────────
def fig_coste_altitud():
    """Figura ia_keep_coste_altitud: Δv/año de mantenimiento frente a la altitud (ejes log-log)
    para los 7 cuerpos; el coste cae al subir la altitud por la densidad exponencial."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for p in CUERPOS:
        env = KeepEnv(planeta=p, aleatorio=False)
        hs = np.linspace(env.h_op_min, env.h_op_max, 60)
        dvs = [_dv_anual_mantenimiento(PLANETAS[p], h) for h in hs]
        ax.plot(hs / 1000.0, dvs, color=COLORES[p], lw=2, label=p.capitalize())
    ax.axhline(DV_OP_ALTO, color="#888", ls="--", lw=1, alpha=0.7)
    ax.axhline(DV_OP_BAJO, color="#888", ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Altitud sobre la superficie (km, escala log)")
    ax.set_ylabel("Δv de mantenimiento (m/s por año, escala log)")
    ax.set_title("Coste de mantener la órbita: cae al subir la altitud (densidad exponencial)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, ncol=2, **LEG)
    guardar(fig, "ia_keep_coste_altitud")
    plt.close(fig)


# ── 4) Ahorro del agente frente a la ingenua, por cuerpo ────────────────────
def fig_ahorro(n=20):
    """Figura ia_keep_ahorro: barras con el ahorro de Δv (%) del agente frente a la ingenua,
    promediado sobre n episodios aleatorios por cuerpo. Mucho ahorro donde el control fino
    importa (Marte) y casi nada donde la física ya fija el Δv (gigantes)."""
    ahorros = []
    for p in CUERPOS:
        m = PPO.load(os.path.join(AQUI, "modelo_keep", p, "best_model"))
        env = KeepEnv(planeta=p, aleatorio=True)
        dv_ag = dv_in = 0.0
        for k in range(n):
            obs, _ = env.reset(seed=1000 + k)
            t = f = False
            inf = {}
            while not (t or f):
                obs, _, t, f, inf = env.step(m.predict(obs, deterministic=True)[0])
            dv_ag += inf.get("dv_acum", 0.0)
            obs, _ = env.reset(seed=1000 + k)
            t = f = False
            inf = {}
            while not (t or f):
                obs, _, t, f, inf = env.step(_ingenua(env, obs))
            dv_in += inf.get("dv_acum", 0.0)
        ahorros.append((1 - dv_ag / dv_in) * 100 if dv_in > 0 else 0)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = [COLORES[p] for p in CUERPOS]
    barras = ax.bar([p.capitalize() for p in CUERPOS], ahorros, color=cols, alpha=0.9)
    for b, a in zip(barras, ahorros):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.5, f"{a:.1f}%",
                ha="center", fontsize=9.5, color="#eee")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("Ahorro de Δv del agente frente a la heurística (%)")
    ax.set_title("El RL aporta según el régimen: mucho donde el control fino importa (Marte),\n"
                 "casi nada donde la física ya fija el Δv (gigantes)")
    ax.grid(True, alpha=0.3, axis="y")
    guardar(fig, "ia_keep_ahorro")
    plt.close(fig)


if __name__ == "__main__":
    print("Generando figuras de mantenimiento orbital en", IMG)
    fig_diente_sierra()
    fig_franjas()
    fig_coste_altitud()
    fig_ahorro()
    print("Hecho.")

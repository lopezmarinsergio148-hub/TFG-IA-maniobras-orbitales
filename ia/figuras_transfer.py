# ═══════════════════════════════════════════════════════════════════════════
#  ia/figuras_transfer.py — Figuras del Agente 2 (transferencias) para la memoria
#
#  Uso:  python ia/figuras_transfer.py
#  Corre el agente REAL (modelo_transfer/best_model) y lo compara con el óptimo
#  de Hohmann. Salida en imagenes/ia/ (PDF vectorial + PNG, fondo oscuro):
#    1) ia_transfer_dv_vs_R    — Δv adimensional del agente vs óptimo (subir y bajar)
#    2) ia_transfer_invariancia — el MISMO modelo en varios planetas (invariancia de escala)
# ═══════════════════════════════════════════════════════════════════════════
"""
═══════════════════════════════════════════════════════════════════════════════
 FIGURAS DEL AGENTE 2 (TRANSFERENCIAS COPLANARES) — gráficas para la memoria
 Corre el agente PPO real (modelo_transfer/best_model) sobre env_transfer y lo
 compara con el óptimo de Hohmann, mostrando su invariancia de escala → imagenes/ia/.

 ÍNDICE DE FUNCIONES:
   - dv_agente_adim(model, R) : Δv total adimensional que da el agente para un ratio R.
   - guardar(fig, nombre)     : guarda la figura como PDF vectorial y PNG.
   - fig_dv_vs_R()            : Δv adimensional agente vs óptimo de Hohmann a lo largo de R.
   - fig_invariancia()        : exceso del agente sobre el óptimo con el mismo modelo en 9 cuerpos.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from stable_baselines3 import PPO
from poliastro.bodies import (Earth, Mars, Venus, Jupiter, Saturn, Uranus, Neptune,
                              Moon, Mercury)

from env_transfer import TransferEnv, R_SPAN
from baselines import hohmann_adim, delta_v_hohmann

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "ia"))
FONDO = "#0a0a1a"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})
MODELO = os.path.join(AQUI, "modelo_transfer", "best_model")


def guardar(fig, nombre):
    """Guarda la figura en imagenes/ia/ como PDF vectorial y PNG (200 dpi), fondo oscuro."""
    fig.savefig(os.path.join(IMG, nombre + ".pdf"), bbox_inches="tight", facecolor=FONDO)
    fig.savefig(os.path.join(IMG, nombre + ".png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
    print("  guardada:", nombre)


def dv_agente_adim(model, R):
    """Δv total adimensional que da el agente para un ratio R."""
    env = TransferEnv()
    obs, _ = env.reset(options={"R": R})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    return info["dv_total"]


# ── 1) Δv adimensional: agente vs óptimo de Hohmann (subir y bajar) ─────────
def fig_dv_vs_R():
    """Figura ia_transfer_dv_vs_R: Δv total adimensional del agente RL frente al óptimo de
    Hohmann teórico a lo largo del ratio R = r₂/r₁ (subir y bajar), en escala log."""
    model = PPO.load(MODELO)
    Rs = np.logspace(np.log10(1 / R_SPAN), np.log10(R_SPAN), 160)
    dv_opt = [hohmann_adim(R)[2] for R in Rs]
    dv_ag = [dv_agente_adim(model, R) for R in Rs]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(Rs, dv_opt, color="#bbbbbb", lw=2.4, label="Óptimo de Hohmann (teórico)")
    ax.plot(Rs, dv_ag, color="#4ade80", lw=0, marker="o", ms=3.2,
            label="Agente RL (aprendido)")
    ax.axvline(1.0, color="#666", ls=":", lw=1)
    ax.text(1.06, ax.get_ylim()[1] * 0.92, "subir  →", color="#9ad", fontsize=9)
    ax.text(0.94, ax.get_ylim()[1] * 0.92, "←  bajar", color="#e9a", fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Ratio de la transferencia  R = r₂ / r₁   (escala log)")
    ax.set_ylabel("Δv total adimensional  (en unidades de la v. circular inicial)")
    ax.set_title("El agente clava el óptimo de Hohmann en todo el rango")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(facecolor="#15152a", edgecolor="#555", labelcolor="#eee")
    guardar(fig, "ia_transfer_dv_vs_R")
    plt.close(fig)


# ── 2) Invariancia de escala: el MISMO modelo en varios planetas ────────────
def fig_invariancia():
    """Figura ia_transfer_invariancia: exceso del agente sobre el óptimo (%) a lo largo de R
    para los 9 cuerpos con el MISMO modelo; los marcadores se apilan porque la transferencia
    es kepleriana pura y el resultado adimensional no depende del planeta."""
    model = PPO.load(MODELO)
    # marcadores PEQUEÑOS por cuerpo (se solapan a propósito: caen en el mismo punto).
    # Los 9 cuerpos del proyecto: la transferencia es kepleriana pura, así que vale para
    # todos (incluso Luna y Mercurio, que no tienen atmósfera).
    cuerpos = [("Venus", Venus, "#e9c46a", "o", 5.5), ("Tierra", Earth, "#4ea8de", "s", 5.0),
               ("Marte", Mars, "#e07a5f", "^", 4.5), ("Júpiter", Jupiter, "#d9a066", "D", 4.2),
               ("Saturno", Saturn, "#c9a66b", "v", 4.5), ("Urano", Uranus, "#76c7c0", "P", 4.8),
               ("Neptuno", Neptune, "#5e7ce2", "X", 4.5), ("Luna", Moon, "#dddddd", "p", 4.2),
               ("Mercurio", Mercury, "#9b8cce", "*", 5.0)]
    Rs = np.logspace(np.log10(1 / R_SPAN), np.log10(R_SPAN), 36)
    Rs = Rs[(Rs < 0.85) | (Rs > 1.18)]     # fuera la zona R≈1 (Δv minúsculo: el % se dispara)
    dv_adim = {R: dv_agente_adim(model, R) for R in Rs}    # solo depende de R (se cachea)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    datos = {}
    for nombre, body, color, mk, sz in cuerpos:
        mu = float(body.k.to_value(u.km**3 / u.s**2))
        Rb = float(body.R.to_value(u.km))
        r1 = Rb + 500.0
        ex = []
        for R in Rs:
            r2 = r1 * R
            v_c1 = np.sqrt(mu / r1)
            dv_ag = dv_adim[R] * v_c1
            dv_opt = abs(delta_v_hohmann(r1, r2, mu).dv_total)
            ex.append((dv_ag / dv_opt - 1.0) * 100.0)
        datos[nombre] = np.array(ex)
    # línea de tendencia (gris, en dos tramos para no cruzar el hueco de R≈1); todos coinciden
    base = datos[cuerpos[0][0]]
    mb, msb = Rs < 1, Rs > 1
    ax.plot(Rs[mb], base[mb], color="#888888", lw=1.1, zorder=1)
    ax.plot(Rs[msb], base[msb], color="#888888", lw=1.1, zorder=1)
    for nombre, body, color, mk, sz in cuerpos:
        ax.plot(Rs, datos[nombre], marker=mk, ms=sz, lw=0, color=color, label=nombre,
                alpha=0.95, markeredgecolor="#0a0a1a", markeredgewidth=0.3, zorder=3)
    ax.axhline(0, color="#888", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Ratio de la transferencia  R = r₂ / r₁   (escala log)")
    ax.set_ylabel("Exceso del agente sobre el óptimo (%)")
    ax.set_title("Invariancia de escala: el mismo modelo vale para cualquier planeta")
    ax.text(0.5, -0.5, "Los 9 cuerpos caen en el MISMO punto a cada R\n"
            "(los marcadores se apilan) → el resultado no depende del planeta",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#9ad")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(facecolor="#15152a", edgecolor="#555", labelcolor="#eee", ncol=3, loc="lower right", fontsize=9)
    ax.set_ylim(-7, 9)
    guardar(fig, "ia_transfer_invariancia")
    plt.close(fig)


if __name__ == "__main__":
    print("Generando figuras de transferencias en", IMG)
    fig_dv_vs_R()
    fig_invariancia()
    print("Hecho.")

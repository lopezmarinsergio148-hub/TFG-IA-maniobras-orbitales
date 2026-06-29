# ═══════════════════════════════════════════════════════════════════════════
#  ia/figuras_orbitas.py — Órbitas dibujadas de las maniobras del agente (Grupo 4)
#
#  Uso:  python ia/figuras_orbitas.py
#  Dibuja la transferencia de Hohmann LEO->GEO con los Δv REALES del agente 1
#  (modelo_hohmann). Salida en imagenes/ia/ (PDF vectorial + PNG, fondo oscuro).
#    1) ia_orbita_hohmann_leo_geo  — LEO + elipse de transferencia + GEO, con los Δv
# ═══════════════════════════════════════════════════════════════════════════

import os

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env_hohmann import HohmannEnv
from env_transfer import TransferEnv
from baselines import R_TIERRA, MU_TIERRA, H_LEO, H_GEO, hohmann_leo_geo, delta_v_hohmann

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "ia"))
FONDO = "#0a0a1a"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})


def guardar(fig, nombre):
    fig.savefig(os.path.join(IMG, nombre + ".pdf"), bbox_inches="tight", facecolor=FONDO)
    fig.savefig(os.path.join(IMG, nombre + ".png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
    print("  guardada:", nombre)


def fig_hohmann_leo_geo():
    # --- Δv del AGENTE (agente 1, especialista LEO->GEO) ---
    model = PPO.load(os.path.join(AQUI, "modelo_hohmann", "best_model"))
    env = HohmannEnv()
    r1, r2 = R_TIERRA + H_LEO, R_TIERRA + H_GEO
    obs, _ = env.reset(options={"r1": r1, "r2": r2})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    dv1, dv2 = info["dv1"], info["dv2"]
    opt = hohmann_leo_geo()

    th = np.linspace(0, 2 * np.pi, 400)
    a_t = 0.5 * (r1 + r2)                       # semieje de la elipse de transferencia
    e_t = (r2 - r1) / (r2 + r1)
    th_tr = np.linspace(0, np.pi, 200)          # media elipse: de perigeo (+x) a apogeo (-x)
    r_tr = a_t * (1 - e_t**2) / (1 + e_t * np.cos(th_tr))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # Tierra
    ax.fill(R_TIERRA * np.cos(th), R_TIERRA * np.sin(th), color="#4ea8de", alpha=0.9, zorder=6)
    ax.text(0, 0, "Tierra", ha="center", va="center", color="#fff", fontsize=9, zorder=7)
    # órbitas LEO y GEO (circulares)
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), color="#4ade80", lw=2, label=f"LEO ({H_LEO:.0f} km)")
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), color="#e9c46a", lw=2, label=f"GEO ({H_GEO:.0f} km)")
    # elipse de transferencia (media, el camino real)
    ax.plot(r_tr * np.cos(th_tr), r_tr * np.sin(th_tr), color="#f87171", lw=2.2, ls="--",
            label="Elipse de transferencia")
    # impulsos del agente (perigeo = +x ; apogeo = -x)
    ax.plot(r1, 0, "o", color="#fff", ms=8, zorder=8)
    ax.plot(-r2, 0, "o", color="#fff", ms=8, zorder=8)
    ax.annotate(f"Δv₁ = {dv1:.3f} km/s\n(inyección)", xy=(r1, 0), xytext=(r1 * 0.5, -r2 * 0.42),
                color="#4ade80", fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.4))
    ax.annotate(f"Δv₂ = {dv2:.3f} km/s\n(circularización)", xy=(-r2, 0), xytext=(-r2 * 0.55, r2 * 0.4),
                color="#e9c46a", fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#e9c46a", lw=1.4))
    # cartel con el resumen agente vs óptimo
    txt = (f"Agente:  Δv total = {dv1 + dv2:.4f} km/s\n"
           f"Óptimo:  Δv total = {opt.dv_total:.4f} km/s\n"
           f"exceso = {((dv1 + dv2) / opt.dv_total - 1) * 100:+.2f} %")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color="#eee", bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555"))

    ax.set_aspect("equal")
    ax.set_xlabel("x en el plano orbital (km)")
    ax.set_ylabel("y en el plano orbital (km)")
    ax.set_title("Transferencia de Hohmann LEO→GEO con los Δv del agente")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", facecolor="#15152a", edgecolor="#555", labelcolor="#eee", fontsize=9)
    guardar(fig, "ia_orbita_hohmann_leo_geo")
    plt.close(fig)


# ── 2) Transferencia de BAJADA GEO->LEO (agente 2, frenando) ────────────────
def fig_transfer_bajada():
    model = PPO.load(os.path.join(AQUI, "modelo_transfer", "best_model"))
    r1, r2 = R_TIERRA + H_GEO, R_TIERRA + H_LEO     # parte de GEO, baja a LEO
    R_ratio = r2 / r1                               # < 1 (bajar)
    v_c1 = np.sqrt(MU_TIERRA / r1)
    env = TransferEnv()
    obs, _ = env.reset(options={"R": R_ratio})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    dv1 = abs(info["dv1"]) * v_c1                    # km/s (magnitud; los impulsos son retrógrados)
    dv2 = abs(info["dv2"]) * v_c1
    dv_opt = abs(delta_v_hohmann(r1, r2, MU_TIERRA).dv_total)

    th = np.linspace(0, 2 * np.pi, 400)
    a_t = 0.5 * (r1 + r2)
    e_t = (r1 - r2) / (r1 + r2)
    # arco INFERIOR: apogeo (-x, GEO) -> perigeo (+x, LEO). Mismo sentido de giro que la
    # subida (antihorario), por eso la bajada recorre la mitad CONTRARIA de la elipse.
    th_tr = np.linspace(np.pi, 2 * np.pi, 200)
    r_tr = a_t * (1 - e_t**2) / (1 + e_t * np.cos(th_tr))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.fill(R_TIERRA * np.cos(th), R_TIERRA * np.sin(th), color="#4ea8de", alpha=0.9, zorder=6)
    ax.text(0, 0, "Tierra", ha="center", va="center", color="#fff", fontsize=9, zorder=7)
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), color="#e9c46a", lw=2, label=f"GEO ({H_GEO:.0f} km, inicio)")
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), color="#4ade80", lw=2, label=f"LEO ({H_LEO:.0f} km, objetivo)")
    ax.plot(r_tr * np.cos(th_tr), r_tr * np.sin(th_tr), color="#f87171", lw=2.2, ls="--",
            label="Elipse de transferencia")
    # impulsos (frenado): Δv1 en GEO (apogeo, -x) ; Δv2 en LEO (perigeo, +x)
    ax.plot(-r1, 0, "o", color="#fff", ms=8, zorder=8)
    ax.plot(r2, 0, "o", color="#fff", ms=8, zorder=8)
    ax.annotate(f"Δv₁ = {dv1:.3f} km/s\n(frenado en GEO)", xy=(-r1, 0), xytext=(-r1 * 0.55, r1 * 0.42),
                color="#e9c46a", fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#e9c46a", lw=1.4))
    ax.annotate(f"Δv₂ = {dv2:.3f} km/s\n(frenado en LEO)", xy=(r2, 0), xytext=(r1 * 0.5, r1 * 0.42),
                color="#4ade80", fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.4))
    txt = (f"Agente:  Δv total = {dv1 + dv2:.4f} km/s\n"
           f"Óptimo:  Δv total = {dv_opt:.4f} km/s\n"
           f"exceso = {((dv1 + dv2) / dv_opt - 1) * 100:+.2f} %")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color="#eee", bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555"))

    ax.set_aspect("equal")
    ax.set_xlabel("x en el plano orbital (km)")
    ax.set_ylabel("y en el plano orbital (km)")
    ax.set_title("Transferencia de bajada GEO→LEO con los Δv del agente (frenando)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", facecolor="#15152a", edgecolor="#555", labelcolor="#eee", fontsize=9)
    guardar(fig, "ia_orbita_bajada_geo_leo")
    plt.close(fig)


if __name__ == "__main__":
    print("Generando figuras de órbitas en", IMG)
    fig_hohmann_leo_geo()
    fig_transfer_bajada()
    print("Hecho.")

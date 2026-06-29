# ═══════════════════════════════════════════════════════════════════════════
#  ia/figuras_aerofrenado.py — Figuras del Agente 3 (aerofrenado) para la memoria
#
#  Uso:  python ia/figuras_aerofrenado.py
#  TODO se genera corriendo los AGENTES REALES (modelo_drag/<planeta>/best_model)
#  dentro del entorno env_drag.py (física King-Hele + atmósferas validadas). No
#  hay datos inventados. La "estrategia tonta" es un baseline de comparación
#  (perigeo fijo conservador, sin aprender).
#
#  Salida en imagenes/ia/ (PDF vectorial + PNG, fondo oscuro del proyecto):
#    1) ia_aero_apogeo_pasada           — apogeo vs nº de pasada (7 planetas, 1 fig)
#    2) ia_aero_vs_tonta_<planeta>      — agente RL vs estrategia tonta (1 por planeta)
#    3) ia_aero_orbita_<planeta>        — la órbita circularizándose (1 por planeta)
# ═══════════════════════════════════════════════════════════════════════════

import os

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env_drag import AeroBrakingEnv

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "ia"))
FONDO = "#0a0a1a"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})
PLANETAS_FIG = ["venus", "tierra", "marte", "jupiter", "saturno", "urano", "neptuno"]
COLORES = {"marte": "#e07a5f", "tierra": "#4ea8de", "venus": "#e9c46a",
           "saturno": "#c9a66b", "urano": "#76c7c0", "neptuno": "#5e7ce2",
           "jupiter": "#d9a066"}
LEG = dict(facecolor="#15152a", edgecolor="#555", labelcolor="#eee", framealpha=0.93)


def correr(planeta, modelo=None, accion_fija=None):
    """Corre un episodio (agente o acción fija) y registra apogeo/perigeo por pasada."""
    env = AeroBrakingEnv(planeta=planeta, aleatorio=False)   # escenario fijo reproducible
    obs, _ = env.reset(seed=0)
    apos = [env.h_apo / 1000]
    pers = [env.h_per / 1000]
    term = trunc = False
    info = {}
    while not (term or trunc):
        accion = accion_fija if accion_fija is not None else modelo.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(accion)
        apos.append(info["h_apo_km"])
        pers.append(info["h_per_km"])
    return np.array(apos), np.array(pers), info, env


def guardar(fig, nombre):
    fig.savefig(os.path.join(IMG, nombre + ".pdf"), bbox_inches="tight", facecolor=FONDO)
    fig.savefig(os.path.join(IMG, nombre + ".png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
    print("  guardada:", nombre)


# ── 1) Apogeo vs pasada (los 7 planetas en una sola figura) ─────────────────
def fig_apogeo_pasada():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ratio_obj = None
    for p in PLANETAS_FIG:
        m = PPO.load(os.path.join(AQUI, "modelo_drag", p, "best_model"))
        apos, _, info, env = correr(p, modelo=m)
        ratio_obj = (env.h_apo_objetivo / 1000.0) / apos[0]
        ax.plot(np.arange(len(apos)), apos / apos[0], color=COLORES[p], lw=2,
                label=f"{p.capitalize()} ({len(apos)-1} pasadas)")
        ax.plot(len(apos) - 1, apos[-1] / apos[0], "o", color=COLORES[p], ms=5)
    ax.axhline(ratio_obj, color="#bbbbbb", ls="--", lw=1.2,
               label=f"Apogeo objetivo (= {ratio_obj*100:.0f}% del inicial)")
    ax.set_xlabel("Número de pasada por el perigeo")
    ax.set_ylabel("Apogeo normalizado  (apogeo / apogeo inicial)")
    ax.set_title("Aerofrenado: descenso del apogeo pasada a pasada (7 cuerpos)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, **LEG)
    ax.set_ylim(0, 1.02)
    guardar(fig, "ia_aero_apogeo_pasada")
    plt.close(fig)


# ── 2) Agente RL vs estrategia tonta — UNA figura por planeta ───────────────
def fig_vs_tonta_individual():
    for p in PLANETAS_FIG:
        m = PPO.load(os.path.join(AQUI, "modelo_drag", p, "best_model"))
        apos_ag, pers_ag, info_ag, env = correr(p, modelo=m)
        h_tonto = env.H_PER_MIN + 0.85 * (env.H_PER_MAX - env.H_PER_MIN)
        a_tonta = np.array([(h_tonto - env.H_PER_MIN) / (env.H_PER_MAX - env.H_PER_MIN) * 2 - 1],
                           dtype=np.float32)
        apos_t, pers_t, info_t, _ = correr(p, accion_fija=a_tonta)
        obj = env.h_apo_objetivo / 1000

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(len(apos_ag)), apos_ag, color="#4ade80", lw=2.2,
                label=f"Agente RL (perigeo ~{np.mean(pers_ag[1:]):.0f} km) → {len(apos_ag)-1} pasadas")
        res_t = f"{len(apos_t)-1} pasadas" if info_t.get("exito") else f"no llega (>{len(apos_t)-1})"
        ax.plot(np.arange(len(apos_t)), apos_t, color="#f87171", lw=2.2, ls="--",
                label=f"Estrategia tonta (perigeo ~{np.mean(pers_t[1:]):.0f} km) → {res_t}")
        ax.axhline(obj, color="#bbbbbb", ls=":", lw=1.3, label=f"Apogeo objetivo = {obj:.0f} km")
        ax.set_xlabel("Número de pasada por el perigeo")
        ax.set_ylabel("Apogeo (km)")
        ax.set_title(f"{p.capitalize()}: agente RL vs estrategia tonta (perigeo alto fijo)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9.5, **LEG)
        guardar(fig, f"ia_aero_vs_tonta_{p}")
        plt.close(fig)


# ── 3) La órbita circularizándose — UNA figura por planeta ──────────────────
#  Vista en el plano orbital, planeta en el centro (0,0). Los ejes x/y son
#  posición respecto al centro; el signo solo indica el lado, no distancia negativa.
def fig_orbita_individual():
    th = np.linspace(0, 2 * np.pi, 320)
    for p in PLANETAS_FIG:
        m = PPO.load(os.path.join(AQUI, "modelo_drag", p, "best_model"))
        apos, pers, info, env = correr(p, modelo=m)
        R_km = env.R / 1000

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.fill(R_km * np.cos(th), R_km * np.sin(th), color=COLORES[p], alpha=0.9,
                zorder=5, label=p.capitalize())
        idxs = np.linspace(0, len(apos) - 1, 7).astype(int)
        cmap = plt.cm.cool(np.linspace(0.05, 0.95, len(idxs)))
        for k, (c, i) in enumerate(zip(cmap, idxs)):
            r_a = R_km + apos[i]
            r_p = R_km + pers[i]
            a = 0.5 * (r_a + r_p)
            e = (r_a - r_p) / (r_a + r_p)
            r = a * (1 - e**2) / (1 + e * np.cos(th))
            etiq = f"1ª órbita (apo {apos[i]:.0f} km)" if k == 0 else \
                   (f"última (apo {apos[i]:.0f} km)" if k == len(idxs) - 1 else None)
            ax.plot(r * np.cos(th), r * np.sin(th), color=c, lw=1.6, label=etiq)
        ax.set_aspect("equal")
        ax.set_title(f"{p.capitalize()}: la órbita se circulariza ({len(apos)-1} pasadas)")
        ax.set_xlabel("posición x en el plano orbital (km)")
        ax.set_ylabel("posición y en el plano orbital (km)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, loc="upper right", **LEG)
        guardar(fig, f"ia_aero_orbita_{p}")
        plt.close(fig)


if __name__ == "__main__":
    print("Generando figuras de aerofrenado en", IMG)
    fig_apogeo_pasada()
    fig_vs_tonta_individual()
    fig_orbita_individual()
    print("Hecho.")

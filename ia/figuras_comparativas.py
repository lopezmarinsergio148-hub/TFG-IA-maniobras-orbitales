# ═══════════════════════════════════════════════════════════════════════════
#  ia/figuras_comparativas.py — Comparativas multi-planeta del aerofrenado (Grupo 2)
#
#  Uso:  python ia/figuras_comparativas.py
#  Corre los 7 agentes reales (escenario fijo) y genera, a imagenes/ia/:
#    1) ia_comp_presion   — presión dinámica de operación vs límite (cuánto apura)
#    2) ia_comp_tiempo     — duración real (días) + nº de pasadas, con ref. MRO
#    3) ia_comp_margen     — margen de seguridad del perigeo (% sobre el crítico)
#  Todo con datos reales (sin inventar): física King-Hele + atmósferas validadas.
# ═══════════════════════════════════════════════════════════════════════════
"""
═══════════════════════════════════════════════════════════════════════════════
 COMPARATIVAS MULTI-PLANETA DEL AEROFRENADO — gráficas de barras para la memoria
 Corre los 7 agentes PPO reales (escenario fijo) sobre env_drag y genera figuras
 de barras comparando presión dinámica, duración y margen de seguridad → imagenes/ia/.

 ÍNDICE DE FUNCIONES:
   - metricas(p)                          : corre el agente y devuelve sus métricas comparativas.
   - guardar(fig, nombre)                 : guarda la figura como PDF vectorial y PNG.
   - etiqueta_barras(ax, barras, valores) : escribe el valor numérico encima de cada barra.
   - main()                               : genera las 3 figuras (presión, tiempo, margen).
═══════════════════════════════════════════════════════════════════════════════
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env_drag import AeroBrakingEnv, Q_MAX, _altura_para_q

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "ia"))
FONDO = "#0a0a1a"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})
PLANETAS = ["venus", "tierra", "marte", "jupiter", "saturno", "urano", "neptuno"]
COLORES = {"marte": "#e07a5f", "tierra": "#4ea8de", "venus": "#e9c46a",
           "saturno": "#c9a66b", "urano": "#76c7c0", "neptuno": "#5e7ce2",
           "jupiter": "#d9a066"}


def metricas(p):
    """Corre el agente (escenario fijo) y devuelve las métricas comparativas."""
    env = AeroBrakingEnv(planeta=p, aleatorio=False)
    m = PPO.load(os.path.join(AQUI, "modelo_drag", p, "best_model"))
    obs, _ = env.reset(seed=0)
    R, mu = env.R / 1000.0, env.MU / 1e9
    apo_ini = env.h_apo
    apos, pers, qs = [env.h_apo / 1000], [env.h_per / 1000], []
    term = trunc = False
    while not (term or trunc):
        a, _ = m.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        apos.append(info["h_apo_km"]); pers.append(info["h_per_km"]); qs.append(info.get("q_din", 0))
    apos, pers = np.array(apos), np.array(pers)
    semejes = (2 * R + apos + pers) / 2.0
    tiempo_d = (2 * np.pi * np.sqrt(semejes**3 / mu)).sum() / 86400.0
    h_crit = _altura_para_q(env.planeta, Q_MAX, apo_ini) / 1000.0
    return {"perigeo_op": float(np.mean(pers[1:])), "q_op": float(max(qs)),
            "h_crit": h_crit, "pasadas": len(apos) - 1, "tiempo_d": tiempo_d}


def guardar(fig, nombre):
    """Guarda la figura en imagenes/ia/ como PDF vectorial y PNG (200 dpi), fondo oscuro."""
    fig.savefig(os.path.join(IMG, nombre + ".pdf"), bbox_inches="tight", facecolor=FONDO)
    fig.savefig(os.path.join(IMG, nombre + ".png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
    print("  guardada:", nombre)


def etiqueta_barras(ax, barras, valores, fmt="{:.2f}", dy=0):
    """Escribe el valor numérico (formato fmt) centrado encima de cada barra."""
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=9, color="#eee")


def main():
    """Genera las 3 comparativas de barras (una por métrica): ia_comp_presion (presión
    dinámica vs límite), ia_comp_tiempo (duración y pasadas) e ia_comp_margen (margen del perigeo)."""
    M = {p: metricas(p) for p in PLANETAS}
    nombres = [p.capitalize() for p in PLANETAS]
    cols = [COLORES[p] for p in PLANETAS]
    x = np.arange(len(PLANETAS))

    # ── 1) Presión dinámica de operación vs límite ──────────────────────────
    q_op = [M[p]["q_op"] for p in PLANETAS]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    barras = ax.bar(x, q_op, color=cols, edgecolor="#222", width=0.62)
    etiqueta_barras(ax, barras, q_op, fmt="{:.2f}", dy=0.008)
    ax.axhline(Q_MAX, color="#f87171", ls="--", lw=1.8, label=f"Límite de la nave = {Q_MAX} N/m²")
    ax.set_xticks(x); ax.set_xticklabels(nombres)
    ax.set_ylabel("Presión dinámica de operación  q = ½ρv²  (N/m²)")
    ax.set_title("Cuánto apura cada agente frente al límite estructural")
    ax.set_ylim(0, Q_MAX * 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(facecolor="#15152a", edgecolor="#555", labelcolor="#eee")
    guardar(fig, "ia_comp_presion"); plt.close(fig)

    # ── 2) Duración real (días) + nº de pasadas ─────────────────────────────
    tiempo = [M[p]["tiempo_d"] for p in PLANETAS]
    pasadas = [M[p]["pasadas"] for p in PLANETAS]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    barras = ax.bar(x, tiempo, color=cols, edgecolor="#222", width=0.62)
    for b, t, n in zip(barras, tiempo, pasadas):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{t:.0f} d\n({n} pasadas)", ha="center", va="bottom", fontsize=8.5, color="#eee")
    ax.set_xticks(x); ax.set_xticklabels(nombres)
    ax.set_ylabel("Duración total del aerofrenado (días)")
    ax.set_title("¿Cuánto dura el aerofrenado? (escenario de referencia)")
    ax.set_ylim(0, max(tiempo) * 1.22)
    ax.grid(True, axis="y", alpha=0.3)
    guardar(fig, "ia_comp_tiempo"); plt.close(fig)

    # ── 3) Margen de seguridad del perigeo (% sobre el crítico) ─────────────
    margen = [(M[p]["perigeo_op"] - M[p]["h_crit"]) / M[p]["h_crit"] * 100 for p in PLANETAS]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    barras = ax.bar(x, margen, color=cols, edgecolor="#222", width=0.62)
    for b, p, mg in zip(barras, PLANETAS, margen):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{mg:.0f}%\nperigeo op {M[p]['perigeo_op']:.0f} km\ncrítico {M[p]['h_crit']:.0f} km",
                ha="center", va="bottom", fontsize=7.5, color="#eee")
    ax.set_xticks(x); ax.set_xticklabels(nombres)
    ax.set_ylabel("Margen del perigeo de operación sobre el crítico (%)")
    ax.set_title("Margen de seguridad: cuánto se aleja cada agente del límite")
    ax.set_ylim(0, max(margen) * 1.35)
    ax.grid(True, axis="y", alpha=0.3)
    guardar(fig, "ia_comp_margen"); plt.close(fig)


if __name__ == "__main__":
    print("Generando comparativas en", IMG)
    main()
    print("Hecho.")

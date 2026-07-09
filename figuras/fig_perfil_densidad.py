# Genera la figura del perfil de densidad rho(h) del modelo de capas (para 4.2.2).
# Escala semilog: cada capa exponencial es un segmento recto; los puntos marcan
# los limites de capa (densidades de referencia rho_base = observaciones).
# Salida: imagenes/fig_perfil_densidad.pdf y .png
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_importantes"))
from Densidades_atmosferica_optimizado import PLANETAS

BG = "#0a0a1a"
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

XMAX = 700.0  # km
cuerpos = [("tierra", "#4da6ff"), ("marte", "#ff6b4d"), ("venus", "#ffd24d"),
           ("jupiter", "#e0a060"), ("neptuno", "#9b6bff")]

for clave, col in cuerpos:
    pl = PLANETAS[clave]
    # Hasta XMAX para TODOS: por encima de la capa superior, el modelo extrapola
    # esa última capa (exactamente lo que hace get_rho en la simulación).
    hs = np.linspace(0, XMAX, 600)
    rhos = [pl.get_rho(h * 1000.0, 0)[0] for h in hs]
    ax.semilogy(hs, rhos, color=col, lw=2.2, label=pl.nombre, zorder=3)
    # marcar los límites de capa (rho_base) dentro del rango
    hb = [c.h_min_km for c in pl.capas if c.h_min_km <= XMAX]
    rb = [c.rho_base_kg_m3 for c in pl.capas if c.h_min_km <= XMAX]
    ax.scatter(hb, rb, color=col, s=28, zorder=4, edgecolor=BG, linewidth=0.6)

ax.set_xlabel("Altitud sobre el nivel de referencia, $h$ (km)", color="white", fontsize=12)
ax.set_ylabel(r"Densidad $\rho$ (kg/m$^3$) — escala logarítmica", color="white", fontsize=12)
ax.set_xlim(0, XMAX)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#444")
ax.grid(True, which="both", color="#333", lw=0.5, alpha=0.5)
leg = ax.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=11,
                title="Cuerpo", title_fontsize=11, loc="upper right")
leg.get_title().set_color("white")
# nota de los puntos
ax.text(0.015, 0.03, "Los puntos marcan los límites de capa ($\\rho_{base}$, las observaciones);\n"
        "cada tramo recto es una capa exponencial.",
        transform=ax.transAxes, color="#aaa", fontsize=9, va="bottom")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "imagenes")
fig.savefig(os.path.join(out, "fig_perfil_densidad.pdf"), facecolor=BG, bbox_inches="tight")
fig.savefig(os.path.join(out, "fig_perfil_densidad.png"), dpi=130, facecolor=BG, bbox_inches="tight")
print("Figura escrita en imagenes/fig_perfil_densidad.pdf / .png")

# Genera la figura explicativa de la ATRACCION GRAVITATORIA MUTUA (problema de los
# dos cuerpos), para acompanar la ecuacion (4.1) del capitulo de Fundamentos.
# Minimalista: las dos masas y las fuerzas iguales y opuestas (ley de Newton).
# Salida: fig_dos_cuerpos.pdf (vectorial, para LaTeX) y .png (vista previa).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BG = "#0a0a1a"
plt.rcParams.update({"font.size": 13, "text.color": "white",
                     "axes.edgecolor": "#444"})

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


def flecha(p0, p1, color, lw=3.4, scale=24):
    ax.add_patch(FancyArrowPatch(p0, p1, color=color, lw=lw,
                 arrowstyle="-|>", mutation_scale=scale, zorder=7))


# ── Posiciones de los dos cuerpos ────────────────────────────────────────────
M = np.array([-4.6, 0.0]); RM = 1.25      # cuerpo central (grande)
m = np.array([5.0, 0.0]);  Rm = 0.45      # satelite (pequeno)

# Cuerpo central M
ax.add_patch(plt.Circle(M, RM, facecolor="#1f3a5f", edgecolor="#cfd8e3", lw=1.8, zorder=5))
ax.plot(*M, "o", color="white", ms=4, zorder=6)
ax.text(M[0], M[1] - RM - 0.55, r"$M$  (cuerpo central)", color="#cfd8e3",
        fontsize=13, ha="center", va="top")

# Satelite m
ax.add_patch(plt.Circle(m, Rm, facecolor="#ffe082", edgecolor="#fff3c4", lw=1.5, zorder=5))
ax.text(m[0], m[1] - Rm - 0.55, "$m$  (satélite)", color="#ffcc33",
        fontsize=13, ha="center", va="top")

# ── Vector de posicion relativa r (de M a m), arriba ─────────────────────────
yc = 2.15
for x in (M[0], m[0]):
    ax.plot([x, x], [0.35, yc], color="#8899aa", lw=0.9, ls=":", zorder=1)
flecha((M[0], yc), (m[0], yc), "#33d6ff", lw=1.8, scale=18)
ax.text((M[0] + m[0]) / 2, yc + 0.2, r"$\vec{r}$", color="#33d6ff", fontsize=16,
        ha="center", va="bottom")
ax.text((M[0] + m[0]) / 2, yc - 0.28, r"$r=|\vec{r}|$", color="#8899aa",
        fontsize=11, ha="center", va="top", style="italic")

# ── Fuerzas gravitatorias: iguales y opuestas, dirigidas a lo largo de la recta ─
L = 2.3                                    # misma longitud -> mismo modulo
# Sobre m: apunta hacia M (a la izquierda)
flecha(m - np.array([Rm + 0.05, 0]), m - np.array([Rm + 0.05 + L, 0]), "#ffd700", lw=4.2)
ax.text(m[0] - Rm - L / 2, 0.5, r"$\vec{F}$", color="#ffd700", fontsize=16,
        ha="center", va="bottom")
# Sobre M: apunta hacia m (a la derecha)
flecha(M + np.array([RM + 0.05, 0]), M + np.array([RM + 0.05 + L, 0]), "#ffd700", lw=4.2)
ax.text(M[0] + RM + L / 2, 0.5, r"$\vec{F}$", color="#ffd700", fontsize=16,
        ha="center", va="bottom")

# ── Caja con la ley de Newton ────────────────────────────────────────────────
ax.text(0.0, -3.0,
        r"$F = G\,\frac{M\,m}{r^{2}}$" + "\n"
        "fuerzas de igual módulo y sentido opuesto (tercera ley de Newton)",
        color="#ffe082", fontsize=13, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#1a1a08", ec="#ffd700", lw=1.4))

ax.set_title("Atracción gravitatoria mutua en el problema de los dos cuerpos",
             color="white", fontsize=15, pad=12)

ax.set_xlim(-8.5, 9.0); ax.set_ylim(-5.0, 3.2)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

plt.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "imagenes"))   # guarda en imagenes/ (donde apunta el LaTeX)
fig.savefig(os.path.join(OUT, "fig_dos_cuerpos.pdf"), facecolor=BG, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_dos_cuerpos.png"), dpi=140, facecolor=BG, bbox_inches="tight")
print("Figura escrita en imagenes/: fig_dos_cuerpos.pdf / .png")

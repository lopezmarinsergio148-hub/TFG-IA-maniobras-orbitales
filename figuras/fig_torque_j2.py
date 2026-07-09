# Genera la figura explicativa del torque de J2 y la precesión nodal.
# Salida: fig_torque_j2.pdf (vectorial, para LaTeX) y .png (vista previa).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

BG = "#0a0a1a"
plt.rcParams.update({"font.size": 12, "text.color": "white",
                     "axes.edgecolor": "#444"})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor(BG)

# ════════════════════ PANEL IZQUIERDO: origen del torque ════════════════════
ax = axL
ax.set_facecolor(BG)
ax.set_title("①  ¿De dónde sale el torque de $J_2$?", color="white", fontsize=15, pad=14)

# Planeta achatado + abultamiento ecuatorial ("panza")
ax.add_patch(Ellipse((0, 0), 8.4, 5.2, facecolor="#1f3a5f",
                     edgecolor="#cfd8e3", lw=1.6, zorder=1))
for xseg in [(3.0, 4.5), (-4.5, -3.0)]:
    ax.plot(xseg, [0, 0], color="#ff3b3b", lw=9, solid_capstyle="round", zorder=2)
ax.plot([-7.2, 9.2], [0, 0], color="#aa4444", lw=1, ls="--", alpha=0.6, zorder=0)
ax.text(-6.8, 0.35, "ECUADOR", color="#ff7777", fontsize=10, style="italic")
ax.text(-7.3, -3.55, 'segmentos rojos = la "panza" (abultamiento ecuatorial)',
        color="#ff7777", fontsize=9.5, style="italic", ha="left")
ax.plot(0, 0, "o", color="white", ms=8, zorder=5)
ax.text(0.15, -0.45, "centro", color="white", fontsize=10)

# Satélite
S = np.array([5.6, 4.0])
ax.plot(*S, "o", color="#ffe082", ms=15, zorder=6)
ax.text(S[0] + 0.15, S[1] + 0.45, "SATÉLITE", color="#ffcc33",
        fontsize=12, fontweight="bold")

# Vectores de fuerza (desde el satélite)
u_centro = (np.array([0, 0]) - S); u_centro = u_centro / np.linalg.norm(u_centro)
F_esf = 2.3 * u_centro                       # gravedad de la esfera -> al centro
F_an  = np.array([0.0, -1.25])               # tirón extra de la panza -> al ecuador
F_tot = F_esf + F_an                         # resultante

def flecha(ax, p0, vec, color, lw=3.2):
    ax.add_patch(FancyArrowPatch(p0, p0 + vec, color=color, lw=lw,
                 arrowstyle="-|>", mutation_scale=22, zorder=7))

flecha(ax, S, F_esf, "#33d6ff")    # cian  - F_esfera
flecha(ax, S, F_an,  "#ff5db1")    # rosa  - F_anillo
flecha(ax, S, F_tot, "#ffd700", lw=4.2)  # oro - F_total (resultante)

# Prolongación de F_total hasta cruzar la zona del centro (muestra que falla)
t_cross = (S[1]) / (-F_tot[1])
x_cross = S[0] + t_cross * F_tot[0]
ax.plot([S[0], x_cross], [S[1], 0.0], color="#ffd700", ls=":", lw=1.2, alpha=0.7, zorder=4)
ax.plot(x_cross, 0.0, "x", color="#d4b106", ms=13, mew=3, zorder=6)
ax.annotate("$F_{tot}$ apunta aquí\n(no al centro)", (x_cross, -0.15),
            color="#d4b106", fontsize=9.5, ha="center", va="top", style="italic")

# Nota del torque (caja)
ax.text(-6.6, 6.2,
        "Como $F_{tot}$ NO pasa por el centro,\n"
        "su brazo de palanca produce un\nTORQUE sobre la órbita.",
        color="#ffe082", fontsize=11.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="#1a1a08", ec="#ffd700", lw=1.4))

# Leyenda de fuerzas (clara, sin amontonar junto a las flechas)
leyenda = [
    Line2D([0], [0], color="#33d6ff", lw=3.2, label="$F_{esfera}$: gravedad de la esfera (al centro)"),
    Line2D([0], [0], color="#ff5db1", lw=3.2, label='$F_{anillo}$: tirón extra de la "panza" (al ecuador)'),
    Line2D([0], [0], color="#ffd700", lw=4.2, label="$F_{total}=F_{esfera}+F_{anillo}$ (resultante)"),
]
ax.legend(handles=leyenda, loc="lower right", facecolor="#111", edgecolor="#555",
          labelcolor="white", fontsize=9.5, framealpha=0.95)

ax.set_xlim(-7.5, 9.5); ax.set_ylim(-4, 7.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_edgecolor("#444")

# ════════════════════ PANEL DERECHO: precesión del plano ════════════════════
ax = axR
ax.set_facecolor(BG)
ax.set_title("②  Consecuencia del torque: el plano orbital PRECESA",
             color="white", fontsize=15, pad=14)

# Planeta
ax.add_patch(plt.Circle((0, 0), 1.0, facecolor="#1f3a5f", edgecolor="#cfd8e3", lw=1.4, zorder=3))
# Plano ecuatorial
ax.plot([-8.5, 8.5], [0, 0], color="#ff5555", ls="--", lw=1.4, zorder=1)
ax.text(7.2, 0.35, "plano ecuatorial", color="#ff7777", fontsize=10, style="italic")

# Planos orbitales (vistos de canto) en distintos instantes -> ángulos que decrecen
tiempos = [(0, 62, "#00ff66"), (10, 47, "#22d3ee"), (20, 32, "#ffd700"),
           (30, 20, "#ff9000"), (60, 6, "#ff6680")]
L = 7.3
for t, ang, col in tiempos:
    a = np.radians(ang)
    dx, dy = L*np.cos(a), L*np.sin(a)
    ax.plot([-dx, dx], [-dy, dy], color=col, lw=2.6,
            label=f"plano orbital  t = {t} d", zorder=2)

ax.legend(loc="upper right", facecolor="#111", edgecolor="#555",
          labelcolor="white", fontsize=10, framealpha=0.95)

# Caja "precesión"
ax.text(-8.0, 6.6, "PRECESIÓN\ndel plano orbital\n(como una peonza)",
        color="#33d6ff", fontsize=12.5, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#06141a", ec="#33d6ff", lw=1.6))

# Flecha curva indicando el giro
ax.add_patch(FancyArrowPatch((5.2, 4.6), (6.6, 2.3), connectionstyle="arc3,rad=-0.45",
             color="#cfd8e3", lw=2.0, arrowstyle="-|>", mutation_scale=20, zorder=4))

ax.text(0, -8.7, "El plano orbital, visto de canto, gira con el tiempo:\n"
        "es la regresión nodal (el RAAN cambia secularmente).",
        color="#cfd8e3", fontsize=10.5, ha="center", style="italic")

ax.set_xlim(-9, 9); ax.set_ylim(-9.5, 8)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_edgecolor("#444")

plt.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(HERE, "fig_torque_j2.pdf"), facecolor=BG, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_torque_j2.png"), dpi=130, facecolor=BG, bbox_inches="tight")
print("Figura escrita: fig_torque_j2.pdf / .png")

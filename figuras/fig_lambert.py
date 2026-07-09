# Genera la figura del PROBLEMA DE LAMBERT para el capitulo de Fundamentos:
# dos posiciones r1, r2 y un tiempo de vuelo dado -> orbita de transferencia, con las
# dos familias de solucion (tipo I < 180 grados y tipo II > 180 grados).
# Guarda en imagenes/ (donde apunta el LaTeX). PDF vectorial + PNG.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch, Arc
from matplotlib.lines import Line2D

BG = "#0a0a1a"
plt.rcParams.update({"font.size": 12, "text.color": "white", "axes.edgecolor": "#444"})
C_I = "#33d6ff"    # tipo I  (< 180 grados)
C_II = "#ff6ec7"   # tipo II (> 180 grados)

fig, ax = plt.subplots(figsize=(10, 8.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)


def pol(ang, r):
    a = np.radians(ang)
    return np.array([r * np.cos(a), r * np.sin(a)])


def flecha(p0, p1, color, lw=2.6, scale=20):
    ax.add_patch(FancyArrowPatch(p0, p1, color=color, lw=lw, arrowstyle="-|>",
                 mutation_scale=scale, zorder=8))


# ── Sol (foco) y las dos posiciones ──────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), 0.09, facecolor="#ffcf33", edgecolor="#fff3c4", lw=1.5, zorder=6))
ax.text(0.14, -0.03, "Sol (foco)", color="#ffe082", fontsize=11, ha="left", va="center")

r1 = pol(205, 1.05)
r2 = pol(340, 1.35)
for P, col in [(r1, "#9db8ff"), (r2, "#ffab9c")]:
    ax.plot([0, P[0]], [0, P[1]], color="#667788", lw=1.1, ls="--", zorder=2)
    ax.plot(*P, "o", color=col, ms=11, zorder=7, markeredgecolor="white", markeredgewidth=0.8)
ax.text(r1[0] - 0.05, r1[1] - 0.14, r"$\vec{r}_1$ (salida)", color="#9db8ff", fontsize=12,
        ha="center", va="top")
ax.text(r2[0] + 0.05, r2[1] - 0.14, r"$\vec{r}_2$ (llegada)", color="#ffab9c", fontsize=12,
        ha="center", va="top")

# ── Angulo de transferencia (por el lado corto, tipo I) ──────────────────────
ax.add_patch(Arc((0, 0), 0.7, 0.7, angle=0, theta1=205, theta2=340, color="#aaa", lw=1.3, zorder=3))
amid = np.radians((205 + 340) / 2)
ax.text(0.55 * np.cos(amid), 0.55 * np.sin(amid), r"$\Delta\theta$", color="#ccc",
        fontsize=13, ha="center", va="center")

# ── Tipo I: arco corto (< 180 grados), abomba hacia afuera ───────────────────
M = (r1 + r2) / 2
u = M / np.linalg.norm(M)
C1 = M + 0.55 * np.linalg.norm(r2 - r1) * u
pathI = Path([r1, C1, r2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
ax.add_patch(PathPatch(pathI, fill=False, edgecolor=C_I, lw=2.8, zorder=5))

# ── Tipo II: arco largo (> 180 grados), rodea el Sol por el lado opuesto ──────
Cc = -u * 2.0
ctrl1 = r1 + (Cc - r1) * 0.92 + np.array([-0.25, 0.0])
ctrl2 = r2 + (Cc - r2) * 0.92 + np.array([0.25, 0.0])
pathII = Path([r1, ctrl1, ctrl2, r2], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
ax.add_patch(PathPatch(pathII, fill=False, edgecolor=C_II, lw=2.4, ls=(0, (6, 4)), zorder=4))

# ── Velocidades v1, v2 (tangentes al arco de tipo I) ─────────────────────────
t1 = (C1 - r1); t1 = t1 / np.linalg.norm(t1)
t2 = (r2 - C1); t2 = t2 / np.linalg.norm(t2)
flecha(r1, r1 + 0.42 * t1, "#ffd54a", lw=2.4, scale=18)
flecha(r2, r2 + 0.42 * t2, "#ffd54a", lw=2.4, scale=18)
ax.text(*(r1 + 0.46 * t1 + np.array([0.02, 0.06])), r"$\vec{v}_1$", color="#ffd54a", fontsize=12)
ax.text(*(r2 + 0.46 * t2 + np.array([0.06, 0.0])), r"$\vec{v}_2$", color="#ffd54a", fontsize=12)

# ── Tiempo de vuelo ──────────────────────────────────────────────────────────
ax.text(0.0, -1.62, r"tiempo de vuelo prefijado  $\Delta t$", color="#33d6ff", fontsize=12,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#06141a", ec="#33d6ff", lw=1.3))

# ── Leyenda ──────────────────────────────────────────────────────────────────
leyenda = [
    Line2D([0], [0], color=C_I, lw=2.8, label=r"Tipo I: $\Delta\theta < 180^\circ$ (arco corto)"),
    Line2D([0], [0], color=C_II, lw=2.4, ls=(0, (6, 4)),
           label=r"Tipo II: $\Delta\theta > 180^\circ$ (arco largo)"),
    Line2D([0], [0], color="#ffd54a", lw=2.4, label=r"velocidades $\vec{v}_1,\ \vec{v}_2$ que resuelve Lambert"),
]
ax.legend(handles=leyenda, loc="upper center", facecolor="#111", edgecolor="#555",
          labelcolor="white", fontsize=10.5, framealpha=0.95)

ax.set_title("El problema de Lambert: dos posiciones y un tiempo de vuelo",
             color="white", fontsize=15, pad=12)
ax.set_xlim(-2.15, 2.15); ax.set_ylim(-1.95, 2.25)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

plt.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "imagenes"))
fig.savefig(os.path.join(OUT, "fig_lambert.pdf"), facecolor=BG, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_lambert.png"), dpi=140, facecolor=BG, bbox_inches="tight")
print("Figura escrita en imagenes/: fig_lambert.pdf / .png")

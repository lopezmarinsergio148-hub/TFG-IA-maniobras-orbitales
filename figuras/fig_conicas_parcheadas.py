# Genera la figura de las CONICAS PARCHEADAS (patched conics) para el subapartado de
# transferencias interplanetarias del capitulo de Fundamentos.
# Muestra las tres conicas: hiperbola de salida (SOI Tierra) -> arco heliocentrico
# (elipse de transferencia) -> hiperbola de llegada (SOI destino).
# Salida: fig_conicas_parcheadas.pdf (vectorial) y .png (vista previa).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

BG = "#0a0a1a"
plt.rcParams.update({"font.size": 12, "text.color": "white", "axes.edgecolor": "#444"})

# Colores de cada conica
C_SAL = "#00e676"   # hiperbola de salida
C_HEL = "#ffd700"   # arco heliocentrico (elipse de transferencia)
C_LLE = "#ff6ec7"   # hiperbola de llegada

fig, ax = plt.subplots(figsize=(12, 8.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# ── Orbitas de los planetas (heliocentricas, en UA) ──────────────────────────
r1, r2 = 1.0, 1.524                      # Tierra, Marte
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(r1 * np.cos(th), r1 * np.sin(th), color="#33d6ff", lw=1.3, alpha=0.8)
ax.plot(r2 * np.cos(th), r2 * np.sin(th), color="#ff8a5c", lw=1.3, alpha=0.8)
ax.text(r1 * np.cos(np.radians(235)), r1 * np.sin(np.radians(235)) - 0.05,
        "órbita de la Tierra", color="#33d6ff", fontsize=10, ha="center", va="top",
        style="italic")
ax.text(r2 * np.cos(np.radians(300)) + 0.05, r2 * np.sin(np.radians(300)) - 0.05,
        "órbita del planeta\nde destino", color="#ff8a5c", fontsize=10, ha="left",
        va="top", style="italic")

# ── Sol ──────────────────────────────────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), 0.11, facecolor="#ffcf33", edgecolor="#fff3c4", lw=1.5, zorder=6))
ax.text(0.0, -0.2, "Sol", color="#ffe082", fontsize=12, ha="center", va="top")

# ── Posiciones de salida y llegada ───────────────────────────────────────────
P_T = np.array([-r1, 0.0])               # Tierra (salida), en perihelio del arco
P_M = np.array([r2, 0.0])                # destino (llegada), en afelio del arco

# ── Esferas de influencia (SOI) de cada planeta ──────────────────────────────
SOI_T, SOI_M = 0.30, 0.24

# ── Arco heliocentrico (elipse de transferencia), RECORTADO fuera de las SOI ─
# Dentro de la SOI manda el planeta (hiperbola); fuera, el Sol (este arco). El
# cosido ocurre en el borde de la SOI: por eso el arco se dibuja SOLO fuera de ellas.
a_t = (r1 + r2) / 2
c_t = a_t - r1
b_t = a_t * np.sqrt(1 - (c_t / a_t) ** 2)
E = np.linspace(np.pi, 0, 900)           # de perihelio (Tierra) a afelio (destino), por arriba
xe = c_t + a_t * np.cos(E)
ye = b_t * np.sin(E)
d_T = np.hypot(xe - P_T[0], ye - P_T[1])
d_M = np.hypot(xe - P_M[0], ye - P_M[1])
i1 = int(np.argmax(d_T >= SOI_T))                       # punto donde sale de la SOI de la Tierra
i2 = len(E) - 1 - int(np.argmax((d_M >= SOI_M)[::-1]))  # punto donde entra en la SOI del destino
P1 = np.array([xe[i1], ye[i1]])          # cosido de salida (sobre la SOI de la Tierra)
P2 = np.array([xe[i2], ye[i2]])          # cosido de llegada (sobre la SOI del destino)
ax.plot(xe[i1:i2 + 1], ye[i1:i2 + 1], color=C_HEL, lw=2.6, zorder=4)
ax.text(c_t, b_t + 0.08, "arco heliocéntrico\n(elipse de transferencia, ley de Lambert)",
        color=C_HEL, fontsize=11, ha="center", va="bottom")


def hiperbola_patch(foco, patch, rp, e, saliente=True, cola=0.18):
    """Rama de hiperbola (foco en 'foco', periapsis rp) que TERMINA (saliente) o
    EMPIEZA (llegada) exactamente en 'patch', un punto sobre la SOI. Asi la hiperbola
    ni se sale de la SOI ni deja hueco: conecta con el arco heliocentrico en el cosido."""
    thmax = np.arccos(-1.0 / e)
    soi = np.hypot(patch[0] - foco[0], patch[1] - foco[1])
    ct = np.clip((rp * (1 + e) / soi - 1) / e, -0.999, 0.999)
    t_cross = np.arccos(ct)                              # anomalia donde la rama corta la SOI
    ang = np.arctan2(patch[1] - foco[1], patch[0] - foco[0])
    if saliente:                                         # el brazo saliente acaba en 'patch'
        rot = ang - t_cross
        t = np.linspace(-cola * thmax, t_cross, 200)
    else:                                                # el brazo entrante parte de 'patch'
        rot = ang + t_cross
        t = np.linspace(-t_cross, cola * thmax, 200)
    r = rp * (1 + e) / (1 + e * np.cos(t))
    x, y = r * np.cos(t), r * np.sin(t)
    xr = foco[0] + x * np.cos(rot) - y * np.sin(rot)
    yr = foco[1] + x * np.sin(rot) + y * np.cos(rot)
    return xr, yr


# ── SOI + hiperbola de SALIDA (Tierra), que conecta con el arco en P1 ────────
ax.add_patch(plt.Circle(P_T, SOI_T, facecolor="none", edgecolor="#8899aa", lw=1.1,
                        ls="--", zorder=3))
ax.plot(*P_T, "o", color="#5b8def", ms=12, zorder=7, markeredgecolor="white", markeredgewidth=0.8)
hx, hy = hiperbola_patch(P_T, P1, 0.06, 1.6, saliente=True)
ax.plot(hx, hy, color=C_SAL, lw=2.6, zorder=5)
ax.text(P_T[0] - 0.34, P_T[1] - 0.30, "Tierra\n(salida)", color="#9db8ff", fontsize=11,
        ha="center", va="top")
ax.text(P_T[0] - SOI_T - 0.03, P_T[1] - 0.02, "SOI", color="#8899aa", fontsize=10,
        ha="right", va="center", style="italic")

# ── SOI + hiperbola de LLEGADA (destino), que conecta con el arco en P2 ──────
ax.add_patch(plt.Circle(P_M, SOI_M, facecolor="none", edgecolor="#8899aa", lw=1.1,
                        ls="--", zorder=3))
ax.plot(*P_M, "o", color="#e0685a", ms=12, zorder=7, markeredgecolor="white", markeredgewidth=0.8)
lx, ly = hiperbola_patch(P_M, P2, 0.06, 1.6, saliente=False)
ax.plot(lx, ly, color=C_LLE, lw=2.6, zorder=5)
ax.text(P_M[0] + 0.32, P_M[1] - 0.02, "planeta\nde destino\n(llegada)", color="#ffab9c",
        fontsize=11, ha="left", va="center")
ax.text(P_M[0] + 0.02, P_M[1] - SOI_M - 0.03, "SOI", color="#8899aa", fontsize=10,
        ha="left", va="top", style="italic")

# ── Impulsos de cosido (Delta v) ─────────────────────────────────────────────
ax.annotate(r"$\Delta v_1$", P_T + np.array([0.05, 0.16]), color="white", fontsize=12,
            ha="left", va="center")
ax.annotate(r"$\Delta v_2$", P_M + np.array([-0.05, 0.16]), color="white", fontsize=12,
            ha="right", va="center")

# ── Leyenda de las tres conicas ──────────────────────────────────────────────
leyenda = [
    Line2D([0], [0], color=C_SAL, lw=2.6, label="1. Hipérbola de salida (SOI de la Tierra)"),
    Line2D([0], [0], color=C_HEL, lw=2.6, label="2. Arco heliocéntrico (dominado por el Sol)"),
    Line2D([0], [0], color=C_LLE, lw=2.6, label="3. Hipérbola de llegada (SOI del destino)"),
]
ax.legend(handles=leyenda, loc="upper left", facecolor="#111", edgecolor="#555",
          labelcolor="white", fontsize=10.5, framealpha=0.95)

ax.text(0.0, -r2 - 0.5,
        "Las esferas de influencia (SOI) y las hipérbolas se muestran agrandadas; no están a escala.",
        color="#888", fontsize=9.5, ha="center", va="top", style="italic")

ax.set_title("Transferencia interplanetaria por cónicas parcheadas",
             color="white", fontsize=15, pad=12)
ax.set_xlim(-2.0, 2.35); ax.set_ylim(-2.05, 1.95)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

plt.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "imagenes"))   # guarda en imagenes/ (donde apunta el LaTeX)
fig.savefig(os.path.join(OUT, "fig_conicas_parcheadas.pdf"), facecolor=BG, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_conicas_parcheadas.png"), dpi=140, facecolor=BG, bbox_inches="tight")
print("Figura escrita en imagenes/: fig_conicas_parcheadas.pdf / .png")

# Genera la figura de los REGIMENES ORBITALES terrestres (LEO, MEO, GEO) A ESCALA,
# para el apartado "Regimenes orbitales" del capitulo de Fundamentos.
# Salida: fig_regimenes_orbitales.pdf (vectorial, LaTeX) y .png (vista previa).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus, FancyArrowPatch

BG = "#0a0a1a"
plt.rcParams.update({"font.size": 12, "text.color": "white", "axes.edgecolor": "#444"})

# ── Radios en unidades de 1000 km ────────────────────────────────────────────
RE = 6.371                       # radio de la Tierra
LEO_IN, LEO_OUT = RE + 0.2, RE + 2.0
LEO_REP = RE + 0.4               # ISS (~400 km)
RGEO = RE + 35.786               # GEO
MEO_IN, MEO_OUT = LEO_OUT, RGEO
MEO_REP = RE + 20.2              # GPS (~20 200 km)

fig, ax = plt.subplots(figsize=(9.5, 9.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# ── Bandas de cada regimen (a escala) ────────────────────────────────────────
ax.add_patch(Annulus((0, 0), MEO_OUT, MEO_OUT - MEO_IN, facecolor="#2ecc71",
                     alpha=0.10, edgecolor="none", zorder=1))       # MEO (ancha)
ax.add_patch(Annulus((0, 0), LEO_OUT, LEO_OUT - LEO_IN, facecolor="#33d6ff",
                     alpha=0.30, edgecolor="none", zorder=2))       # LEO (fina)

# ── Tierra ───────────────────────────────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), RE, facecolor="#1f3a5f", edgecolor="#cfd8e3",
                        lw=1.6, zorder=4))
ax.text(0, 0, "Tierra", color="#cfd8e3", fontsize=11, ha="center", va="center",
        style="italic", zorder=5)


def orbita(r, color, ls="-", lw=1.8):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r * np.cos(th), r * np.sin(th), color=color, ls=ls, lw=lw, zorder=3)


def satelite(r, ang_deg, color, label):
    a = np.radians(ang_deg)
    x, y = r * np.cos(a), r * np.sin(a)
    ax.plot(x, y, "o", color=color, ms=11, zorder=6,
            markeredgecolor="white", markeredgewidth=0.8)
    return x, y


# ── Orbitas representativas + satelites ──────────────────────────────────────
orbita(LEO_REP, "#33d6ff")
orbita(MEO_REP, "#2ecc71")
orbita(RGEO, "#ffd700", lw=2.2)

satelite(LEO_REP, 90, "#33d6ff", "LEO")
satelite(MEO_REP, 48, "#2ecc71", "MEO")
satelite(RGEO, 0, "#ffd700", "GEO")

# ── Etiquetas con lineas guia (fuera, para no amontonar) ─────────────────────
def etiqueta(x, y, tx, ty, texto, color):
    ax.add_patch(FancyArrowPatch((tx, ty), (x, y), color=color, lw=1.1,
                 arrowstyle="-", connectionstyle="arc3,rad=0.0", zorder=7, alpha=0.8))
    ax.text(tx, ty, texto, color=color, fontsize=12, ha="left", va="center",
            zorder=8, bbox=dict(boxstyle="round,pad=0.4", fc="#0d0d18",
                                ec=color, lw=1.2, alpha=0.95))

etiqueta(0, LEO_REP, -20, 14,
         "LEO — órbita baja\n200 a 2000 km\n(ISS, observación)", "#33d6ff")
etiqueta(MEO_REP * np.cos(np.radians(48)), MEO_REP * np.sin(np.radians(48)), 6, 30,
         "MEO — órbita media\n2000 km hasta GEO\n(GPS, Galileo)", "#2ecc71")
etiqueta(RGEO, 0, RGEO + 2.5, -8,
         "GEO — geoestacionaria\n35 786 km\n(telecomunicaciones)", "#ffd700")

# ── Escala de altitud de referencia (eje inferior; 0 = superficie) ───────────
y0 = -RGEO - 2
xL, xR = RE, RE + 42
ax.annotate("", (xR, y0), (xL, y0), arrowprops=dict(arrowstyle="-", color="#555", lw=1))
for alt in [0, 10, 20, 30, 40]:
    xr = RE + alt
    ax.plot([xr, xr], [y0 - 0.4, y0 + 0.4], color="#777", lw=1)
    ax.text(xr, y0 - 0.7, f"{alt*1000:,}".replace(",", " "), color="#888",
            fontsize=8.5, ha="center", va="top")
ax.text((xL + xR) / 2, y0 - 2.2, "altitud sobre la superficie (km)",
        color="#888", fontsize=9.5, ha="center", va="top", style="italic")

ax.set_title("Regímenes orbitales terrestres (a escala)", color="white",
             fontsize=15, pad=14)
lim = RGEO + 8
ax.set_xlim(-lim, lim); ax.set_ylim(-lim - 2, lim)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

plt.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "imagenes"))   # guarda en imagenes/ (donde apunta el LaTeX)
fig.savefig(os.path.join(OUT, "fig_regimenes_orbitales.pdf"), facecolor=BG, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_regimenes_orbitales.png"), dpi=140, facecolor=BG, bbox_inches="tight")
print("Figura escrita en imagenes/: fig_regimenes_orbitales.pdf / .png")

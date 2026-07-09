# -*- coding: utf-8 -*-
"""
Genera el diagrama de ARQUITECTURA DE MÓDULOS del sistema (para cap4, 4.2.1).
Muestra cómo se conectan: usuario -> capa LLM -> (agentes RL | solvers clásicos)
-> entorno físico validado, con Validación y Visualización como módulos transversales.
Salida: imagenes/arquitectura_modulos.{pdf,png} (fondo oscuro del proyecto).
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes"))
FONDO = "#0a0a1a"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "text.color": "#eeeeee", "font.size": 10.5,
})


def caja(ax, cx, cy, w, h, titulo, sub, color):
    """Dibuja una caja redondeada centrada en (cx,cy) con título y subtítulo."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2, edgecolor=color, facecolor=color, alpha=0.18, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2, edgecolor=color, facecolor="none", zorder=3))
    ax.text(cx, cy + (0.16 if sub else 0), titulo, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#ffffff", zorder=4)
    if sub:
        ax.text(cx, cy - h / 2 + 0.22, sub, ha="center", va="center",
                fontsize=8.6, color="#cfcfe0", zorder=4)


def flecha(ax, p0, p1, texto="", color="#9aa0b5", rad=0.0):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if texto:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my, texto, ha="center", va="center", fontsize=8,
                style="italic", color="#b8b8c8",
                bbox=dict(boxstyle="round,pad=0.15", fc=FONDO, ec="none"), zorder=5)


fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

AZUL, NAR, TIERRA = "#4ea8de", "#e07a5f", "#c9a66b"
MORA, DORA, CIAN, GRIS = "#8b7cd8", "#e9c46a", "#76c7c0", "#8a8a9a"

# --- cajas ---
caja(ax, 6.0, 8.3, 3.4, 0.8, "Usuario", "petición en lenguaje natural", GRIS)
caja(ax, 6.0, 6.7, 6.8, 1.1, "Capa LLM: orquestador y explicador",
     "tool use, anclado a los datos reales", MORA)
caja(ax, 4.1, 4.7, 3.8, 1.3, "Agentes de RL (5)",
     "transferencias · cambio de plano\naerofrenado · mantenimiento", AZUL)
caja(ax, 8.4, 4.7, 3.8, 1.3, "Solvers clásicos",
     "Hohmann · Lambert\nporkchop", NAR)
caja(ax, 6.0, 2.4, 6.8, 1.3, "Entorno físico validado",
     "atmósferas (9 cuerpos) · propagador $J_2$ + arrastre", TIERRA)
caja(ax, 1.35, 2.4, 2.3, 1.3, "Validación", "datos de misión\n+ tests", DORA)
caja(ax, 10.65, 2.4, 2.3, 1.3, "Visualización", "Plotly /\nMatplotlib", CIAN)

# --- flechas ---
flecha(ax, (6.0, 7.9), (6.0, 7.28))                              # usuario -> LLM
flecha(ax, (5.0, 6.18), (4.4, 5.4), "invoca")                    # LLM -> RL
flecha(ax, (7.0, 6.18), (8.1, 5.4), "invoca")                    # LLM -> solvers
flecha(ax, (4.4, 4.05), (5.2, 3.08), "entrenan\nsobre")         # RL -> física
flecha(ax, (8.1, 4.05), (6.9, 3.08))                            # solvers -> física
flecha(ax, (2.5, 2.4), (3.0, 2.4), "valida")                    # validación -> física
flecha(ax, (9.0, 2.4), (9.5, 2.4), "representa")               # física -> visualización

ax.set_title("Arquitectura del sistema: conexión entre módulos",
             fontsize=13, color="#ffffff", pad=14)

fig.savefig(os.path.join(IMG, "arquitectura_modulos.pdf"), bbox_inches="tight", facecolor=FONDO)
fig.savefig(os.path.join(IMG, "arquitectura_modulos.png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
print("Guardada: imagenes/arquitectura_modulos.{pdf,png}")

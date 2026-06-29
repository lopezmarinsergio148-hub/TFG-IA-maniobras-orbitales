# ═══════════════════════════════════════════════════════════════════════════
#  FIGURA DIDÁCTICA de la validación de Urano (2 paneles):
#    A) ρ(h): modelo del TFG (original y recalibrado) vs Voyager 2 reconstruido
#    B) factor modelo/observación vs altura (cómo mejora al recalibrar)
#  Reutiliza la reconstrucción de reconstruir_validar.py (no duplica física).
# ═══════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib.pyplot as plt

from reconstruir_validar import (
    reconstruir_perfil, derivar_H, rho_modelo, CAPAS_VIEJO, CAPAS_NUEVO,
)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

h_km, Pbar, T, rho_obs = reconstruir_perfil()
viejo = derivar_H(CAPAS_VIEJO)
nuevo = derivar_H(CAPAS_NUEVO)

hgrid = np.linspace(0, 320, 500)
rho_v = np.array([rho_modelo(h, viejo) for h in hgrid])
rho_n = np.array([rho_modelo(h, nuevo) for h in hgrid])

# Para el panel de factores, interpolamos la obs a la rejilla del modelo
rho_obs_grid = np.interp(hgrid, h_km, rho_obs)
fac_v = rho_v / rho_obs_grid
fac_n = rho_n / rho_obs_grid

# Dos figuras separadas (un panel cada una) para la memoria.
figA, axA = plt.subplots(figsize=(8, 7))
figB, axB = plt.subplots(figsize=(8, 7))
for _f in (figA, figB):
    _f.patch.set_facecolor("#0a0a1a")

# ─────────────────── PANEL A: densidad vs altura ───────────────────
axA.set_facecolor("#0a0a1a")
mask = (h_km >= -10) & (h_km <= 320)
axA.plot(rho_obs[mask], h_km[mask], color="#7fdbe0", lw=3.0,
         label="Voyager 2 (reconstruido)", zorder=4)
axA.plot(rho_v, hgrid, color="#ff8888", lw=1.8, ls="--",
         label="Modelo TFG ORIGINAL (H=38 km)", zorder=3)
axA.plot(rho_n, hgrid, color="#ffcc66", lw=2.4,
         label="Modelo TFG RECALIBRADO (capa 150 km)", zorder=3)

# anclas duras de Lindal (posición exacta por interpolación en ln P)
lnPb = np.log(Pbar[::-1])
for P, nota in [(0.11, "Tropopausa\n52 K @ 110 mbar"),
                (0.0005, "0.5 mbar\n114 K")]:
    h_a = np.interp(np.log(P), lnPb, h_km[::-1])
    r_a = np.interp(np.log(P), lnPb, rho_obs[::-1])
    axA.scatter(r_a, h_a, color="white", s=55, zorder=6, edgecolor="#0a0a1a")
    axA.annotate(f"  {nota}", (r_a, h_a), color="white", fontsize=8.5, va="center")

# capa nueva resaltada
h150 = 150
r150 = rho_modelo(150, nuevo)
axA.scatter(r150, h150, color="#ffcc66", s=90, marker="D",
            zorder=7, edgecolor="white", label="Ancla NUEVA (150 km)")

axA.set_xscale("log")
axA.set_xlabel("Densidad ρ (kg/m³)  —  escala log", color="white", fontsize=11)
axA.set_ylabel("Altitud sobre el nivel de 1 bar (km)", color="white", fontsize=11)
axA.set_title("A) Perfil de densidad: modelo vs Voyager 2", color="white", fontsize=12, pad=10)
axA.tick_params(colors="white")
for sp in axA.spines.values():
    sp.set_edgecolor("#444")
axA.grid(True, color="#333", lw=0.5, alpha=0.6, which="both")
axA.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=9, loc="upper right")
axA.set_ylim(0, 320)

# ─────────────────── PANEL B: factor modelo/obs ───────────────────
axB.set_facecolor("#0a0a1a")
m2 = (hgrid <= 262)   # solo rango validado por Voyager
axB.plot(fac_v[m2], hgrid[m2], color="#ff8888", lw=2.2, ls="--",
         label="Original (mediana 2.49×)")
axB.plot(fac_n[m2], hgrid[m2], color="#ffcc66", lw=2.6,
         label="Recalibrado (mediana 1.15×)")
axB.axvline(1.0, color="#7fdbe0", lw=1.4, label="Acuerdo perfecto (factor = 1)")
axB.axvspan(0.5, 2.0, color="#2d8095", alpha=0.15, label="Banda ±2×")

axB.set_xlabel("Factor  ρ_modelo / ρ_Voyager", color="white", fontsize=11)
axB.set_ylabel("Altitud sobre 1 bar (km)", color="white", fontsize=11)
axB.set_title("B) Cuánto se desvía el modelo (rango 0–262 km)", color="white", fontsize=12, pad=10)
axB.tick_params(colors="white")
for sp in axB.spines.values():
    sp.set_edgecolor("#444")
axB.grid(True, color="#333", lw=0.5, alpha=0.6)
axB.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=9, loc="upper right")
axB.set_xlim(0, 4.3)
axB.set_ylim(0, 262)

# Sin título común (suptitle): la info de la cabecera va en el pie de figura del
# LaTeX. Se exporta cada panel por separado en PDF vectorial (carpeta imagenes/).
IMG = os.path.join(RAIZ, "imagenes")
figA.tight_layout()
figB.tight_layout()
for fg, nombre in [(figA, "validacion_urano_perfil"), (figB, "validacion_urano_ratio")]:
    fg.savefig(os.path.join(IMG, nombre + ".pdf"),
               facecolor=fg.get_facecolor(), bbox_inches="tight")
    fg.savefig(os.path.join(IMG, nombre + ".png"), dpi=150,
               facecolor=fg.get_facecolor(), bbox_inches="tight")
print("PDF y PNG escritos en", IMG)

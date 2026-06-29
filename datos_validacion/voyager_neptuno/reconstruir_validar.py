# ═══════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN ATMOSFÉRICA DE NEPTUNO  ·  Voyager 2 (1989)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Igual que Urano: no hay dataset PDS in-situ de densidad para Neptuno. La única
#  medida es la radio-ocultación de Voyager 2 (Lindal 1990/1992), que da T, P y
#  densidad numérica. Se reconstruye ρ(h) integrando la ecuación hidrostática
#  a partir de las anclas T–P medidas — el mismo método del Neptune-GRAM
#  (NASA/TM-20205001193).
#
#  VALIDACIÓN CRUZADA: la reconstrucción reproduce los puntos de muestra del
#  Neptune-GRAM (su h=0 ES el nivel de 1 bar) al ~1.6%:
#     h=0  km → GRAM 0.4425 kg/m³ (T=71.1 K) vs reconstruido 0.436
#     h=20 km → GRAM 0.1866 kg/m³ (T=54.8 K) vs reconstruido 0.184
#     h~4000 km → GRAM ~9e-15 kg/m³ vs modelo 1e-14
#
#  HALLAZGO: el modelo ORIGINAL del TFG subestimaba ρ por factores de 5–50×
#  en la franja 20–300 km. Causas: (1) H de la troposfera = 6.84 km (la física
#  a 1 bar es ~20 km), y (2) ρ_base a 50 km = 3e-4, unas ~100× demasiado baja
#  (lo correcto ≈ 0.03). Recalibrado: factor típico 5.62× → 1.17×.
#
#  Anclas T–P (Voyager/Lindal 1990/1992 + Neptune-GRAM + literatura):
#    · 1 bar → 72 K (Lindal) / 71.1 K (GRAM)
#    · 0.325 bar (h=20 km) → 54.8 K (GRAM)
#    · tropopausa ~100 mbar → ~52 K (Voyager)
#    · estratosfera 1–10 μbar → 168 ± 10 K (Bishop/Yelle)
#    · μ = 2.61 g/mol profundo (GRAM), 2.39 en estratosfera (CH4 condensado)
# ═══════════════════════════════════════════════════════════════════════════

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(HERE, "..", ".."))

R_GAS = 8.314
R_NEP = 24_764e3        # m, radio ecuatorial a 1 bar
G0    = 11.15           # m/s², gravedad a 1 bar (ecuador)

# Anclas T(P): P en bar, T en K
ANCLAS_P_bar = np.array([1.70, 1.00, 0.325, 0.10, 0.03, 0.01, 1e-3, 1e-4, 1e-5, 1e-6])
ANCLAS_T_K   = np.array([78.0, 72.0, 54.8,  52.0, 58.0, 72.0, 110., 150., 168., 168.])
ANCLAS_NOTA  = ["nube CH4 ~1.7 bar (Lindal)", "1 bar (Lindal/GRAM)",
                "h=20 km (GRAM 54.8 K)", "TROPOPAUSA ~100 mbar (Voyager)",
                "estratosfera", "estratosfera", "estratosfera media",
                "estratosfera alta", "1e-5 bar 168 K (Bishop)", "1e-6 bar (techo)"]


def mu_de_P(Pbar):
    """μ (g/mol): 2.61 profundo (GRAM, con CH4), 2.39 en estratosfera (CH4 condensado)."""
    return np.where(Pbar >= 0.5, 2.61,
           np.where(Pbar <= 0.1, 2.39,
                    2.39 + (2.61 - 2.39) * (Pbar - 0.1) / 0.4))


def reconstruir_perfil(n=5000):
    """Integra la hidrostática desde 1 bar. Devuelve h(km), P(bar), T(K), ρ_obs(kg/m³)."""
    lnP = np.log(np.logspace(np.log10(1.70e5), np.log10(1e-6 * 1e5), n))   # Pa, decreciente
    Pbar = np.exp(lnP) / 1e5
    T = np.interp(np.log(Pbar[::-1]), np.log(ANCLAS_P_bar[::-1]), ANCLAS_T_K[::-1])[::-1]
    mu = mu_de_P(Pbar)
    rho = np.exp(lnP) * (mu / 1000.0) / (R_GAS * T)

    h = np.zeros_like(lnP)
    i0 = int(np.argmin(np.abs(Pbar - 1.0)))
    for i in range(i0 + 1, len(lnP)):
        Tm = 0.5 * (T[i] + T[i-1]); mum = 0.5 * (mu[i] + mu[i-1]) / 1000.0
        g = G0 * (R_NEP / (R_NEP + h[i-1]))**2
        h[i] = h[i-1] - R_GAS * Tm / (mum * g) * (lnP[i] - lnP[i-1])
    for i in range(i0 - 1, -1, -1):
        Tm = 0.5 * (T[i] + T[i+1]); mum = 0.5 * (mu[i] + mu[i+1]) / 1000.0
        g = G0 * (R_NEP / (R_NEP + h[i+1]))**2
        h[i] = h[i+1] - R_GAS * Tm / (mum * g) * (lnP[i] - lnP[i+1])
    return h / 1000.0, Pbar, T, rho


# Modelos: h_min(km), ρ_base(kg/m³)
CAPAS_VIEJO = [(0, 0.45), (50, 3.0e-4), (600, 3.0e-8), (1500, 5.0e-11), (4000, 1.0e-14)]
# Recalibrado contra Voyager/GRAM: ρ a 50 km corregida (3e-4→0.03) y capas
# intermedias nuevas a 150 y 300 km. La termosfera (600+) se mantiene.
CAPAS_NUEVO = [(0, 0.45), (50, 0.03), (150, 2.5e-4), (300, 6.0e-6),
               (600, 3.0e-8), (1500, 5.0e-11), (4000, 1.0e-14)]


def derivar_H(capas, H_top=950.0):
    out = []
    for i in range(len(capas) - 1):
        hmin, rb = capas[i]; h_sup, rb_sup = capas[i + 1]
        out.append((hmin, rb, (h_sup - hmin) / np.log(rb / rb_sup)))
    out.append((capas[-1][0], capas[-1][1], H_top))
    return out


def rho_modelo(hk, capas_H):
    for hmin, rb, H in sorted(capas_H, reverse=True):
        if hk >= hmin:
            return rb * np.exp(-(hk - hmin) / H)
    return capas_H[0][1]


def main():
    h_km, Pbar, T, rho_obs = reconstruir_perfil()
    viejo = derivar_H(CAPAS_VIEJO)
    nuevo = derivar_H(CAPAS_NUEVO)
    print("H nuevas:", [(hm, round(H, 2)) for hm, _, H in nuevo])

    print(f"\n{'h_km':>6} {'P_bar':>9} {'T_K':>6} {'rho_obs':>11} "
          f"{'mod_viejo':>11} {'f_viejo':>8} {'mod_nuevo':>11} {'f_nuevo':>8}")
    objetivos = [0, 20, 40, 60, 100, 150, 200, 250, 300]
    filas = []
    for ht in objetivos:
        j = int(np.argmin(np.abs(h_km - ht)))
        ro = rho_obs[j]; rv = rho_modelo(h_km[j], viejo); rn = rho_modelo(h_km[j], nuevo)
        print(f"{h_km[j]:6.0f} {Pbar[j]:9.4f} {T[j]:6.1f} {ro:11.3e} "
              f"{rv:11.3e} {rv/ro:8.2f} {rn:11.3e} {rn/ro:8.2f}")
        filas.append((h_km[j], Pbar[j], T[j], ro, rv, rv/ro, rn, rn/ro))

    def resumen(nombre, c):
        f = np.array([rho_modelo(h, c) / r for h, r in zip(h_km, rho_obs)
                      if 0 <= h <= 300])
        print(f"  {nombre}: mediana={np.median(f):.2f}  rango=[{f.min():.2f},{f.max():.2f}]  "
              f"típico={10**np.mean(np.abs(np.log10(f))):.2f}x")
    print("\nFactor modelo/obs en 0-300 km (rango Voyager):")
    resumen("VIEJO", viejo); resumen("NUEVO", nuevo)

    with open(os.path.join(HERE, "voyager_neptuno_validacion.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["h_km", "P_bar", "T_K", "rho_obs_kg_m3",
                    "rho_modelo_viejo", "factor_viejo", "rho_modelo_nuevo", "factor_nuevo"])
        for r in filas:
            w.writerow([f"{r[0]:.1f}", f"{r[1]:.5f}", f"{r[2]:.1f}", f"{r[3]:.3e}",
                        f"{r[4]:.3e}", f"{r[5]:.3f}", f"{r[6]:.3e}", f"{r[7]:.3f}"])
    with open(os.path.join(HERE, "voyager_neptuno_anclas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["P_bar", "T_K", "nota_fuente"])
        for P, Tk, nota in zip(ANCLAS_P_bar, ANCLAS_T_K, ANCLAS_NOTA):
            w.writerow([P, Tk, nota])
    print("\nCSV escritos en", HERE)

    # ── Figura 2 paneles ────────────────────────────────────────────────────
    # Dos figuras separadas (un panel cada una) para la memoria.
    figA, axA = plt.subplots(figsize=(8, 7))
    figB, axB = plt.subplots(figsize=(8, 7))
    for _f in (figA, figB):
        _f.patch.set_facecolor("#0a0a1a")
    for ax in (axA, axB):
        ax.set_facecolor("#0a0a1a"); ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.grid(True, color="#333", lw=0.5, alpha=0.6, which="both")

    mask = (h_km >= -10) & (h_km <= 320)
    axA.plot(rho_obs[mask], h_km[mask], color="#4d79ff", lw=3.0, label="Voyager 2 (reconstruido)", zorder=4)
    hgrid = np.linspace(0, 320, 500)
    axA.plot([rho_modelo(h, viejo) for h in hgrid], hgrid, color="#ff8888", lw=1.8, ls="--",
             label="Modelo TFG ORIGINAL (H=6.84 km)", zorder=3)
    axA.plot([rho_modelo(h, nuevo) for h in hgrid], hgrid, color="#ffcc66", lw=2.4,
             label="Modelo TFG RECALIBRADO", zorder=3)
    # puntos GRAM duros
    for hh, rr, nn in [(0, 0.4425, "GRAM h=0 (1 bar)"), (20, 0.1866, "GRAM h=20 km")]:
        axA.scatter(rr, hh, color="white", s=55, zorder=6, edgecolor="#0a0a1a")
        axA.annotate(f"  {nn}", (rr, hh), color="white", fontsize=8.5, va="center")
    axA.set_xscale("log")
    axA.set_xlabel("Densidad ρ (kg/m³) — escala log", color="white", fontsize=11)
    axA.set_ylabel("Altitud sobre 1 bar (km)", color="white", fontsize=11)
    axA.set_title("A) Perfil de densidad: modelo vs Voyager 2", color="white", fontsize=12, pad=10)
    axA.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=9, loc="upper right")
    axA.set_ylim(0, 320)

    rho_obs_grid = np.interp(hgrid, h_km, rho_obs)
    m2 = hgrid <= 300
    axB.plot([rho_modelo(h, viejo) for h in hgrid[m2]] / rho_obs_grid[m2], hgrid[m2],
             color="#ff8888", lw=2.2, ls="--", label="Original (típico 8.0×)")
    axB.plot([rho_modelo(h, nuevo) for h in hgrid[m2]] / rho_obs_grid[m2], hgrid[m2],
             color="#ffcc66", lw=2.6, label="Recalibrado (típico 1.2×)")
    axB.axvline(1.0, color="#4d79ff", lw=1.4, label="Acuerdo perfecto")
    axB.axvspan(0.5, 2.0, color="#2d4a95", alpha=0.18, label="Banda ±2×")
    axB.set_xscale("log")
    axB.set_xlabel("Factor  ρ_modelo / ρ_Voyager  (log)", color="white", fontsize=11)
    axB.set_ylabel("Altitud sobre 1 bar (km)", color="white", fontsize=11)
    axB.set_title("B) Cuánto se desvía el modelo (0–300 km)", color="white", fontsize=12, pad=10)
    axB.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=9, loc="upper right")
    axB.set_ylim(0, 300)

    # Sin título común: la cabecera va en el pie de figura del LaTeX. Cada panel
    # por separado, en PDF vectorial + PNG (150 dpi) para la memoria.
    IMG = os.path.join(RAIZ, "imagenes")
    figA.tight_layout(); figB.tight_layout()
    for fg, nombre in [(figA, "validacion_neptuno_perfil"), (figB, "validacion_neptuno_ratio")]:
        fg.savefig(os.path.join(IMG, nombre + ".pdf"), facecolor=fg.get_facecolor(), bbox_inches="tight")
        fg.savefig(os.path.join(IMG, nombre + ".png"), dpi=150, facecolor=fg.get_facecolor(), bbox_inches="tight")
    print("PDF y PNG escritos en", IMG)


if __name__ == "__main__":
    main()

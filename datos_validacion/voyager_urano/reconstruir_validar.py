# ═══════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN ATMOSFÉRICA DE URANO  ·  Voyager 2 (1986)
# ═══════════════════════════════════════════════════════════════════════════
#
#  A diferencia de Júpiter (Galileo Probe) y Saturno (Cassini Grand Finale),
#  de Urano NO existe un dataset PDS in-situ de densidad. La única medida es la
#  radio-ocultación de Voyager 2 (Lindal et al. 1987), que da perfiles de
#  T, P y densidad numérica — NO una tabla ρ(h) directa, y está de pago en JGR.
#
#  Estrategia (la misma que usó NASA para construir el Uranus-GRAM,
#  NASA/TM-20210017250): tomar las anclas T–P medidas por Voyager e integrar
#  la ECUACIÓN HIDROSTÁTICA para reconstruir ρ(h). Luego comparar contra el
#  modelo piecewise-exponencial del TFG y calcular el factor mediano (igual
#  que en las validaciones de Júpiter y Saturno).
#
#  Rango de validez: el que cubrió Voyager → P de 2.3 bar a 0.3 mbar,
#  o sea h ≈ -25 a +262 km sobre el nivel de 1 bar (≈ -27.5 a 323.5 km según
#  el Uranus-GRAM). Por encima de 262 km el modelo NO está validado por datos.
#
#  Fuentes de las anclas T–P:
#    · Lindal, G.F. et al. (1987) JGR 92(A13), 14987 — tropopausa 52 K @ 110 mbar;
#      114 K @ 0.5 mbar; nube CH4 81 K @ 1.3 bar; 1 bar ≈ 76 K; rango 0.3 mbar–2.3 bar.
#    · NASA/TM-20210017250 (Uranus-GRAM, 2021) — método y rango de altitud.
#    · Composición (μ): 82.5% H2 / 15.2% He / 2.3% CH4 (urano.md) → μ=2.64 g/mol
#      en profundidad; CH4 condensa por encima de la nube → μ≈2.325 en estratosfera.
# ═══════════════════════════════════════════════════════════════════════════

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(HERE, "..", ".."))

# ── Constantes físicas de Urano ────────────────────────────────────────────
R_GAS   = 8.314          # J/mol/K
R_URANO = 25_559e3       # m, radio ecuatorial a 1 bar
G0      = 8.69           # m/s², gravedad a 1 bar (ecuador)

# ── Anclas T(P) medidas / derivadas de Voyager 2 (Lindal 1987) ──────────────
#    P en bar, T en K. Los puntos DUROS (medidos) son 1 bar, la tropopausa
#    (52 K @ 0.11 bar) y 114 K @ 0.5 mbar; el resto interpola el perfil
#    troposfera-estratosfera entre ellos.
ANCLAS_P_bar = np.array([2.30, 1.30, 1.00, 0.50, 0.11, 0.030, 0.005, 0.0005, 0.0003])
ANCLAS_T_K   = np.array([92.0, 81.0, 76.0, 64.0, 52.0, 62.0,  90.0,  114.0,  120.0])
ANCLAS_NOTA  = ["~2.3 bar (base datos, extrapol.)", "nube CH4 (Lindal)",
                "nivel 1 bar (h=0)", "troposfera media", "TROPOPAUSA 52 K (Lindal)",
                "estratosfera", "estratosfera", "0.5 mbar 114 K (Lindal)",
                "0.3 mbar (techo datos)"]


def mu_de_P(Pbar):
    """Peso molecular medio (g/mol). 2.64 en profundidad (con CH4),
    2.325 en estratosfera (CH4 condensado), transición en la nube ~0.1-0.5 bar."""
    return np.where(Pbar >= 0.5, 2.64,
           np.where(Pbar <= 0.1, 2.325,
                    2.325 + (2.64 - 2.325) * (Pbar - 0.1) / 0.4))


def reconstruir_perfil(n=4000):
    """Integra la hidrostática desde 1 bar y devuelve h(km), P(bar), T(K), ρ_obs(kg/m³)."""
    # Rejilla LOGARÍTMICA en presión: resolución uniforme en ln(P), lo que da
    # pasos de integración ~constantes en altura y buena resolución arriba.
    lnP = np.log(np.logspace(np.log10(2.30e5), np.log10(0.0003e5), n))   # Pa, decreciente
    Pbar = np.exp(lnP) / 1e5
    # T(P) por interpolación lineal en ln(P)
    T = np.interp(np.log(Pbar[::-1]), np.log(ANCLAS_P_bar[::-1]), ANCLAS_T_K[::-1])[::-1]
    mu = mu_de_P(Pbar)
    rho = np.exp(lnP) * (mu / 1000.0) / (R_GAS * T)        # kg/m³

    # Altura por integración hidrostática: dh = -H dlnP, con H = R T / (μ g(h))
    h = np.zeros_like(lnP)
    i0 = int(np.argmin(np.abs(Pbar - 1.0)))               # ancla h=0 en 1 bar
    for i in range(i0 + 1, len(lnP)):                     # hacia arriba
        Tm = 0.5 * (T[i] + T[i-1]); mum = 0.5 * (mu[i] + mu[i-1]) / 1000.0
        g = G0 * (R_URANO / (R_URANO + h[i-1]))**2
        h[i] = h[i-1] - R_GAS * Tm / (mum * g) * (lnP[i] - lnP[i-1])
    for i in range(i0 - 1, -1, -1):                       # hacia abajo
        Tm = 0.5 * (T[i] + T[i+1]); mum = 0.5 * (mu[i] + mu[i+1]) / 1000.0
        g = G0 * (R_URANO / (R_URANO + h[i+1]))**2
        h[i] = h[i+1] - R_GAS * Tm / (mum * g) * (lnP[i] - lnP[i+1])
    return h / 1000.0, Pbar, T, rho


# ── Modelos del TFG (capas bajas): h_min(km), ρ_base(kg/m³) ──────────────────
#    H se deriva para continuidad: H = (h_sup - h_min) / ln(ρ_base / ρ_base_sup)
CAPAS_VIEJO = [(0, 0.42), (50, 6.0e-2), (320, 5.0e-5), (1000, 5.0e-7)]
# Recalibrado contra Voyager: capa intermedia NUEVA en 150 km (ρ≈1.1e-3).
CAPAS_NUEVO = [(0, 0.42), (50, 6.0e-2), (150, 1.1e-3), (320, 5.0e-5), (1000, 5.0e-7)]


def derivar_H(capas):
    """Devuelve lista (h_min, ρ_base, H) con H por continuidad."""
    out = []
    for i, (hmin, rb) in enumerate(capas[:-1]):
        h_sup, rb_sup = capas[i + 1]
        H = (h_sup - hmin) / np.log(rb / rb_sup)
        out.append((hmin, rb, H))
    return out


def rho_modelo(hk, capas_H):
    for hmin, rb, H in sorted(capas_H, reverse=True):     # de mayor a menor h_min
        if hk >= hmin:
            return rb * np.exp(-(hk - hmin) / H)
    return capas_H[0][1]


def factores(h_km, rho_obs, capas_H, h_lo=0, h_hi=262):
    """Factor modelo/obs en el rango validado por Voyager."""
    mask = (h_km >= h_lo) & (h_km <= h_hi)
    f = np.array([rho_modelo(h, capas_H) / r for h, r in zip(h_km[mask], rho_obs[mask])])
    return f


def main():
    h_km, Pbar, T, rho_obs = reconstruir_perfil()
    viejo = derivar_H(CAPAS_VIEJO)
    nuevo = derivar_H(CAPAS_NUEVO)

    print("Alturas de escala derivadas (continuidad):")
    print("  VIEJO:", [(hm, round(H, 2)) for hm, _, H in viejo])
    print("  NUEVO:", [(hm, round(H, 2)) for hm, _, H in nuevo])

    print(f"\n{'h_km':>7} {'P_bar':>10} {'T_K':>6} {'rho_obs':>11} "
          f"{'mod_viejo':>11} {'f_viejo':>8} {'mod_nuevo':>11} {'f_nuevo':>8}")
    objetivos = [0, 25, 50, 80, 110, 150, 210, 262]
    filas = []
    for htgt in objetivos:
        j = int(np.argmin(np.abs(h_km - htgt)))
        ro = rho_obs[j]
        rv, rn = rho_modelo(h_km[j], viejo), rho_modelo(h_km[j], nuevo)
        print(f"{h_km[j]:7.1f} {Pbar[j]:10.4f} {T[j]:6.1f} {ro:11.4e} "
              f"{rv:11.4e} {rv/ro:8.2f} {rn:11.4e} {rn/ro:8.2f}")
        filas.append((h_km[j], Pbar[j], T[j], ro, rv, rv/ro, rn, rn/ro))

    fv = factores(h_km, rho_obs, viejo)
    fn = factores(h_km, rho_obs, nuevo)
    def resumen(nombre, f):
        tip = 10 ** np.mean(np.abs(np.log10(f)))
        print(f"  {nombre}: mediana={np.median(f):.2f}  rango=[{f.min():.2f}, {f.max():.2f}]  "
              f"factor típico={tip:.2f}x")
    print("\nFactor modelo/obs en 0-262 km (rango Voyager):")
    resumen("VIEJO", fv)
    resumen("NUEVO", fn)

    # ── CSV de anclas + perfil reconstruido + modelos ───────────────────────
    csv_path = os.path.join(HERE, "voyager_urano_validacion.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["h_km", "P_bar", "T_K", "rho_obs_kg_m3",
                    "rho_modelo_viejo", "factor_viejo",
                    "rho_modelo_nuevo", "factor_nuevo"])
        for r in filas:
            w.writerow([f"{r[0]:.2f}", f"{r[1]:.5f}", f"{r[2]:.1f}",
                        f"{r[3]:.4e}", f"{r[4]:.4e}", f"{r[5]:.3f}",
                        f"{r[6]:.4e}", f"{r[7]:.3f}"])
    print(f"\nCSV escrito: {csv_path}")

    # CSV de anclas crudas (trazabilidad de fuentes)
    anc_path = os.path.join(HERE, "voyager_urano_anclas.csv")
    with open(anc_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["P_bar", "T_K", "nota_fuente"])
        for P, Tk, nota in zip(ANCLAS_P_bar, ANCLAS_T_K, ANCLAS_NOTA):
            w.writerow([P, Tk, nota])
    print(f"CSV anclas:  {anc_path}")

    # ── Gráfica modelo vs Voyager reconstruido ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0a1a"); ax.set_facecolor("#0a0a1a")

    mask = (h_km >= -30) & (h_km <= 320)
    ax.plot(rho_obs[mask], h_km[mask], color="#7fdbe0", lw=2.6,
            label="Voyager 2 reconstruido (hidrostática)", zorder=4)

    hgrid = np.linspace(0, 320, 400)
    ax.plot([rho_modelo(h, viejo) for h in hgrid], hgrid, color="#ff8888",
            lw=1.6, ls="--", label="Modelo TFG (original)", zorder=3)
    ax.plot([rho_modelo(h, nuevo) for h in hgrid], hgrid, color="#ffcc66",
            lw=2.0, label="Modelo TFG (recalibrado Voyager)", zorder=3)

    # anclas duras de Lindal (posición exacta por interpolación en ln P)
    lnPb = np.log(Pbar[::-1])
    for P, Tk, nota in [(0.11, 52, "Tropopausa 52 K"), (0.0005, 114, "0.5 mbar 114 K")]:
        h_a = np.interp(np.log(P), lnPb, h_km[::-1])
        r_a = np.interp(np.log(P), lnPb, rho_obs[::-1])
        ax.scatter(r_a, h_a, color="white", s=45, zorder=6)
        ax.annotate(f" {nota}", (r_a, h_a), color="white", fontsize=8, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Densidad ρ (kg/m³)", color="white", fontsize=11)
    ax.set_ylabel("Altitud sobre 1 bar (km)", color="white", fontsize=11)
    ax.set_title("Validación atmosférica de URANO contra Voyager 2 (1986)\n"
                 "Reconstrucción hidrostática de Lindal et al. 1987",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(True, color="#333", lw=0.5, alpha=0.6, which="both")
    ax.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=9)
    ax.text(0.02, 0.02, "Rango validado: 0–262 km (P: 2.3 bar → 0.3 mbar)",
            transform=ax.transAxes, color="#888", fontsize=8)

    png_path = os.path.join(RAIZ, "validacion_urano.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=130, facecolor=fig.get_facecolor())
    print(f"PNG escrito:  {png_path}")


if __name__ == "__main__":
    main()

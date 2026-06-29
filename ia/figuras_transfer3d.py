# ═══════════════════════════════════════════════════════════════════════════
#  ia/figuras_transfer3d.py — Figuras del Agente 4 (transferencias 3D) para la memoria
#
#  Uso:  python ia/figuras_transfer3d.py
#  Corre el agente REAL (modelo_transfer3d/best_model) y lo compara con el optimo
#  de Hohmann + cambio de plano (baselines.delta_v_hohmann_plano).
#  Salida en imagenes/ia/ (PDF vectorial + PNG, fondo oscuro):
#    1) ia_transfer3d_orbita      — la maniobra LEO->GEO con cambio de plano, en 3D
#    2) ia_transfer3d_reparto     — como reparte el giro (grados abajo vs arriba) por Δi
#    3) ia_transfer3d_dv_vs_R     — Δv adimensional agente vs optimo (con Δi=28.5 fijo)
#    4) ia_transfer3d_invariancia — el MISMO modelo en los 9 cuerpos (invariancia 3D)
# ═══════════════════════════════════════════════════════════════════════════

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from stable_baselines3 import PPO
from poliastro.bodies import (Earth, Mars, Venus, Jupiter, Saturn, Uranus, Neptune,
                              Moon, Mercury)

from env_transfer3d import Transfer3DEnv, R_SPAN, DI_MAX
from baselines import R_TIERRA, MU_TIERRA, H_LEO, delta_v_hohmann_plano

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(AQUI, "..", "imagenes", "ia"))
FONDO = "#0a0a1a"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": "#888888", "text.color": "#eeeeee", "axes.labelcolor": "#eeeeee",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#333355",
    "axes.titlecolor": "#ffffff", "font.size": 11,
})
MODELO = os.path.join(AQUI, "modelo_transfer3d", "best_model")


def guardar(fig, nombre):
    fig.savefig(os.path.join(IMG, nombre + ".pdf"), bbox_inches="tight", facecolor=FONDO)
    fig.savefig(os.path.join(IMG, nombre + ".png"), dpi=200, bbox_inches="tight", facecolor=FONDO)
    print("  guardada:", nombre)


def geom_agente(model, R, di):
    """
    Corre el agente en (R, di) y devuelve la geometria de la maniobra que necesita
    el dibujo: impulsos, reparto del giro (gamma1 = grados girados ABAJO), inclinacion
    final, semieje y apogeo de la elipse de transferencia (todo adimensional, r1=1).
    """
    env = Transfer3DEnv()
    obs, _ = env.reset(options={"R": R, "di": di})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    dv1t, dv1n = info["dv1t"], info["dv1n"]
    g1 = np.arctan2(dv1n, 1.0 + dv1t)                 # inclinacion metida en el 1er impulso
    v1_sq = (1.0 + dv1t) ** 2 + dv1n ** 2
    a1 = -1.0 / (2.0 * (0.5 * v1_sq - 1.0))           # semieje de transferencia (mu=1, r1=1)
    r_apo = 2.0 * a1 - 1.0
    info.update({"gamma1": g1, "a_t": a1, "r_apo": r_apo})
    return info


# ── 1) La maniobra en 3D: LEO inclinada -> transferencia -> GEO ──────────────
def fig_orbita_3d():
    model = PPO.load(MODELO)
    R, di_deg = 6.22, 28.5
    di = np.radians(di_deg)
    info = geom_agente(model, R, di)
    g1 = info["gamma1"]                                # giro en el 1er impulso (rad)
    i_f = np.radians(info["i_f_deg"])                  # inclinacion final lograda
    r_apo = info["r_apo"]
    a_t, e_t = info["a_t"], (info["r_apo"] - 1.0) / (info["r_apo"] + 1.0)

    r1 = R_TIERRA + H_LEO                              # escala real (km)
    r2 = R * r1
    esc = r1                                           # dibujamos en km

    th = np.linspace(0, 2 * np.pi, 400)
    # Orbita inicial: circular r1 en el plano de referencia (XY)
    o0 = np.array([np.cos(th), np.sin(th), np.zeros_like(th)]) * r1
    # Orbita final: circular r2 en el plano inclinado i_f (nodos sobre el eje X)
    of = np.array([np.cos(th), np.sin(th) * np.cos(i_f), np.sin(th) * np.sin(i_f)]) * r2
    # Elipse de transferencia (perigeo +X a r1, apogeo -X a r_apo), plano inclinado g1
    nu = np.linspace(0, np.pi, 200)
    r_tr = a_t * (1 - e_t ** 2) / (1 + e_t * np.cos(nu)) * esc
    xt, yt = r_tr * np.cos(nu), r_tr * np.sin(nu)
    tr = np.array([xt, yt * np.cos(g1), yt * np.sin(g1)])   # rotada g1 sobre el eje X

    fig = plt.figure(figsize=(9.5, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(FONDO)
    # Tierra (esfera)
    uu, vv = np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 16)
    xe = R_TIERRA * np.outer(np.cos(uu), np.sin(vv))
    ye = R_TIERRA * np.outer(np.sin(uu), np.sin(vv))
    ze = R_TIERRA * np.outer(np.ones_like(uu), np.cos(vv))
    ax.plot_surface(xe, ye, ze, color="#4ea8de", alpha=0.55, linewidth=0, zorder=1)
    # orbitas
    ax.plot(*o0, color="#4ade80", lw=2.2, label="Órbita inicial (LEO, plano de referencia)")
    ax.plot(*tr, color="#f87171", lw=2.4, ls="--", label="Elipse de transferencia")
    ax.plot(*of, color="#e9c46a", lw=2.2, label=f"Órbita final (GEO, inclinada {di_deg:.1f}°)")
    # puntos de impulso: perigeo en +X (r1) y apogeo en -X (r_apo); Δv en km/s reales
    v_c1 = np.sqrt(MU_TIERRA / r1)
    dv1_kms = np.hypot(info["dv1t"], info["dv1n"]) * v_c1
    dv2_kms = np.hypot(info["dv2t"], info["dv2n"]) * v_c1
    ax.scatter([r1], [0], [0], color="#fff", s=45, zorder=5)
    ax.scatter([-r_apo * esc], [0], [0], color="#fff", s=45, zorder=5)
    ax.text(r1 * 1.08, 0, -r2 * 0.13, f"Δv₁ = {dv1_kms:.3f} km/s\n(inyección)",
            color="#4ade80", fontsize=9)
    ax.text(-r2 * 1.0, 0, r2 * 0.15, f"Δv₂ = {dv2_kms:.3f} km/s\n(circulariza + gira el plano)",
            color="#e9c46a", fontsize=9)

    dv_opt = delta_v_hohmann_plano(r1, r2, di, MU_TIERRA).dv_total
    dv_ag = info["dv_total"] * v_c1
    exceso = (dv_ag / dv_opt - 1.0) * 100.0
    txt = (f"Agente:  Δv total = {dv_ag:.4f} km/s\n"
           f"Óptimo:  Δv total = {dv_opt:.4f} km/s\n"
           f"exceso = {exceso:+.2f} %")
    ax.text2D(0.99, 0.99, txt, transform=ax.transAxes, fontsize=9, color="#eee",
              ha="right", va="top",
              bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555"))
    ax.set_title("Transferencia con cambio de plano (Δv del agente)\n"
                 "el grueso del giro se hace en el apogeo, donde la velocidad es menor",
                 fontsize=10.5)
    lim = r2 * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (km)", labelpad=6); ax.set_ylabel("y (km)", labelpad=6)
    ax.set_zlabel("z (km)", labelpad=6)
    ax.view_init(elev=26, azim=-60)
    ax.tick_params(labelsize=7, pad=1)
    ax.locator_params(nbins=5)                         # menos marcas en los ejes
    ax.legend(loc="upper left", facecolor="#15152a", edgecolor="#555",
              labelcolor="#eee", fontsize=8.5)
    ax.xaxis.set_pane_color((0, 0, 0, 0)); ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    guardar(fig, "ia_transfer3d_orbita")
    plt.close(fig)


# ── 2) Reparto del giro: que TROZO del Δi total se gira abajo vs arriba ──────
def fig_reparto():
    # Solo el OPTIMO (el concepto fisico, limpio). Para cada cambio de plano total
    # pedido (Δi, el eje X = la diagonal), se ve que TROZO se gira abajo (perigeo) y
    # cual arriba (apogeo). La diagonal gris es el total; la franja verde (abajo) es
    # una rajita y la amarilla (arriba) se lo come casi todo.
    R = 6.22
    dis = np.linspace(0, np.degrees(DI_MAX), 120)
    abajo = np.array([delta_v_hohmann_plano(1.0, R, np.radians(d), mu=1.0).di1_deg
                      for d in dis])

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.fill_between(dis, 0, abajo, color="#4ade80", alpha=0.85,
                    label="Girado ABAJO (perigeo, caro)")
    ax.fill_between(dis, abajo, dis, color="#e9c46a", alpha=0.85,
                    label="Girado ARRIBA (apogeo, barato)")
    ax.plot(dis, dis, color="#dddddd", lw=1.6, label="Total pedido (Δi)")

    # marca el caso GTO->GEO (Δi = 28.5 grados)
    d0 = 28.5
    ab0 = delta_v_hohmann_plano(1.0, R, np.radians(d0), mu=1.0).di1_deg
    ax.axvline(d0, color="#888", ls=":", lw=1)
    ax.annotate(f"Ej.: para girar {d0:.1f}°,\nsolo ~{ab0:.0f}° se hacen abajo\n"
                f"y ~{d0 - ab0:.0f}° arriba",
                xy=(d0, ab0 + (d0 - ab0) * 0.5), xytext=(11.5, 23.5),
                color="#eee", fontsize=9, ha="left",
                bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2))

    ax.set_xlabel("Cambio de plano TOTAL que se quiere hacer,  Δi  (grados)")
    ax.set_ylabel("De esos grados, cuántos se giran en cada sitio")
    ax.set_title("Reparto del cambio de plano: casi todo se gira ARRIBA (en el apogeo)")
    ax.set_xlim(0, np.degrees(DI_MAX)); ax.set_ylim(0, np.degrees(DI_MAX))
    ax.text(0.03, 0.97, "Girar el plano cuesta 2·v·sin(Δi/2): más barato\n"
            "donde vas lento (el apogeo). El agente RL reproduce\n"
            "esta estrategia: concentra el giro arriba.",
            transform=ax.transAxes, va="top", fontsize=9.5, color="#9ad",
            bbox=dict(boxstyle="round", facecolor="#15152a", edgecolor="#555"))
    ax.grid(True, alpha=0.2)
    # leyenda DEBAJO del grafico (fuera del area de datos, para no tapar las franjas)
    ax.legend(facecolor="#15152a", edgecolor="#555", labelcolor="#eee", fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    guardar(fig, "ia_transfer3d_reparto")
    plt.close(fig)


# ── 3) Δv adimensional agente vs optimo a lo largo de R (Δi = 28.5 fijo) ─────
def fig_dv_vs_R():
    model = PPO.load(MODELO)
    di = np.radians(28.5)
    Rs = np.linspace(1.12, R_SPAN, 90)
    dv_opt = [delta_v_hohmann_plano(1.0, R, di, mu=1.0).dv_total for R in Rs]
    dv_ag = [geom_agente(model, R, di)["dv_total"] for R in Rs]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(Rs, dv_opt, color="#bbbbbb", lw=2.4, label="Óptimo (Hohmann + cambio de plano)")
    ax.plot(Rs, dv_ag, color="#4ade80", lw=0, marker="o", ms=3.4, label="Agente RL (aprendido)")
    ax.set_xlabel("Ratio de la transferencia  R = r₂ / r₁")
    ax.set_ylabel("Δv total adimensional  (en unidades de la v. circular inicial)")
    ax.set_title("Con Δi = 28,5° fijo, el agente sigue al óptimo en todo el rango de R")
    ax.grid(True, alpha=0.25)
    ax.legend(facecolor="#15152a", edgecolor="#555", labelcolor="#eee")
    guardar(fig, "ia_transfer3d_dv_vs_R")
    plt.close(fig)


# ── 4) Invariancia de escala 3D: el MISMO modelo en los 9 cuerpos ───────────
def fig_invariancia():
    model = PPO.load(MODELO)
    di = np.radians(20.0)                              # un cambio de plano representativo
    cuerpos = [("Venus", Venus, "#e9c46a", "o", 5.5), ("Tierra", Earth, "#4ea8de", "s", 5.0),
               ("Marte", Mars, "#e07a5f", "^", 4.5), ("Júpiter", Jupiter, "#d9a066", "D", 4.2),
               ("Saturno", Saturn, "#c9a66b", "v", 4.5), ("Urano", Uranus, "#76c7c0", "P", 4.8),
               ("Neptuno", Neptune, "#5e7ce2", "X", 4.5), ("Luna", Moon, "#dddddd", "p", 4.2),
               ("Mercurio", Mercury, "#9b8cce", "*", 5.5)]
    Rs = np.linspace(1.2, R_SPAN, 28)
    dv_adim = {R: geom_agente(model, R, di)["dv_total"] for R in Rs}   # solo depende de (R,di)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    datos = {}
    for nombre, body, color, mk, sz in cuerpos:
        mu = float(body.k.to_value(u.km**3 / u.s**2))
        Rb = float(body.R.to_value(u.km))
        r1 = Rb + 500.0
        ex = []
        for R in Rs:
            v_c1 = np.sqrt(mu / r1)
            dv_ag = dv_adim[R] * v_c1
            dv_opt = delta_v_hohmann_plano(r1, r1 * R, di, mu).dv_total
            ex.append((dv_ag / dv_opt - 1.0) * 100.0)
        datos[nombre] = np.array(ex)
    ax.plot(Rs, datos[cuerpos[0][0]], color="#888888", lw=1.1, zorder=1)
    for nombre, body, color, mk, sz in cuerpos:
        ax.plot(Rs, datos[nombre], marker=mk, ms=sz, lw=0, color=color, label=nombre,
                alpha=0.95, markeredgecolor="#0a0a1a", markeredgewidth=0.3, zorder=3)
    ax.axhline(0, color="#888", ls="--", lw=1)
    ax.set_xlabel("Ratio de la transferencia  R = r₂ / r₁")
    ax.set_ylabel("Exceso del agente sobre el óptimo (%)")
    ax.set_title("Invariancia de escala en 3D: el mismo modelo vale para cualquier planeta")
    ax.text(0.5, -0.46, "Con Δi = 20° fijo, los 9 cuerpos caen en el MISMO punto a cada R\n"
            "(los marcadores se apilan) → el resultado no depende del planeta",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#9ad")
    ax.grid(True, alpha=0.25)
    ax.legend(facecolor="#15152a", edgecolor="#555", labelcolor="#eee", ncol=3,
              loc="upper right", fontsize=9)
    guardar(fig, "ia_transfer3d_invariancia")
    plt.close(fig)


if __name__ == "__main__":
    print("Generando figuras del Agente 4 (3D) en", IMG)
    fig_orbita_3d()
    fig_reparto()
    fig_dv_vs_R()
    fig_invariancia()
    print("Hecho.")

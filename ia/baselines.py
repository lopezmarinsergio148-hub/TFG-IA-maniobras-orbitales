# ═══════════════════════════════════════════════════════════════════════════
#  ia/baselines.py — Óptimos clásicos de referencia ("baselines")
#
#  Este módulo NO usa IA. Implementa las soluciones analíticas conocidas, que
#  sirven de JUEZ para evaluar al agente de RL: si el agente se acerca a estos
#  valores, es que ha aprendido bien.
#
#  Fase 1: transferencia de Hohmann LEO -> GEO (dos impulsos tangenciales), que
#  es la maniobra de MÍNIMO Δv entre dos órbitas circulares coplanarias.
#  (Las fórmulas son las del capítulo de Fundamentos de la memoria.)
#
#  Convenciones: unidades planas (km, km/s, s) por rapidez; las constantes
#  físicas se toman de poliastro (convención del proyecto) y se guardan como
#  floats para no arrastrar unidades en el bucle de entrenamiento.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 BASELINES — óptimos clásicos de referencia ("juez")
 Soluciones analíticas de la maniobra de Hohmann (con y sin cambio de plano),
 dimensionales y adimensionales. No usan IA: sirven de patrón de mínimo Δv contra
 el que se mide lo bien que ha aprendido cada agente de RL.

 ÍNDICE DE FUNCIONES:
   - velocidad_circular(r, mu)             : velocidad de una órbita circular.
   - delta_v_hohmann(r1, r2, mu)           : Hohmann coplanar (dos impulsos).
   - hohmann_leo_geo(...)                  : caso concreto LEO -> GEO de la Fase 1.
   - _dv_combinada(v_a, v_b, ang)          : coste de un impulso que gira el plano.
   - delta_v_hohmann_plano(r1, r2, di, mu) : Hohmann + cambio de plano óptimo.
   - hohmann_plano_adim(R, di)             : óptimo adimensional 3D (juez agente 3D).
   - hohmann_adim(R)                       : óptimo adimensional coplanar (juez 2D).
═══════════════════════════════════════════════════════════════════════════════
"""

from collections import namedtuple

import numpy as np
from scipy.optimize import minimize_scalar
from astropy import units as u
from poliastro.bodies import Earth

# ── Constantes físicas (de poliastro, convertidas a float) ──────────────────
MU_TIERRA = float(Earth.k.to_value(u.km**3 / u.s**2))   # μ = GM  [km³/s²]
R_TIERRA  = float(Earth.R.to_value(u.km))               # radio ecuatorial [km]

# ── Altitudes de referencia de la Fase 1 [km] ───────────────────────────────
H_LEO = 400.0       # órbita baja típica (tipo ISS)
H_GEO = 35786.0     # órbita geoestacionaria

# Resultado de una transferencia de Hohmann (todo en km/s y s)
ResultadoHohmann = namedtuple(
    "ResultadoHohmann", "dv1 dv2 dv_total t_transfer_s a_transfer")

# Resultado de una Hohmann CON cambio de plano (reparto del giro entre impulsos)
ResultadoHohmannPlano = namedtuple(
    "ResultadoHohmannPlano", "dv1 dv2 dv_total f_opt di1_deg di2_deg a_transfer")


def velocidad_circular(r, mu=MU_TIERRA):
    """Velocidad de una órbita circular de radio r [km]  ->  [km/s]."""
    return np.sqrt(mu / r)


def delta_v_hohmann(r1, r2, mu=MU_TIERRA):
    """
    Transferencia de Hohmann entre dos órbitas circulares coplanarias de radios
    r1 y r2 [km] (r1 = inicial, r2 = final). Devuelve un ResultadoHohmann con:
      dv1 : impulso en r1 para entrar en la elipse de transferencia [km/s]
      dv2 : impulso en r2 para circularizar [km/s]
      dv_total, t_transfer_s (medio periodo de la elipse) y a_transfer.
    """
    a_t = 0.5 * (r1 + r2)                       # semieje de la elipse de transferencia
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    dv2 = np.sqrt(mu / r2) * (1.0 - np.sqrt(2.0 * r1 / (r1 + r2)))
    t_transfer = np.pi * np.sqrt(a_t**3 / mu)   # medio periodo de la elipse [s]
    return ResultadoHohmann(dv1, dv2, dv1 + dv2, t_transfer, a_t)


def hohmann_leo_geo(mu=MU_TIERRA, r_planeta=R_TIERRA, h_leo=H_LEO, h_geo=H_GEO):
    """Caso concreto de la Fase 1: LEO -> GEO (de altitudes a radios)."""
    return delta_v_hohmann(r_planeta + h_leo, r_planeta + h_geo, mu)


def _dv_combinada(v_a, v_b, ang):
    """
    Magnitud de UN impulso que lleva de una velocidad de modulo v_a a otra v_b
    formando un angulo 'ang' entre ambas (ley del coseno). Combina en un solo
    encendido el cambio de modulo (parte tangencial, Hohmann) y el giro del plano
    (parte fuera del plano). Si ang=0 se reduce a |v_b - v_a| (Hohmann puro).
    """
    return np.sqrt(v_a**2 + v_b**2 - 2.0 * v_a * v_b * np.cos(ang))


def delta_v_hohmann_plano(r1, r2, di_rad, mu=MU_TIERRA):
    """
    Transferencia de Hohmann con CAMBIO DE PLANO combinado (di_rad = cambio total
    de inclinacion, en radianes). El giro se REPARTE entre los dos impulsos: una
    fraccion f en r1 y (1-f) en r2. El reparto OPTIMO se halla minimizando el
    dv_total(f) (problema 1D, sin formula cerrada -> minimizacion numerica).

    Idea fisica: girar el plano cuesta ~2*v*sin(di/2); como en el apogeo (r2) se va
    mas lento, el optimo mete casi todo el giro en el 2.o impulso. Hace de JUEZ del
    agente 3D igual que delta_v_hohmann hacia de juez del agente coplanar.

    Devuelve ResultadoHohmannPlano(dv1, dv2, dv_total, f_opt, di1_deg, di2_deg, a_t).
    """
    a_t = 0.5 * (r1 + r2)                        # semieje de la elipse de transferencia
    v_c1 = np.sqrt(mu / r1)                       # circular en la orbita inicial
    v_c2 = np.sqrt(mu / r2)                       # circular en la orbita final
    v_p = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))   # velocidad de transferencia en r1
    v_a = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))   # velocidad de transferencia en r2

    def total(f):
        dv1 = _dv_combinada(v_c1, v_p, f * di_rad)
        dv2 = _dv_combinada(v_a, v_c2, (1.0 - f) * di_rad)
        return dv1 + dv2

    f = float(minimize_scalar(total, bounds=(0.0, 1.0), method="bounded").x)
    dv1 = _dv_combinada(v_c1, v_p, f * di_rad)
    dv2 = _dv_combinada(v_a, v_c2, (1.0 - f) * di_rad)
    return ResultadoHohmannPlano(dv1, dv2, dv1 + dv2, f,
                                 np.degrees(f * di_rad),
                                 np.degrees((1.0 - f) * di_rad), a_t)


def hohmann_plano_adim(R, di_rad):
    """
    Optimo ADIMENSIONAL de la Hohmann con cambio de plano (juez del agente 3D).
    Todo en unidades de v_c1 = sqrt(mu/r1); depende SOLO de R = r2/r1 y di.
    Devuelve (dv_total_adim, f_opt). Reutiliza la version dimensional con mu=1, r1=1.
    """
    res = delta_v_hohmann_plano(1.0, R, di_rad, mu=1.0)
    return res.dv_total, res.f_opt


def hohmann_adim(R):
    """
    Optimo de Hohmann en forma ADIMENSIONAL (Fase 1 GENERALIZADA).

    Todo medido en unidades de la velocidad circular inicial v_c1 = sqrt(mu/r1).
    Se demuestra que el optimo SOLO depende del ratio R = r2/r1: al dividir las
    formulas de delta_v_hohmann entre v_c1, mu y r1 se cancelan. Esto es lo que
    hace la maniobra "invariante de escala" -> la misma politica vale para
    cualquier planeta y cualquier par de orbitas circulares.

    Vale para subir (R>1 -> impulsos progrados, signo +) y bajar (R<1 ->
    impulsos retrogrados, signo -). El "coste" total es |dv1| + |dv2|.

    Devuelve (dv1_adim, dv2_adim, dv_total_adim).
    """
    dv1 = np.sqrt(2.0 * R / (1.0 + R)) - 1.0
    dv2 = (1.0 / np.sqrt(R)) * (1.0 - np.sqrt(2.0 / (1.0 + R)))
    return dv1, dv2, abs(dv1) + abs(dv2)


if __name__ == "__main__":
    res = hohmann_leo_geo()
    print("=" * 60)
    print("  OPTIMO CLASICO - Transferencia de Hohmann LEO -> GEO (Tierra)")
    print("=" * 60)
    print(f"  mu Tierra = {MU_TIERRA:.1f} km3/s2   R = {R_TIERRA:.1f} km")
    print(f"  LEO: {H_LEO:.0f} km alt.  (r1 = {R_TIERRA + H_LEO:.1f} km)")
    print(f"  GEO: {H_GEO:.0f} km alt.  (r2 = {R_TIERRA + H_GEO:.1f} km)")
    print("-" * 60)
    print(f"  dv1 (inyeccion)     = {res.dv1:.4f} km/s")
    print(f"  dv2 (circularizar)  = {res.dv2:.4f} km/s")
    print(f"  dv TOTAL (optimo)   = {res.dv_total:.4f} km/s")
    print(f"  Tiempo de transfer. = {res.t_transfer_s / 3600.0:.2f} h")
    print("=" * 60)

    # ── Hohmann CON cambio de plano: LEO -> GEO girando 28.5 grados ──────────
    # (28.5 = inclinacion de un lanzamiento desde Cabo Canaveral hacia GEO ecuat.)
    di = np.radians(28.5)
    r1, r2 = R_TIERRA + H_LEO, R_TIERRA + H_GEO
    rp = delta_v_hohmann_plano(r1, r2, di)
    print()
    print("=" * 60)
    print("  HOHMANN CON CAMBIO DE PLANO - LEO -> GEO girando 28.5 grados")
    print("=" * 60)
    print(f"  dv1 (en LEO)        = {rp.dv1:.4f} km/s   (gira {rp.di1_deg:.2f} grados aqui)")
    print(f"  dv2 (en GEO/apogeo) = {rp.dv2:.4f} km/s   (gira {rp.di2_deg:.2f} grados aqui)")
    print(f"  dv TOTAL (optimo)   = {rp.dv_total:.4f} km/s")
    print(f"  reparto optimo f    = {rp.f_opt:.3f}  (fraccion del giro en el 1er impulso)")
    print("-" * 60)
    # Comparacion con las dos estrategias 'tontas' (todo el plano de un lado)
    todo_abajo = (_dv_combinada(np.sqrt(MU_TIERRA / r1),
                                np.sqrt(MU_TIERRA * (2 / r1 - 1 / rp.a_transfer)), di)
                  + _dv_combinada(np.sqrt(MU_TIERRA * (2 / r2 - 1 / rp.a_transfer)),
                                  np.sqrt(MU_TIERRA / r2), 0.0))
    todo_arriba = (_dv_combinada(np.sqrt(MU_TIERRA / r1),
                                 np.sqrt(MU_TIERRA * (2 / r1 - 1 / rp.a_transfer)), 0.0)
                   + _dv_combinada(np.sqrt(MU_TIERRA * (2 / r2 - 1 / rp.a_transfer)),
                                   np.sqrt(MU_TIERRA / r2), di))
    print(f"  Si girara TODO en LEO (abajo)   : {todo_abajo:.4f} km/s")
    print(f"  Si girara TODO en GEO (arriba)  : {todo_arriba:.4f} km/s")
    print(f"  El optimo reparte y mejora a ambos.")
    print("=" * 60)

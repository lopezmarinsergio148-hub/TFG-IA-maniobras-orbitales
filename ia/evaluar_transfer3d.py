# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_transfer3d.py — Evalua el agente 3D (Hohmann + cambio de plano)
#
#  Uso:   python ia/evaluar_transfer3d.py
#
#  PARTE A — Generalizacion en la Tierra: varios (R, di), agente vs optimo del juez
#            (baselines.delta_v_hohmann_plano). Mide exceso de Δv y error de llegada.
#  PARTE B — El MISMO modelo, SIN reentrenar, en otros planetas: se le da (R, di) de
#            una maniobra real, devuelve el Δv ADIMENSIONAL y se traduce a km/s
#            multiplicando por la v_c1 del planeta -> invariancia de escala en 3D.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 EVALUAR_TRANSFER3D — Evaluacion del agente 3D (Hohmann + cambio de plano)

 Como evaluar_transfer pero en 3D (salto de radio R mas cambio de inclinacion di):
   PARTE A: en la Tierra, varios pares (R, di) frente al optimo del juez
            (baselines.delta_v_hohmann_plano); mide exceso de Δv, error geometrico
            y error de inclinacion de llegada.
   PARTE B: el MISMO modelo, sin reentrenar, en otros planetas: se pasa (R, di)
            real, devuelve el Δv adimensional y se traduce a km/s por la v_c1 del
            planeta -> invariancia de escala tambien en 3D.

 ÍNDICE DE FUNCIONES:
   - pedir(model, R, di) : ejecuta la maniobra 3D del agente para (R, di) y devuelve su info.
   - main()              : corre las partes A y B e imprime las tablas comparativas.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

import numpy as np
from astropy import units as u
from stable_baselines3 import PPO
from poliastro.bodies import Earth, Mars, Jupiter

from env_transfer3d import Transfer3DEnv
from baselines import delta_v_hohmann_plano

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "modelo_transfer3d", "best_model")


def pedir(model, R, di):
    """Ejecuta la maniobra 3D del agente para el ratio R y el cambio de plano di; devuelve el info."""
    env = Transfer3DEnv()
    obs, _ = env.reset(options={"R": R, "di": di})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    return info


def main():
    """Evalua el agente 3D en la Tierra (parte A) y en otros planetas (parte B)."""
    model = PPO.load(MODELO)

    # ── PARTE A: la Tierra, varios (R, di) ──────────────────────────────────
    print("=" * 86)
    print("  PARTE A - Generalizacion en la Tierra (varios R y cambios de plano di)")
    print("=" * 86)
    print(f"  {'R':>6} {'di':>6} | {'dv_agente':>10} {'dv_optimo':>10} {'exceso':>8} "
          f"{'err_geom':>9} {'err_incl':>9}")
    print("-" * 86)
    casos = [(6.22, 28.5), (6.22, 0.0), (2.0, 10.0), (4.0, 20.0),
             (3.0, 30.0), (8.0, 15.0), (1.5, 25.0), (11.0, 20.0), (11.9, 28.5)]
    for R, di_deg in casos:
        di = np.radians(di_deg)
        info = pedir(model, R, di)
        if "dv_total" not in info:                      # la maniobra fallo (escape/degenerada)
            print(f"  {R:6.2f} {di_deg:6.1f} |  FALLO: {info.get('fallo', '?')}")
            continue
        dv_opt = delta_v_hohmann_plano(1.0, R, di, mu=1.0).dv_total
        exceso = (info["dv_total"] / dv_opt - 1.0) * 100.0
        print(f"  {R:6.2f} {di_deg:6.1f} | {info['dv_total']:10.4f} {dv_opt:10.4f} "
              f"{exceso:+7.2f}% {info['err_geom'] * 100:8.3f}% "
              f"{info['err_incl_deg']:7.2f} deg")

    # ── PARTE B: el MISMO modelo en otros planetas (cero reentrenamiento) ────
    print()
    print("=" * 92)
    print("  PARTE B - El MISMO modelo en otros planetas (SIN reentrenar)")
    print("=" * 92)
    casos_b = [
        ("Tierra",  Earth,    400.0,   35786.0, 28.5),   # LEO -> GEO con 28.5 grados
        ("Marte",   Mars,     400.0,   17000.0, 20.0),   # a areoestacionaria, inclinada
        ("Jupiter", Jupiter, 2000.0,  150000.0, 15.0),   # subida grande con plano
    ]
    print(f"  {'planeta':>8} {'alt r1->r2 (km)':>18} {'di':>6} {'R':>6} | "
          f"{'dv_ag km/s':>11} {'dv_opt km/s':>12} {'exceso':>8}")
    print("-" * 92)
    for nombre, body, h1, h2, di_deg in casos_b:
        mu = float(body.k.to_value(u.km**3 / u.s**2))
        Rb = float(body.R.to_value(u.km))
        r1, r2 = Rb + h1, Rb + h2
        R = r2 / r1
        di = np.radians(di_deg)
        v_c1 = np.sqrt(mu / r1)
        info = pedir(model, R, di)
        if "dv_total" not in info:
            print(f"  {nombre:>8} {f'{h1:.0f}->{h2:.0f}':>18} {di_deg:6.1f} {R:6.3f} |  "
                  f"FALLO: {info.get('fallo', '?')}")
            continue
        dv_ag = info["dv_total"] * v_c1
        dv_opt = delta_v_hohmann_plano(r1, r2, di, mu).dv_total
        exceso = (dv_ag / dv_opt - 1.0) * 100.0
        print(f"  {nombre:>8} {f'{h1:.0f}->{h2:.0f}':>18} {di_deg:6.1f} {R:6.3f} | "
              f"{dv_ag:11.4f} {dv_opt:12.4f} {exceso:+7.2f}%")
    print("=" * 92)
    print("  El agente solo se entreno en adimensional: clava otros planetas por escala.")


if __name__ == "__main__":
    main()

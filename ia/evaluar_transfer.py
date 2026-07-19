# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_transfer.py — Evalua el agente ADIMENSIONAL (Fase 1 generalizada)
#
#  Uso:   python ia/evaluar_transfer.py
#
#  Dos partes:
#    PARTE A — Generalizacion en la Tierra: el agente resuelve varios ratios R
#              (subiendo y bajando) y se compara con el optimo de Hohmann.
#    PARTE B — El MISMO modelo, SIN reentrenar, en otros planetas (Marte,
#              Jupiter): se le da el ratio R de una maniobra real, devuelve el
#              impulso ADIMENSIONAL, y lo traducimos a km/s multiplicando por la
#              v_c1 del planeta. Se compara con el optimo de Hohmann real.
#              -> Demuestra la invariancia de escala (el nucleo del enfoque).
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 EVALUAR_TRANSFER — Evaluacion del agente ADIMENSIONAL (Fase 1 generalizada)

 Comprueba la invariancia de escala del agente coplanar en dos partes:
   PARTE A: en la Tierra, resuelve varios ratios R (subir y bajar) y los compara
            con el optimo de Hohmann adimensional.
   PARTE B: el MISMO modelo, sin reentrenar, en otros planetas (Marte, Jupiter):
            se le pasa el R real, devuelve el impulso adimensional y se traduce a
            km/s multiplicando por la velocidad circular v_c1 del planeta.

 ÍNDICE DE FUNCIONES:
   - pedir_maniobra(model, R) : ejecuta la maniobra del agente para el ratio R y devuelve su info.
   - main()                   : corre las partes A y B e imprime las tablas comparativas.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

import numpy as np
from astropy import units as u
from stable_baselines3 import PPO
from poliastro.bodies import Earth, Mars, Jupiter

from env_transfer import TransferEnv
from baselines import hohmann_adim, delta_v_hohmann

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "modelo_transfer", "best_model")


def pedir_maniobra(model, R):
    """Le pasa el ratio R al agente y devuelve el info de la maniobra ejecutada."""
    env = TransferEnv()
    obs, _ = env.reset(options={"R": R})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    return info


def main():
    """Evalua el agente adimensional en la Tierra (parte A) y en otros planetas (parte B)."""
    model = PPO.load(MODELO)

    # ── PARTE A: la Tierra, varios ratios (subir y bajar) ───────────────────
    print("=" * 72)
    print("  PARTE A - Generalizacion en la Tierra (varios ratios R)")
    print("=" * 72)
    print(f"  {'R':>7} {'tipo':>6} | {'dv_agente':>10} {'dv_optimo':>10} "
          f"{'exceso':>8} {'error':>8}")
    print("-" * 72)
    for R in [0.1, 0.2, 0.5, 0.8, 2.0, 4.0, 6.22, 9.0, 11.0]:
        info = pedir_maniobra(model, R)
        _, _, dv_opt = hohmann_adim(R)
        exceso = (info["dv_total"] / dv_opt - 1.0) * 100.0
        tipo = "subir" if R > 1.0 else "bajar"
        print(f"  {R:7.3f} {tipo:>6} | {info['dv_total']:10.4f} {dv_opt:10.4f} "
              f"{exceso:+7.2f}% {info['error'] * 100:7.3f}%")

    # ── PARTE B: el MISMO modelo en otros planetas (cero reentrenamiento) ────
    print()
    print("=" * 78)
    print("  PARTE B - El MISMO modelo en otros planetas (SIN reentrenar)")
    print("=" * 78)
    casos = [
        ("Tierra",  Earth,    400.0,    35786.0),   # LEO -> GEO (subir, control)
        ("Marte",   Mars,     300.0,    17000.0),   # baja -> areoestacionaria aprox
        ("Jupiter", Jupiter,  2000.0,  200000.0),   # subida grande
        ("Marte",   Mars,    17000.0,     300.0),   # BAJAR de alta a baja
    ]
    print(f"  {'planeta':>8} {'alt r1->r2 (km)':>20} {'R':>7} | "
          f"{'dv_ag km/s':>11} {'dv_opt km/s':>12} {'exceso':>8} {'error':>8}")
    print("-" * 86)
    for nombre, body, h1, h2 in casos:
        mu = float(body.k.to_value(u.km**3 / u.s**2))
        Rb = float(body.R.to_value(u.km))
        r1, r2 = Rb + h1, Rb + h2
        R = r2 / r1
        v_c1 = np.sqrt(mu / r1)                       # velocidad circular real [km/s]
        info = pedir_maniobra(model, R)
        dv_ag_kms = info["dv_total"] * v_c1           # adimensional -> km/s reales
        dv_opt_kms = abs(delta_v_hohmann(r1, r2, mu).dv_total)
        exceso = (dv_ag_kms / dv_opt_kms - 1.0) * 100.0
        etiqueta = f"{h1:.0f}->{h2:.0f}"
        print(f"  {nombre:>8} {etiqueta:>20} {R:7.3f} | {dv_ag_kms:11.4f} "
              f"{dv_opt_kms:12.4f} {exceso:+7.2f}% {info['error'] * 100:7.3f}%")
    print("=" * 78)
    print("  NOTA: el agente solo se entreno con numeros adimensionales (mu=1, r1=1).")
    print("  Nunca 'vio' Marte ni Jupiter: clava la maniobra por invariancia de escala.")


if __name__ == "__main__":
    main()

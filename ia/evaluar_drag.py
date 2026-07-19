# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_drag.py — Evalúa el agente de aerofrenado de un planeta
#
#  Uso:   python ia/evaluar_drag.py  [planeta]
#  Ej.:   python ia/evaluar_drag.py  venus
#  Prueba al especialista en varios escenarios ALEATORIOS (distintos apogeo
#  inicial y objetivo) -> demuestra que generaliza dentro del planeta, no memoriza.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 EVALUAR_DRAG — Evaluacion del agente de AEROFRENADO de un planeta

 Carga el especialista entrenado y lo prueba en N escenarios ALEATORIOS (distinto
 apogeo inicial y objetivo dentro del mismo planeta), contando exitos. Demuestra
 que el agente generaliza dentro del planeta en vez de memorizar un caso concreto.

 ÍNDICE DE FUNCIONES:
   - _resultado(info) : traduce el info del episodio a "EXITO"/"DESTRUIDO"/"TIMEOUT".
   - main(planeta)    : corre los N escenarios y resume la tasa de exito.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

from stable_baselines3 import PPO

from env_drag import AeroBrakingEnv

AQUI = os.path.dirname(os.path.abspath(__file__))
N = 8   # nº de escenarios aleatorios a probar


def _resultado(info):
    """Devuelve la etiqueta de resultado del episodio a partir de su dict info."""
    if info.get("exito"):
        return "EXITO"
    if info.get("fallo"):
        return "DESTRUIDO"
    return "TIMEOUT"


def main(planeta="marte"):
    """Carga el agente de aerofrenado del planeta y lo evalua en N escenarios aleatorios."""
    modelo = os.path.join(AQUI, "modelo_drag", planeta, "best_model")
    model = PPO.load(modelo)
    env = AeroBrakingEnv(planeta=planeta, aleatorio=True)

    print("=" * 72)
    print(f"  AEROFRENADO en {planeta.upper()} — agente en escenarios ALEATORIOS")
    print(f"  corredor de perigeo: {env.H_PER_MIN/1000:.1f} .. {env.H_PER_MAX/1000:.1f} km")
    print("=" * 72)
    exitos = 0
    for s in range(N):
        obs, _ = env.reset(seed=s) #Cada valor de s genera un escenario distinto con distinto apoapsis inicial, apoapsis objetivo, mismo planeta y mismo coorredor de periapsis.
        apo0 = env.h_apo / 1000
        objetivo = env.h_apo_objetivo / 1000
        term = trunc = False
        info = {}
        while not (term or trunc):
            accion, _ = model.predict(obs, deterministic=True) #Comprobamos que para los mismos casos siempre da el mismo resultado óptimo.
            obs, rec, term, trunc, info = env.step(accion)
        res = _resultado(info)
        if info.get("exito"):
            exitos += 1
        print(f"  apogeo {apo0:7.0f} -> objetivo {objetivo:5.0f} km  |  "
              f"perigeo ~{info.get('h_per_km', 0):6.1f} km  |  "
              f"{info.get('pasos', 0):3d} pasadas  |  {res}")
    print("-" * 72)
    print(f"  RESULTADO: EXITO en {exitos}/{N} escenarios aleatorios")
    print("=" * 72)



if __name__ == "__main__":
    planeta = sys.argv[1] if len(sys.argv) > 1 else "marte"
    main(planeta)

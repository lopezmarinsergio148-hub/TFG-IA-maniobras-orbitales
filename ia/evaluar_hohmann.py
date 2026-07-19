# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_hohmann.py — Evalúa el agente de la Fase 1 frente al óptimo Hohmann
#
#  Uso:   python ia/evaluar_hohmann.py
#  Carga ia/modelo_hohmann/best_model.zip, le pide la maniobra LEO->GEO y la
#  compara con el óptimo analítico (ia/baselines.py). Éxito: Δv ≤ 5% del óptimo.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 EVALUAR_HOHMANN — Evaluacion del agente de la Fase 1 frente al optimo de Hohmann

 Carga el agente entrenado, le pide la maniobra LEO->GEO y la compara con el
 optimo analitico (baselines.hohmann_leo_geo). Reporta el exceso de Δv sobre el
 optimo y el error de llegada; el criterio de exito es Δv <= 5% del optimo y
 llegar a GEO (error de radio < 2%).

 ÍNDICE DE FUNCIONES:
   - main() : ejecuta la maniobra del agente, la compara con el optimo e imprime el veredicto.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

from stable_baselines3 import PPO

from env_hohmann import HohmannEnv
from baselines import hohmann_leo_geo, R_TIERRA, H_LEO, H_GEO

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "modelo_hohmann", "best_model")   # mejor modelo (EvalCallback)


def main():
    """Ejecuta la maniobra LEO->GEO del agente y la compara con el optimo de Hohmann."""
    model = PPO.load(MODELO) #Carga el cerebro (Agente) entrenado.
    env = HohmannEnv()
    r1, r2 = R_TIERRA + H_LEO, R_TIERRA + H_GEO

    obs, _ = env.reset(options={"r1": r1, "r2": r2}) #Monta la situación concreta (De LEO a GEO) y te devuelve el estado (La "foto" que verá el agente).
    action, _ = model.predict(obs, deterministic=True) #Le das el estado y te devuelve la acción (Su decisión)
    obs, rec, term, trunc, info = env.step(action) #Ejecuta esa maniobra en el simulador y te devuelve el info con los impulsos reales para que los veas.

    opt = hohmann_leo_geo() #Llama al JUEZ (baselines.py): calcula el óptimo clásico de Hohmann LEO->GEO. La "respuesta correcta".
    dv_ag, dv_op = info["dv_total"], opt.dv_total #Guarda en dos variables: el Δv total del AGENTE (de 'info', lo que hizo) y el del ÓPTIMO (del juez).
    exceso = (dv_ag / dv_op - 1.0) * 100.0 #Cuánto se PASA el agente sobre el óptimo, en %.
    llego = info["error_rel"] < 0.02 #¿Llego DE VERDAD a GEO? (circular, <2% error) -> True/False

    print("=" * 64)
    print("  EVALUACION: agente PPO vs optimo de Hohmann (LEO -> GEO)")
    print("=" * 64)
    print(f"  AGENTE : dv1={info['dv1']:.4f}  dv2={info['dv2']:.4f}  total={dv_ag:.4f} km/s")
    print(f"  OPTIMO : dv1={opt.dv1:.4f}  dv2={opt.dv2:.4f}  total={dv_op:.4f} km/s")
    print("-" * 64)
    print(f"  exceso del agente sobre el optimo = {exceso:+.2f} %")
    print(f"  error de llegada (circular en GEO)= {info['error_rel'] * 100:.3f} %")
    ok = (exceso <= 5.0) and llego
    print(f"  CUMPLE criterio (<=5% y llega)    = {'SI' if ok else 'NO'}")
    print("=" * 64)


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_keep.py — Evalua el agente de MANTENIMIENTO ORBITAL vs la baseline
#
#  Compara, en el mismo escenario, el agente PPO entrenado contra la estrategia
#  INGENUA (re-boost al maximo en cuanto cae a la mitad inferior de la banda).
#  Metricas: Dv total gastado (m/s) y si mantuvo la orbita en banda toda la mision.
#
#  Uso:   python ia/evaluar_keep.py  [planeta]  [n_escenarios]
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 EVALUAR_KEEP — Evaluacion del agente de MANTENIMIENTO ORBITAL vs baseline ingenua

 En el mismo escenario compara el agente PPO contra una estrategia INGENUA
 (re-boost al maximo en cuanto la orbita cae a la mitad inferior de la banda).
 Metricas por episodio: Δv total gastado (m/s) y si mantuvo la orbita en banda
 toda la mision; agrega exito y ahorro medio de Δv sobre la ingenua.

 ÍNDICE DE FUNCIONES:
   - _correr(env, politica, seed) : corre un episodio con una politica; devuelve (dv, exito, h, inc).
   - _ingenua(env, obs)           : politica baseline de re-boost por umbral.
   - main(planeta, n, aleatorio)  : compara agente vs ingenua en n escenarios y resume.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

import numpy as np
from stable_baselines3 import PPO

from env_keep import KeepEnv, BANDA_KM

AQUI = os.path.dirname(os.path.abspath(__file__))


def _correr(env, politica, seed):
    """Corre un episodio con una politica dada. Devuelve (dv, exito, h_obj_km, inc_deg)."""
    obs, _ = env.reset(seed=seed)
    h_obj_km = (env.a_obj - env.R) / 1000.0
    inc_deg = np.degrees(env.i_obj)
    dv_total = 0.0
    term = trunc = False
    info = {}
    while not (term or trunc):
        accion = politica(env, obs)
        obs, _, term, trunc, info = env.step(accion)
        dv_total += info["dv"]
    return dv_total, bool(info.get("exito")), h_obj_km, inc_deg


def _ingenua(env, obs):
    """Baseline: re-boost al maximo si esta por debajo de la mitad inferior de la banda."""
    desv_km = (env.a - env.a_obj) / 1000.0
    return np.array([1.0 if desv_km < -BANDA_KM * 0.5 else -1.0], dtype=np.float32)


def main(planeta="tierra", n=8, aleatorio=False):
    """Carga el agente del planeta y lo compara con la ingenua en n escenarios (Δv y exito)."""
    modelo = os.path.join(AQUI, "modelo_keep", planeta, "best_model.zip")
    if not os.path.exists(modelo):
        print(f"No encuentro el modelo: {modelo}\nEntrena primero con train_keep.py")
        return
    agente = PPO.load(modelo)
    env = KeepEnv(planeta=planeta, aleatorio=aleatorio)

    def pol_agente(env, obs):
        """Politica del agente PPO: predice la accion determinista para la observacion."""
        accion, _ = agente.predict(obs, deterministic=True)
        return accion

    print("=" * 72)
    print(f"  MANTENIMIENTO ORBITAL en {planeta.upper()}  —  agente PPO vs estrategia ingenua")
    print("=" * 72)
    print(f"  {'esc':>3} | {'h_km':>6} {'i_deg':>6} | {'Dv agente':>10} {'ok':>4} |"
          f" {'Dv ingenua':>11} {'ok':>4} | {'ahorro':>8}")
    print("-" * 84)
    g_ag = g_in = 0.0
    ok_ag = ok_in = 0
    for k in range(n):
        dv_ag, ex_ag, h_km, inc = _correr(env, pol_agente, seed=1000 + k)
        dv_in, ex_in, _, _ = _correr(env, _ingenua, seed=1000 + k)
        ok_ag += ex_ag
        ok_in += ex_in
        g_ag += dv_ag
        g_in += dv_in
        ahorro = (1.0 - dv_ag / dv_in) * 100.0 if dv_in > 0 else 0.0
        print(f"  {k:3d} | {h_km:6.0f} {inc:6.1f} | {dv_ag:9.1f} {str(ex_ag):>4} |"
              f" {dv_in:10.1f} {str(ex_in):>4} | {ahorro:7.1f}%")
    print("-" * 72)
    print(f"  EXITO: agente {ok_ag}/{n}  |  ingenua {ok_in}/{n}")
    print(f"  Dv medio: agente {g_ag/n:.1f} m/s  |  ingenua {g_in/n:.1f} m/s"
          f"  |  ahorro medio {(1.0 - g_ag/g_in)*100.0:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    planeta = sys.argv[1] if len(sys.argv) > 1 else "tierra"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    aleatorio = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    main(planeta, n, aleatorio)

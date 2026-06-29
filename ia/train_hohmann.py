# ═══════════════════════════════════════════════════════════════════════════
#  ia/train_hohmann.py — Entrena el agente PPO de la Fase 1 (Hohmann LEO->GEO fijo)
#
#  Uso:   python ia/train_hohmann.py  [n_timesteps]
#  Guarda el MEJOR modelo en  ia/modelo_hohmann/best_model.zip
#
#  No reimplementa PPO: usa stable-baselines3. Solo apunta el algoritmo a nuestro
#  entorno (env_hohmann) y lo deja aprender por prueba y error.
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env_hohmann import HohmannEnv

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.path.join(AQUI, "modelo_hohmann")   # carpeta propia del agente 1


def main(timesteps=400_000):
    # Monitor registra la recompensa por episodio (para ver el progreso).
    env = Monitor(HohmannEnv()) #El entorno donde entrena
    eval_env = Monitor(HohmannEnv())  #Un segundo entorno solo para evaluar (Lo usa el EvalCallback).

    # ent_coef > 0 mantiene algo de exploración, obliga al agente a seguir buscando y no cerrarse pronto en una solución mala.
    model = PPO("MlpPolicy", env, verbose=1, seed=0, ent_coef=0.01) #Crea el agente PPO. "MlpPolicy es su cerebro que es una red neuronal básica. seed=0=reproducible (Fija el azar).
    # EvalCallback evalúa cada cierto número de pasos y GUARDA EL MEJOR modelo visto:
    # así, aunque el entrenamiento empeore más tarde (inestabilidad), conservamos el bueno.
    eval_cb = EvalCallback(eval_env, best_model_save_path=DIR_MODELO, log_path=DIR_MODELO, #Evalúa cada 5000 pasos y guarda el mejor resultado.
                           eval_freq=5000, n_eval_episodes=5,
                           deterministic=True, verbose=0)
    print(f"Entrenando PPO durante {timesteps} pasos (guardando el mejor modelo)...")
    model.learn(total_timesteps=timesteps, callback=eval_cb) #Aquí entrena de verdad, practica "timesteps" veces y con el callback guarda el mejor por el camino.
    print("Mejor modelo guardado en", os.path.join(DIR_MODELO, "best_model.zip"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    main(n)

# ═══════════════════════════════════════════════════════════════════════════
#  ia/train_drag.py — Entrena el agente de AEROFRENADO (Fase 2) con PPO
#
#  Uso:   python ia/train_drag.py  [planeta]  [n_timesteps]
#  Ej.:   python ia/train_drag.py  venus
#  Guarda el mejor modelo en  ia/modelo_drag/<planeta>/best_model.zip
#  (un ESPECIALISTA por planeta; subcarpeta propia para no pisar a los demás).
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env_drag import AeroBrakingEnv

AQUI = os.path.dirname(os.path.abspath(__file__))


def main(planeta="marte", timesteps=400_000):
    dir_modelo = os.path.join(AQUI, "modelo_drag", planeta)
    env = Monitor(AeroBrakingEnv(planeta=planeta))
    eval_env = Monitor(AeroBrakingEnv(planeta=planeta))
    # gamma alto (0.999): el aerofrenado tiene horizonte LARGO (cientos de pasadas);
    # con el gamma por defecto (0.99) el premio de exito final queda descontado a casi
    # nada y el agente se vuelve timido (timeout). Ver lecciones de la guia de defensa.
    model = PPO("MlpPolicy", env, verbose=0, seed=0, ent_coef=0.02, gamma=0.999)
    # Guarda el MEJOR modelo (igual que en la Fase 1: robusto ante inestabilidad).
    eval_cb = EvalCallback(eval_env, best_model_save_path=dir_modelo, log_path=dir_modelo,
                           eval_freq=10000, n_eval_episodes=5,
                           deterministic=True, verbose=0)
    print(f"Entrenando aerofrenado en {planeta.upper()} durante {timesteps} pasos...")
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    print("Mejor modelo guardado en", os.path.join(dir_modelo, "best_model.zip"))


if __name__ == "__main__":
    planeta = sys.argv[1] if len(sys.argv) > 1 else "marte"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    main(planeta, n)

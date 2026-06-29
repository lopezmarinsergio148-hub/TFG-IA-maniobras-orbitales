# ═══════════════════════════════════════════════════════════════════════════
#  ia/train_keep.py — Entrena el agente de MANTENIMIENTO ORBITAL (station-keeping)
#
#  Uso:   python ia/train_keep.py  [planeta]  [n_timesteps]  [aleatorio]
#  Ej.:   python ia/train_keep.py  tierra
#         python ia/train_keep.py  tierra 600000 1     (generalizar: orbita aleatoria)
#  Guarda el mejor modelo en  ia/modelo_keep/<planeta>/best_model.zip
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env_keep import KeepEnv

AQUI = os.path.dirname(os.path.abspath(__file__))


def main(planeta="tierra", timesteps=400_000, aleatorio=False):
    dir_modelo = os.path.join(AQUI, "modelo_keep", planeta)
    env = Monitor(KeepEnv(planeta=planeta, aleatorio=aleatorio))
    eval_env = Monitor(KeepEnv(planeta=planeta, aleatorio=aleatorio))
    # gamma ALTO (0.999): la mision es larga (365 pasos); con gamma=0.99 el premio de
    # exito final se descuenta a 0.99^365 ~ 0.025 (invisible) -> agente timido. Misma
    # leccion del horizonte que el aerofrenado (ver guia de defensa).
    model = PPO("MlpPolicy", env, verbose=0, seed=0, ent_coef=0.01, gamma=0.999)
    eval_cb = EvalCallback(eval_env, best_model_save_path=dir_modelo, log_path=dir_modelo,
                           eval_freq=10000, n_eval_episodes=5,
                           deterministic=True, verbose=0)
    modo = "ALEATORIO (general)" if aleatorio else "FIJO (ISS)"
    print(f"Entrenando station-keeping en {planeta.upper()} [{modo}] durante {timesteps} pasos...")
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    print("Mejor modelo guardado en", os.path.join(dir_modelo, "best_model.zip"))


if __name__ == "__main__":
    planeta = sys.argv[1] if len(sys.argv) > 1 else "tierra"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    aleatorio = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    main(planeta, n, aleatorio)

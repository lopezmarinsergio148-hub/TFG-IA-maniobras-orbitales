# ═══════════════════════════════════════════════════════════════════════════
#  ia/train_transfer3d.py — Entrena el PPO en el entorno 3D (Hohmann + cambio de plano)
#
#  Uso:   python ia/train_transfer3d.py
#  Guarda el MEJOR modelo en  ia/modelo_transfer3d/best_model.zip
#
#  CURRICULUM en DOS ejes a la vez (R y di): el problema 3D es mas dificil que el
#  coplanar (la accion es 4D), asi que se empieza con maniobras suaves (poco salto
#  de radio y poco giro) y se amplian AMBOS rangos por etapas, reutilizando el
#  mismo modelo (set_env). El EvalCallback evalua SIEMPRE en el rango pleno.
#
#  Red algo mayor ([128,128]) que la de los agentes 2D: 4 grados de libertad de
#  accion + 2 de estado piden algo mas de capacidad (decision defendible, no capricho).
# ═══════════════════════════════════════════════════════════════════════════

import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env_transfer3d import Transfer3DEnv, R_SPAN, DI_MAX

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.path.join(AQUI, "modelo_transfer3d")

# Etapas del currículum: (tope de R, tope de di en grados, nº de pasos).
# Con R_SPAN ampliado a ~12 se añade una etapa intermedia (hasta R=8) para no saltar
# de golpe al rango pleno (el salto grande de R no convergia bien).
ETAPAS = [
    (2.0, 10.0, 200_000),    # maniobras suaves (poco salto, poco giro)
    (4.0, 25.0, 250_000),    # rango medio (ya incluye casos tipo GTO)
    (8.0, 40.0, 300_000),    # subidas grandes (lo que era el rango pleno anterior)
    (R_SPAN, np.degrees(DI_MAX), 550_000),   # rango pleno (R<=11.94, di<=40 grados)
]


def _env(r_span, di_max_deg):
    return Monitor(Transfer3DEnv(aleatorio=True, r_span=r_span,
                                 di_max=np.radians(di_max_deg)))


def main():
    eval_env = _env(R_SPAN, np.degrees(DI_MAX))          # SIEMPRE rango pleno
    eval_cb = EvalCallback(eval_env, best_model_save_path=DIR_MODELO, log_path=DIR_MODELO,
                           eval_freq=5000, n_eval_episodes=40,
                           deterministic=True, verbose=0)

    model = PPO("MlpPolicy", _env(*ETAPAS[0][:2]), verbose=0, seed=0, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]))

    primero = True
    for r_span, di_deg, pasos in ETAPAS:
        print(f"--- Currículum: R<=[{r_span:.1f}]  di<=[{di_deg:.0f} grados]  ({pasos} pasos) ---")
        model.set_env(_env(r_span, di_deg))
        model.learn(total_timesteps=pasos, callback=eval_cb, reset_num_timesteps=primero)
        primero = False

    print("Mejor modelo guardado en", os.path.join(DIR_MODELO, "best_model.zip"))


if __name__ == "__main__":
    main()

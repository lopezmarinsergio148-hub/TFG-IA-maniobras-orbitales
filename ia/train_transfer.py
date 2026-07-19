# ═══════════════════════════════════════════════════════════════════════════
#  ia/train_transfer.py — Entrena el PPO en el entorno ADIMENSIONAL (Fase 1 gen.)
#
#  Uso:   python ia/train_transfer.py
#  Guarda el MEJOR modelo en  ia/modelo_transfer/best_model.zip
#
#  CURRICULUM: el rango de ratios R (subir/bajar) es enorme (de ~1/12 a ~12), asi
#  que entrenar de golpe en todo el rango no converge. Se entrena por etapas,
#  AMPLIANDO el rango poco a poco (primero maniobras suaves, R cerca de 1; al
#  final, el rango pleno). Se reutiliza el MISMO modelo entre etapas (set_env).
#
#  El EvalCallback evalua SIEMPRE sobre el rango pleno (la distribucion objetivo
#  real) y guarda el mejor modelo visto -> asi "el mejor" se mide donde importa.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 TRAIN_TRANSFER — Entrenamiento del PPO en el entorno ADIMENSIONAL (Fase 1 gen.)

 Generaliza la transferencia de Hohmann a CUALQUIER ratio de radios R (subir o
 bajar) trabajando en variables adimensionales (mu=1, r1=1), lo que da
 invariancia de escala entre planetas. Entrena con CURRICULUM: amplia el rango de
 R por etapas reutilizando el mismo modelo (set_env); el EvalCallback evalua
 siempre en el rango pleno y guarda el mejor. Salida: ia/modelo_transfer/best_model.zip.

 ÍNDICE DE FUNCIONES:
   - main() : recorre las etapas del curriculum, entrena el PPO y guarda el mejor modelo.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env_transfer import TransferEnv, R_SPAN

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.path.join(AQUI, "modelo_transfer")   # carpeta propia (no pisar Fase 1)

# Etapas del currículum: (span de R, nº de pasos). Se amplia el rango poco a poco;
# el último tramo (rango pleno) es el más largo, pues es el más difícil (incluye
# los extremos de subir/bajar fuertes).
ETAPAS = [
    (1.6,     120_000),   # maniobras suaves (R en ~0.6 .. 1.6)
    (3.0,     150_000),   # rango medio
    (6.0,     180_000),   # ya incluye LEO->GEO (R=6.22)
    (R_SPAN,  350_000),   # rango pleno (~1/12 .. 12)
]


def main():
    """Entrena el PPO adimensional por etapas de curriculum y guarda el mejor modelo.

    Evalua siempre sobre el rango pleno de R; en cada etapa amplia el rango del
    entorno (set_env) sin reiniciar el contador de pasos entre tramos.
    """
    eval_env = Monitor(TransferEnv(aleatorio=True, r_span=R_SPAN))   # SIEMPRE rango pleno
    eval_cb = EvalCallback(eval_env, best_model_save_path=DIR_MODELO, log_path=DIR_MODELO,
                           eval_freq=5000, n_eval_episodes=30,
                           deterministic=True, verbose=0)

    # ent_coef > 0 mantiene exploracion (no cerrarse pronto en una mala solucion).
    model = PPO("MlpPolicy", Monitor(TransferEnv(aleatorio=True, r_span=ETAPAS[0][0])),
                verbose=0, seed=0, ent_coef=0.01)

    primero = True
    for span, pasos in ETAPAS:
        print(f"--- Currículum: R en [1/{span:.2f}, {span:.2f}]  ({pasos} pasos) ---")
        model.set_env(Monitor(TransferEnv(aleatorio=True, r_span=span)))
        model.learn(total_timesteps=pasos, callback=eval_cb,
                    reset_num_timesteps=primero)   # no reiniciar el contador entre etapas
        primero = False

    print("Mejor modelo guardado en", os.path.join(DIR_MODELO, "best_model.zip"))


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════
#  ia/evaluar_estadistico.py — Evaluacion ESTADISTICA de los 5 agentes
#
#  Para cada agente corre N episodios de evaluacion sobre escenarios ALEATORIOS
#  (semillas distintas de las del entrenamiento) y agrega:
#     - numero de episodios de evaluacion
#     - tasa de exito
#     - media +- desviacion tipica de las metricas clave
#     - (aerofrenado) % de destrucciones y % de timeouts
#
#  Objetivo: pasar de "50/50" a una evaluacion con media, desviacion y tasas,
#  para respaldar los resultados de RL de forma reproducible. NO reentrena nada;
#  usa los best_model ya guardados y solo mide.
#
#  Uso:   python ia/evaluar_estadistico.py            (todos, N=50)
#         python ia/evaluar_estadistico.py  20        (todos, N=20)
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys

import numpy as np
from astropy import units as u
from stable_baselines3 import PPO
from poliastro.bodies import Earth, Mars, Jupiter

from env_transfer import TransferEnv
from env_transfer3d import Transfer3DEnv
from env_drag import AeroBrakingEnv
from env_keep import KeepEnv, BANDA_KM
from env_hohmann import HohmannEnv
from baselines import (hohmann_adim, delta_v_hohmann, delta_v_hohmann_plano,
                       hohmann_leo_geo, R_TIERRA, H_LEO, H_GEO)

AQUI = os.path.dirname(os.path.abspath(__file__))
PLANETAS_ATMOSFERA = ["venus", "tierra", "marte", "jupiter", "saturno", "urano", "neptuno"]

# Semillas de evaluacion: se parte de una base ALTA (100000+) para no solapar con
# las semillas usadas al entrenar/validar (0..N en los otros scripts) -> separacion
# limpia entre entrenamiento y evaluacion.
SEMILLA_BASE = 100_000


def ms(v):
    """media +- desviacion tipica de una lista (o (nan, nan) si vacia)."""
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


# ───────────────────────────────────────────────────────────────────────────
#  Agente 1 — Hohmann LEO->GEO (escenario FIJO, determinista: 1 evaluacion)
# ───────────────────────────────────────────────────────────────────────────
def eval_hohmann():
    model = PPO.load(os.path.join(AQUI, "modelo_hohmann", "best_model"))
    env = HohmannEnv()
    r1, r2 = R_TIERRA + H_LEO, R_TIERRA + H_GEO
    obs, _ = env.reset(options={"r1": r1, "r2": r2})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    opt = hohmann_leo_geo()
    exceso = (info["dv_total"] / opt.dv_total - 1.0) * 100.0
    err = info["error_rel"] * 100.0
    print("\n" + "=" * 74)
    print("  AGENTE 1 — Hohmann LEO->GEO (escenario FIJO, politica determinista)")
    print("=" * 74)
    print(f"  Episodios de evaluacion : 1 (escenario unico, sin aleatoriedad)")
    print(f"  Exceso de Dv sobre optimo: {exceso:+.3f} %")
    print(f"  Error de llegada (circ.) : {err:.3f} %")
    print(f"  Criterio (<=5% y llega)  : {'CUMPLE' if exceso <= 5 and err < 2 else 'NO'}")


# ───────────────────────────────────────────────────────────────────────────
#  Agente 2 — Transferencias coplanares (adimensional, R aleatorio)
# ───────────────────────────────────────────────────────────────────────────
def _corre_transfer(model, env, R):
    """Fuerza un ratio R y devuelve (exceso%, error%)."""
    obs, _ = env.reset(options={"R": R})
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    _, _, dv_opt = hohmann_adim(R)
    return (info["dv_total"] / dv_opt - 1.0) * 100.0, info["error"] * 100.0


def eval_transfer(n):
    model = PPO.load(os.path.join(AQUI, "modelo_transfer", "best_model"))
    env = TransferEnv(aleatorio=True)
    # Rango UTIL (el que declara la memoria): R en [0.2, 11], log-uniforme, subir y bajar.
    rng = np.random.default_rng(SEMILLA_BASE)
    exc, err = [], []
    exito = 0
    for _ in range(n):
        R = float(np.exp(rng.uniform(np.log(0.2), np.log(11.0))))
        e_exc, e_err = _corre_transfer(model, env, R)
        exc.append(e_exc); err.append(e_err)
        if e_err < 2.0:               # criterio de exito: llega (<2% de error de radio)
            exito += 1
    # Extremos (bajadas profundas R<0.2): limitacion documentada, se reporta aparte.
    exc_x, err_x = [], []
    for _ in range(n):
        R = float(np.exp(rng.uniform(np.log(0.083), np.log(0.2))))
        e_exc, e_err = _corre_transfer(model, env, R)
        exc_x.append(e_exc); err_x.append(e_err)
    print("\n" + "=" * 74)
    print("  AGENTE 2 — Transferencias coplanares generales (R log-uniforme, subir/bajar)")
    print("=" * 74)
    print(f"  Episodios de evaluacion  : {n}   (rango util R en [0.2, 11])")
    print(f"  Tasa de exito (err<2%)   : {exito}/{n}  ({100.0*exito/n:.0f} %)")
    print(f"  Exceso de Dv sobre optimo: {ms(exc)[0]:+.2f} +- {ms(exc)[1]:.2f} %")
    print(f"  Error de llegada         : {ms(err)[0]:.2f} +- {ms(err)[1]:.2f} %")
    print(f"  --- Extremos R en [0.083, 0.2] (bajada profunda, limitacion conocida): "
          f"exceso {ms(exc_x)[0]:+.1f}%, error {ms(err_x)[0]:.1f}% (N={n})")


# ───────────────────────────────────────────────────────────────────────────
#  Agente 4 — Transferencias 3D con cambio de plano (R, di aleatorios)
# ───────────────────────────────────────────────────────────────────────────
def eval_transfer3d(n):
    model = PPO.load(os.path.join(AQUI, "modelo_transfer3d", "best_model"))
    env = Transfer3DEnv(aleatorio=True)
    exc, egeom, eincl = [], [], []
    exito = fallos = 0
    for k in range(n):
        obs, _ = env.reset(seed=SEMILLA_BASE + k)
        R, di = env.R, env.di
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        if "dv_total" not in info:
            fallos += 1
            continue
        dv_opt = delta_v_hohmann_plano(1.0, R, di, mu=1.0).dv_total
        e_exc = (info["dv_total"] / dv_opt - 1.0) * 100.0
        exc.append(e_exc)
        egeom.append(info["err_geom"] * 100.0)
        eincl.append(info["err_incl_deg"])
        if e_exc < 2.0:               # criterio: eficiencia (exceso de Dv < 2% del optimo)
            exito += 1
    print("\n" + "=" * 74)
    print("  AGENTE 4 — Transferencias 3D con cambio de plano (R, di aleatorios)")
    print("=" * 74)
    print(f"  Episodios de evaluacion : {n}")
    print(f"  Tasa de exito (exc<2%)  : {exito}/{n}  ({100.0*exito/n:.0f} %)")
    print(f"  Maniobras degeneradas   : {fallos}")
    print(f"  Exceso de Dv sobre optimo: {ms(exc)[0]:+.2f} +- {ms(exc)[1]:.2f} %")
    print(f"  Error geometrico llegada: {ms(egeom)[0]:.2f} +- {ms(egeom)[1]:.2f} %")
    print(f"  Error de inclinacion    : {ms(eincl)[0]:.2f} +- {ms(eincl)[1]:.2f} grados")


# ───────────────────────────────────────────────────────────────────────────
#  Agente 3 — Aerofrenado multi-planeta (por planeta, apogeo/objetivo aleatorios)
# ───────────────────────────────────────────────────────────────────────────
def eval_drag(n):
    print("\n" + "=" * 90)
    print("  AGENTE 3 — Aerofrenado (por planeta, escenarios aleatorios)")
    print("=" * 90)
    print(f"  {'planeta':>8} | {'exito':>7} {'destr':>6} {'timeout':>7} | "
          f"{'pasadas':>15} | {'Dv ajuste (m/s)':>18} | {'q_max (N/m2)':>16}")
    print("-" * 90)
    for planeta in PLANETAS_ATMOSFERA:
        model = PPO.load(os.path.join(AQUI, "modelo_drag", planeta, "best_model"))
        env = AeroBrakingEnv(planeta=planeta, aleatorio=True)
        pasadas, dv_tot, q_max = [], [], []
        exito = destr = tout = 0
        for k in range(n):
            obs, _ = env.reset(seed=SEMILLA_BASE + k)
            term = trunc = False
            dv_ep = 0.0
            q_ep = 0.0
            info = {}
            while not (term or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                dv_ep += info.get("dv_ajuste", 0.0)
                q_ep = max(q_ep, info.get("q_din", 0.0))
            if info.get("exito"):
                exito += 1
                pasadas.append(info.get("pasos", 0))
                dv_tot.append(dv_ep)
                q_max.append(q_ep)
            elif info.get("fallo"):
                destr += 1
            else:
                tout += 1
        mp = ms(pasadas); md = ms(dv_tot); mq = ms(q_max)
        print(f"  {planeta:>8} | {exito:>3}/{n:<3} {destr:>6} {tout:>7} | "
              f"{mp[0]:6.0f} +- {mp[1]:<5.0f} | {md[0]:8.2f} +- {md[1]:<6.2f} | "
              f"{mq[0]:6.3f} +- {mq[1]:<6.3f}")
    print("-" * 90)
    print(f"  (medias/desv calculadas SOLO sobre los episodios con exito; N={n} por planeta)")


# ───────────────────────────────────────────────────────────────────────────
#  Agente 5 — Mantenimiento orbital (por planeta, objetivo aleatorio) vs ingenua
# ───────────────────────────────────────────────────────────────────────────
def _ingenua_keep(env, obs):
    desv_km = (env.a - env.a_obj) / 1000.0
    return np.array([1.0 if desv_km < -BANDA_KM * 0.5 else -1.0], dtype=np.float32)


def _correr_keep(env, politica, seed):
    obs, _ = env.reset(seed=seed)
    dv_total = 0.0
    term = trunc = False
    info = {}
    while not (term or trunc):
        obs, _, term, trunc, info = env.step(politica(env, obs))
        dv_total += info["dv"]
    return dv_total, bool(info.get("exito"))


def eval_keep(n):
    print("\n" + "=" * 88)
    print("  AGENTE 5 — Mantenimiento orbital (por planeta, objetivo aleatorio) vs ingenua")
    print("=" * 88)
    print(f"  {'planeta':>8} | {'exito':>7} | {'Dv agente (m/s)':>20} | "
          f"{'Dv ingenua (m/s)':>20} | {'ahorro medio':>12}")
    print("-" * 88)
    for planeta in PLANETAS_ATMOSFERA:
        ruta = os.path.join(AQUI, "modelo_keep", planeta, "best_model.zip")
        if not os.path.exists(ruta):
            print(f"  {planeta:>8} |  (sin modelo)")
            continue
        agente = PPO.load(ruta)
        env = KeepEnv(planeta=planeta, aleatorio=True)

        def pol_ag(env, obs, _a=agente):
            accion, _ = _a.predict(obs, deterministic=True)
            return accion

        dv_ag, dv_in = [], []
        exito = 0
        for k in range(n):
            d_a, ex_a = _correr_keep(env, pol_ag, SEMILLA_BASE + k)
            d_i, _ = _correr_keep(env, _ingenua_keep, SEMILLA_BASE + k)
            dv_ag.append(d_a)
            dv_in.append(d_i)
            exito += ex_a
        ma = ms(dv_ag); mi = ms(dv_in)
        ahorro = (1.0 - ma[0] / mi[0]) * 100.0 if mi[0] > 0 else 0.0
        print(f"  {planeta:>8} | {exito:>3}/{n:<3} | {ma[0]:8.1f} +- {ma[1]:<7.1f} | "
              f"{mi[0]:8.1f} +- {mi[1]:<7.1f} | {ahorro:>10.1f} %")
    print("-" * 88)
    print(f"  (N={n} escenarios aleatorios por planeta; misma semilla para agente e ingenua)")


def main(n=50):
    print("#" * 90)
    print(f"#  EVALUACION ESTADISTICA DE LOS AGENTES RL   (N={n} episodios de evaluacion)")
    print(f"#  Semillas de evaluacion desde {SEMILLA_BASE} (separadas de las de entrenamiento)")
    print("#" * 90)
    eval_hohmann()
    eval_transfer(n)
    eval_transfer3d(n)
    eval_drag(n)
    eval_keep(n)
    print("\nHecho.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(n)

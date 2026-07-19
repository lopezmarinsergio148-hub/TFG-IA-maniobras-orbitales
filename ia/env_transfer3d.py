# ═══════════════════════════════════════════════════════════════════════════
#  ia/env_transfer3d.py — Entorno Gymnasium para la Fase 1 en 3D (cambio de plano)
#
#  Extiende el agente coplanar (env_transfer) a TRES dimensiones: ahora la orbita
#  destino, ademas de tener otro radio, esta INCLINADA un angulo di respecto a la
#  inicial. El agente debe hacer la transferencia de Hohmann Y girar el plano,
#  repartiendo el giro entre los dos impulsos de la forma mas barata posible.
#
#  Fisica clave (lo defendible):
#   - Girar el plano cuesta ~2*v*sin(di/2): es MAS BARATO donde se va mas lento
#     (en el apogeo). El optimo mete casi todo el giro en el 2.o impulso.
#   - Cada impulso es un VECTOR: una parte tangencial (cambia el tamaño de la
#     orbita, la Hohmann de siempre) + una parte fuera del plano (gira). Se suman
#     vectorialmente (ley del coseno) -> de ahi el coste combinado.
#
#  Accion VECTORIAL (4 numeros): el agente da las dos componentes (tangencial y
#  fuera de plano) de CADA impulso. NO se le dice como repartir el giro: lo
#  descubre solo (igual que el agente coplanar descubrio Hohmann).
#
#  ADIMENSIONAL (como env_transfer): mu=1, r1=1, v_c1=1. El optimo depende solo de
#  (R=r2/r1, di) -> invariante de escala: lo aprendido vale para cualquier planeta.
#
#  Alcance de esta version: SUBIR con cambio de plano (R>=1), el caso real
#  (inyeccion a GEO/orbitas altas inclinadas). Bajar + cambiar de plano a la vez
#  es raro y carisimo -> fuera de alcance (ademas acota los impulsos).
#
#  Estado:     [log(R)/log(R_SPAN), di/DI_MAX]            (2 numeros, en ~[0,1])
#  Accion:     [dv1_t, dv1_n, dv2_t, dv2_n] in [-1,1]^4   (se reescalan a su rango)
#  Recompensa: -(error_geom + W_INCL*error_incl) - C_DV*dv_total
#  Episodio de UN paso (bandit contextual), como el resto de agentes de Fase 1.
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 ENV_TRANSFER3D — entorno Gymnasium de la Fase 1 en 3D (cambio de plano)
 Extiende el agente coplanar a tres dimensiones: la órbita destino, además de otro
 radio, está inclinada un ángulo di. Cada impulso es un VECTOR (parte tangencial +
 parte fuera de plano, 4 componentes) y el agente descubre solo cómo repartir el
 giro. Adimensional (mu=1, r1=1): el óptimo depende de (R, di). Solo subir (R>=1).

 ÍNDICE DE CLASES/FUNCIONES:
   - _map(a, lo, hi)    : reescala una acción [-1,1] al rango físico [lo, hi].
   - _unmap(dv, lo, hi) : inversa de _map (impulso físico -> acción).
   - Transfer3DEnv      : entorno Gym 3D de un paso; acción vectorial de 4 números.
   - accion_optima(R,di): traduce el óptimo del juez a las 4 componentes de acción.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from baselines import delta_v_hohmann_plano

# ── Rangos del problema ──────────────────────────────────────────────────────
# R_SPAN = 11.94 es el MISMO limite fisico que el agente 2 coplanar: por encima, la
# transferencia BIELIPTICA supera a Hohmann -> ahi Hohmann ya no es el optimo y el
# "juez" deja de ser valido (seria deshonesto entrenar mas alla). (jun 2026: ampliado
# de 8.0 a 11.94 para igualar el alcance del agente 2; backup del modelo R8 guardado.)
R_MIN, R_SPAN = 1.10, 11.94        # ratio r2/r1: solo SUBIR (1.1 .. 11.94)
DI_MAX = np.radians(40.0)          # cambio de plano maximo (rad). 28.5 (GTO->GEO) cabe.
LOG_RMAX = np.log(R_SPAN)          # normalizacion FIJA del estado (rango pleno)

# Limites fisicos de cada componente de impulso (en unidades de v_c1).
# LECCION del agente 2 (defensa): el 1er impulso se TOPA por debajo del umbral de
# escape para que NUNCA pueda escapar -> se elimina el "acantilado" de castigo que
# hacia colapsar la politica (saturar dv1 y escapar). Condicion de no-escape en r1:
# (1+dv1_t)^2 + dv1_n^2 < 2; con (0.38, 0.22) -> 1.953 < 2 (con margen). El optimo
# (subir hasta R=8 pide dv1_t~0.33; girar poco abajo pide dv1_n pequeno) cabe holgado.
# Limites fisicos de cada componente de impulso (en unidades de v_c1).
# LECCION del agente 2 (defensa): el 1er impulso se TOPA por debajo del umbral de
# escape para que NUNCA pueda escapar -> se elimina el "acantilado" de castigo que
# hacia colapsar la politica (saturar dv1 y escapar). Condicion de no-escape en r1:
# (1+dv1_t)^2 + dv1_n^2 < 2; con (0.38, 0.22) -> 1.953 < 2 (con margen).
# NOTA (jun 2026): se probo ESTRECHAR estos rangos para bajar el exceso de ~4% a ~2%,
# pero el agente dejaba de COMPLETAR el giro en los casos de di grande (no llegaba en
# inclinacion: err 8-15 grados) -> se REVIRTIO a estos rangos (la solucion estable de ~4%).
DV1T_LO, DV1T_HI = -0.50, 0.38     # tangencial del 1er impulso (inyeccion, topado)
DV1N_LO, DV1N_HI = -0.22, 0.22     # fuera de plano del 1er impulso (poco giro abajo)
DV2T_LO, DV2T_HI = -0.60, 0.60     # tangencial del 2.o impulso (circulariza)
DV2N_LO, DV2N_HI = -0.90, 0.90     # fuera de plano del 2.o impulso (casi todo el giro)

# LECCION 3D (defensa): a diferencia del agente coplanar (donde C_DV podia ser
# minusculo porque "llegar" ya fijaba el optimo), aqui hay MUCHAS maniobras que
# llegan a la misma orbita inclinada con costes muy distintos -> el coste del
# combustible DEBE pesar para que el agente elija la barata. Se sube C_DV y se
# reescalan los errores para que "llegar" siga siendo prioritario.
# Pesos SUAVES (la config estable). Subir W a 6 y C_DV a 0.10-0.25 de golpe
# desestabilizaba (la politica colapsaba a saturar un impulso y escapar). Tuning
# gradual: pesos de llegada moderados + un C_DV un poco mayor que en 2D (el coste
# debe pesar algo mas que en el agente coplanar, pero sin reventar el aprendizaje).
W_GEOM = 1.0       # peso del error geometrico (llegar circular a R)
W_INCL = 2.0       # peso del error de inclinacion (rad; algo mayor por estar en rad)
C_DV = 0.05        # coste del combustible (5x el del agente 2D, sigue siendo suave)
ESCAPE_PEN = -3.0  # castigo si un impulso hace escapar / degenera la orbita
TOL_GEOM = 0.02    # tolerancia geometrica (llegar circular a R)
TOL_INCL = np.radians(1.0)   # tolerancia de inclinacion (1 grado)

# Ejes del marco inercial: plano de referencia = XY, linea de apsides/nodos = X.
_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def _map(a, lo, hi):
    """Reescala una accion en [-1, 1] al rango fisico [lo, hi]."""
    return lo + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


def _unmap(dv, lo, hi):
    """Inversa de _map: del impulso fisico a la accion en [-1, 1]."""
    return (dv - lo) / (hi - lo) * 2.0 - 1.0


class Transfer3DEnv(gym.Env):
    """Transferencia coplanar->inclinada, adimensional, de un paso (vectorial)."""
    metadata = {"render_modes": []}

    def __init__(self, aleatorio=True, r_span=R_SPAN, di_max=DI_MAX):
        """Configura el entorno 3D adimensional: modo (aleatorio o caso fijo), topes de
        R y di del currículum y los espacios de observación (log R, di) y acción (4)."""
        super().__init__()
        # aleatorio=True -> sortea (R, di) en el rango (politica GENERAL, entrenamiento).
        # aleatorio=False -> caso fijo LEO->GEO con 28.5 grados (depurar).
        self.aleatorio = aleatorio
        self.r_span = r_span        # tope de R al entrenar (para el currículum)
        self.di_max = di_max        # tope de di al entrenar (para el currículum)
        # estado: [log(R) normalizado, di normalizado]
        self.observation_space = spaces.Box(low=-0.2, high=1.2, shape=(2,), dtype=np.float32)
        # accion: 4 componentes de impulso en [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.R = None
        self.di = None

    def _obs(self):
        """Observación: [log(R) normalizado, di normalizado por DI_MAX]."""
        return np.array([np.log(self.R) / LOG_RMAX, self.di / DI_MAX], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Reinicia el episodio fijando el ratio R y el cambio de plano di (por
        'options', aleatorios en el rango o el caso fijo LEO->GEO a 28.5 grados)."""
        super().reset(seed=seed)
        if options and "R" in options and "di" in options:
            self.R = float(options["R"])
            self.di = float(options["di"])
        elif self.aleatorio:
            self.R = float(self.np_random.uniform(R_MIN, self.r_span))
            self.di = float(self.np_random.uniform(0.0, self.di_max))
        else:
            self.R = 6.22                 # LEO -> GEO
            self.di = np.radians(28.5)    # inclinacion de un lanzamiento a GEO
        return self._obs(), {}

    def step(self, action):
        """Aplica los dos impulsos vectoriales: monta la elipse de transferencia en su
        plano inclinado, hace el coasting al apogeo, cierra la órbita final y mide el
        error geométrico (log) y de inclinación, además del Δv gastado.
        Devuelve (obs, recompensa, terminado, truncado, info)."""
        dv1t = float(_map(action[0], DV1T_LO, DV1T_HI))
        dv1n = float(_map(action[1], DV1N_LO, DV1N_HI))
        dv2t = float(_map(action[2], DV2T_LO, DV2T_HI))
        dv2n = float(_map(action[3], DV2N_LO, DV2N_HI))
        R, di = self.R, self.di
        info = {"dv1t": dv1t, "dv1n": dv1n, "dv2t": dv2t, "dv2n": dv2n, "R": R, "di": di}

        # ── Impulso 1 en el perigeo P1 = (1,0,0); v circular inicial = (0,1,0) ──
        P1 = _X
        dv1 = dv1t * _Y + dv1n * _Z          # tangencial (Y) + fuera de plano (Z)
        v1 = 1.0 * _Y + dv1                   # v_c1 = 1
        e1 = 0.5 * float(np.dot(v1, v1)) - 1.0   # energia (mu = 1, r1 = 1)
        if e1 >= 0.0:
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "escape_1"}
        a1 = -1.0 / (2.0 * e1)
        r_apo = 2.0 * a1 - 1.0                # P1 es perigeo (v1 perpendicular a P1)
        if r_apo <= 0.0:
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "degenerada"}
        h1 = np.cross(P1, v1)
        h1n = h1 / np.linalg.norm(h1)         # normal del plano de transferencia

        # ── Coasting hasta el apogeo P2 = (-r_apo,0,0); velocidad de llegada ──
        P2 = -r_apo * _X
        r2hat = P2 / np.linalg.norm(P2)
        v_apo_mag = np.sqrt(2.0 / r_apo - 1.0 / a1)
        v2_lleg = v_apo_mag * np.cross(h1n, r2hat)   # progrado, en el plano inclinado

        # ── Impulso 2 en el marco local del apogeo: t2 (a lo largo de v) + w2 (fuera) ──
        t2 = v2_lleg / np.linalg.norm(v2_lleg)
        w2 = np.cross(r2hat, t2)
        w2 = w2 / np.linalg.norm(w2)
        dv2 = dv2t * t2 + dv2n * w2
        v_fin = v2_lleg + dv2
        e2 = 0.5 * float(np.dot(v_fin, v_fin)) - 1.0 / r_apo
        if e2 >= 0.0:
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "escape_2"}

        # ── Orbita final: forma (perigeo/apogeo) e inclinacion ──
        a_f = -1.0 / (2.0 * e2)
        h_f = np.cross(P2, v_fin)
        h_f_mag = np.linalg.norm(h_f)
        i_f = float(np.arccos(np.clip(h_f[2] / h_f_mag, -1.0, 1.0)))
        e_vec = ((float(np.dot(v_fin, v_fin)) - 1.0 / r_apo) * P2
                 - float(np.dot(P2, v_fin)) * v_fin)        # mu = 1
        e_f = float(np.linalg.norm(e_vec))
        r_peri_f = a_f * (1.0 - e_f)
        r_apo_f = a_f * (1.0 + e_f)

        # ── Recompensa: llegar CIRCULAR a R y con inclinacion di, gastando poco ──
        # error geometrico en escala log (simetrico/acotado, leccion del agente 2)
        err_geom = 0.5 * (abs(np.log(max(r_apo_f, 1e-6) / R))
                          + abs(np.log(max(r_peri_f, 1e-6) / R)))
        err_incl = abs(i_f - di)
        dv_total = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
        recompensa = -(W_GEOM * err_geom + W_INCL * err_incl + C_DV * dv_total)
        exito = (err_geom < TOL_GEOM) and (err_incl < TOL_INCL)
        info.update({"err_geom": err_geom, "err_incl_deg": np.degrees(err_incl),
                     "i_f_deg": np.degrees(i_f), "r_peri_f": r_peri_f, "r_apo_f": r_apo_f,
                     "dv_total": dv_total, "exito": exito})
        return self._obs(), float(recompensa), True, False, info


def accion_optima(R, di):
    """
    Traduce el OPTIMO del juez (baselines.delta_v_hohmann_plano) a las 4 componentes
    de accion de este entorno. Sirve para la prueba a mano: si el entorno es
    correcto, alimentado con esto debe dar error ~0 e inclinacion final = di.
    """
    res = delta_v_hohmann_plano(1.0, R, di, mu=1.0)
    g1 = res.f_opt * di                                  # giro metido en el 1er impulso
    v_p = np.sqrt(2.0 * R / (1.0 + R))                   # vel. de transferencia en r1
    v_a = np.sqrt(2.0 / (R * (1.0 + R)))                 # vel. de transferencia en r2
    v_c2 = 1.0 / np.sqrt(R)                              # circular en r2
    dv1t = v_p * np.cos(g1) - 1.0
    dv1n = v_p * np.sin(g1)
    dv2t = v_c2 * np.cos(di - g1) - v_a
    dv2n = -v_c2 * np.sin(di - g1)                       # negativo en el marco w2
    return np.array([_unmap(dv1t, DV1T_LO, DV1T_HI), _unmap(dv1n, DV1N_LO, DV1N_HI),
                     _unmap(dv2t, DV2T_LO, DV2T_HI), _unmap(dv2n, DV2N_LO, DV2N_HI)],
                    dtype=np.float32)


if __name__ == "__main__":
    # PRUEBA A MANO (no entrena): alimenta el entorno con el OPTIMO del juez en
    # varios (R, di) y comprueba que el error geometrico ~0 y la inclinacion
    # final = di. Valida que la geometria 3D del entorno es correcta.
    print("=" * 78)
    print("  PRUEBA A MANO del entorno 3D con el OPTIMO de Hohmann + cambio de plano")
    print("=" * 78)
    print(f"  {'R':>6} {'di':>6} | {'err_geom':>9} {'i_final':>8} {'di_obj':>7} "
          f"{'dv_tot':>8} {'exito':>6}")
    print("-" * 78)
    casos = [(6.22, 28.5), (6.22, 0.0), (2.0, 15.0), (4.0, 40.0), (1.5, 30.0),
             (8.0, 10.0), (11.0, 20.0), (11.9, 28.5), (11.9, 5.0)]
    for R, di_deg in casos:
        di = np.radians(di_deg)
        env = Transfer3DEnv()
        env.reset(options={"R": R, "di": di})
        _, rec, _, _, info = env.step(accion_optima(R, di))
        print(f"  {R:6.2f} {di_deg:6.1f} | {info['err_geom'] * 100:8.3f}% "
              f"{info['i_f_deg']:7.2f} {di_deg:7.1f} {info['dv_total']:8.4f} "
              f"{str(info['exito']):>6}")
    print("=" * 78)
    print("  Si err_geom ~ 0 e i_final = di_obj -> la geometria del entorno es correcta.")
